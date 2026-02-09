import argparse
import polars as pl
import torch
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, f1_score, matthews_corrcoef, accuracy_score

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Předpokládám, že importy z vašeho projektu fungují
from protbert.ModelComposter import (
    prepare_data_dynamic,
    Config,
    ProteinMutationDataset,
    ComposerProteinModel
)


def calculate_metrics(df: pl.DataFrame, output_dir: str):
    """
    Vypočítá Pearson, MSE, Pos F1, Binární MCC a Ternární MCC.
    Očekává sloupce 'fitness' (ground truth) a 'predicted_fitness'.
    """
    print("\n--- Počítám statistiky ---")

    # Kontrola, zda máme ground truth (jestli to není jen dummy 0.0)
    # Pokud je rozptyl nulový, pravděpodobně jde o dummy data
    if df["fitness"].std() == 0:
        print("VAROVÁNÍ: Sloupec 'fitness' má nulový rozptyl (asi dummy data). Statistiky nebudou relevantní.")
        return

    y_true = df["fitness"].to_numpy()
    y_pred = df["predicted_fitness"].to_numpy()

    # 1. Regresní metriky
    # Pearson
    pearson_corr, _ = pearsonr(y_true, y_pred)
    # MSE
    mse = mean_squared_error(y_true, y_pred)

    # 2. Binární Klasifikace (Threshold 0.0 -> Zlepšuje vs Nezlepšuje)
    # 1 = Positive (fitness > 0), 0 = Negative/Neutral (fitness <= 0)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    pos_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)
    binary_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    # 3. Ternární Klasifikace (Boundaries -0.15, 0.15)
    # 0 = Negative (< -0.15)
    # 1 = Neutral  (>= -0.15 a <= 0.15)
    # 2 = Positive (> 0.15)

    def to_ternary(values):
        bins = []
        for v in values:
            if v < -0.15:
                bins.append(0)  # Negative
            elif v > 0.15:
                bins.append(2)  # Positive
            else:
                bins.append(1)  # Neutral
        return np.array(bins)

    y_true_ter = to_ternary(y_true)
    y_pred_ter = to_ternary(y_pred)

    ternary_mcc = matthews_corrcoef(y_true_ter, y_pred_ter)
    ternary_acc = accuracy_score(y_true_ter, y_pred_ter)

    # --- Výpis a Uložení ---
    stats_output = (
        f"--- VÝSLEDKY INFERENCE ---\n"
        f"Pearson Correlation: {pearson_corr:.4f}\n"
        f"MSE:                 {mse:.4f}\n"
        f"--------------------------\n"
        f"Binary (Thresh > 0.0)\n"
        f"  Pos F1 Score:      {pos_f1:.4f}\n"
        f"  Binary MCC:        {binary_mcc:.4f}\n"
        f"--------------------------\n"
        f"Ternary (Bounds -0.15, 0.15)\n"
        f"  Ternary MCC:       {ternary_mcc:.4f}\n"
        f"  Ternary Accuracy:  {ternary_acc:.4f}\n"
    )

    print(stats_output)

    # Uložení do souboru metrics.txt
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(stats_output)
    print(f"Statistiky uloženy do: {metrics_path}")


def run_inference(input_file, checkpoint_path, output_file, batch_size, device):
    print(f"--- Spouštím inferenci na {device} ---")

    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)

    # Načtení dat
    if input_file.endswith(".parquet"):
        df_raw = pl.read_parquet(input_file)
    else:
        df_raw = pl.read_csv(input_file)

    # Dummy target pokud chybí
    if "fitness" not in df_raw.columns:
        if "target" in df_raw.columns:
            df_raw = df_raw.rename({"target": "fitness"})
        else:
            df_raw = df_raw.with_columns(pl.lit(0.0).alias("fitness"))

    processed_df = prepare_data_dynamic(
        df_raw,
        max_total_length=config.max_length,
        window_size=config.seq_window_size
    )

    dataset = ProteinMutationDataset(processed_df, tokenizer, max_length=config.max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ComposerProteinModel(config.pretrained_model, tokenizer)
    model.to(device)

    print(f"Načítám checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state']['model'] if 'state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []

    print("Provádím predikci...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device)
            }
            outputs = model(inputs)
            preds = outputs.squeeze().cpu().numpy().tolist()
            if isinstance(preds, float): preds = [preds]
            predictions.extend(preds)

    # Spojení výsledků
    # Pokud by délka neseděla (kvůli drop_last nebo chybě), ořízneme nebo doplníme
    final_df = processed_df.with_columns(pl.Series("predicted_fitness", predictions))

    # Vytvoření složky a uložení
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_df.write_csv(output_file)
    print(f"Predikce uloženy do: {output_file}")

    # --- VÝPOČET STATISTIK ---
    calculate_metrics(final_df, output_dir if output_dir else ".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(args.input, args.checkpoint, args.output, args.batch_size, device)