import argparse
import polars as pl
import torch
import os
import numpy as np
import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, f1_score, matthews_corrcoef, accuracy_score

from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports from the project
from protbert.ModelComposter import (
    prepare_data_dynamic as prepare_data_protbert,
    Config as ConfigProtbert,
    ProteinMutationDataset as DatasetProtbert,
    ComposerProteinModel
)

from protbert.ModelComposerESM import (
    prepare_data_dynamic as prepare_data_esm,
    ConfigESM,
    ProteinMutationDataset as DatasetESM,
    ComposerProteinModelESM
)


def calculate_metrics(df: pl.DataFrame, output_dir: str, metadata: dict = None):
    """
    Calculates Pearson correlation, MSE, Positive F1, Binary MCC, and Ternary MCC.
    Expects columns 'fitness' (ground truth) and 'predicted_fitness'.
    """
    print("\n--- Calculating Statistics ---")

    # Check if we have ground truth (not just dummy 0.0)
    has_ground_truth = df["fitness"].std() != 0
    
    if not has_ground_truth:
        print("WARNING: 'fitness' column has zero variance (likely dummy data). Statistics will not be relevant.")

    pearson_corr = 0.0
    mse = 0.0
    pos_f1 = 0.0
    binary_mcc = 0.0
    ternary_mcc = 0.0
    ternary_acc = 0.0

    if has_ground_truth:
        y_true = df["fitness"].to_numpy()
        y_pred = df["predicted_fitness"].to_numpy()

        # 1. Regression Metrics
        pearson_corr, _ = pearsonr(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        # 2. Binary Classification (Threshold 0.0 -> Improves vs Not Improves)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)

        pos_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)
        binary_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

        # 3. Ternary Classification (Boundaries -0.15, 0.15)
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

    # --- Prepare Metadata Header ---
    header = "--- INFERENCE EXECUTION DETAILS ---\n"
    if metadata:
        for key, value in metadata.items():
            header += f"{key.replace('_', ' ').title()}: {value}\n"
    header += f"Execution Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "-----------------------------------\n"

    # --- Prepare Stats Output ---
    stats_output = ""
    if has_ground_truth:
        stats_output = (
            f"--- INFERENCE RESULTS ---\n"
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
    else:
        stats_output = "No ground truth available for metrics calculation.\n"

    full_output = header + stats_output
    print(full_output)

    # Save to metrics.txt
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(full_output)
    print(f"Statistics and metadata saved to: {metrics_path}")


def run_inference(input_file, checkpoint_path, output_file, batch_size, device, model_type):
    print(f"--- Starting {model_type} inference on {device} ---")
    
    metadata = {
        "model_type": model_type,
        "checkpoint": os.path.abspath(checkpoint_path),
        "input_dataset": os.path.abspath(input_file),
        "device": device,
        "batch_size": batch_size
    }

    # Configuration based on model type
    if model_type.lower() == "esm":
        config = ConfigESM()
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        prep_fn = prepare_data_esm
        dataset_cls = DatasetESM
        model_cls = ComposerProteinModelESM
    else:
        config = ConfigProtbert()
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
        prep_fn = prepare_data_protbert
        dataset_cls = DatasetProtbert
        model_cls = ComposerProteinModel

    # Load data
    if input_file.endswith(".parquet"):
        df_raw = pl.read_parquet(input_file)
    else:
        df_raw = pl.read_csv(input_file)

    # Check for target column
    if "fitness" not in df_raw.columns:
        if "target" in df_raw.columns:
            df_raw = df_raw.rename({"target": "fitness"})
        else:
            df_raw = df_raw.with_columns(pl.lit(0.0).alias("fitness"))

    # Prepare data for model
    processed_df = prep_fn(
        df_raw,
        max_total_length=config.max_length,
        window_size=config.seq_window_size
    )

    dataset = dataset_cls(processed_df, tokenizer, max_length=config.max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = model_cls(config.pretrained_model, tokenizer)
    model.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict (handle different saving formats)
    if 'state' in checkpoint and 'model' in checkpoint['state']:
        state_dict = checkpoint['state']['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []

    print("Predicting...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if model_type.lower() == "esm":
                inputs = {
                    'input_ids_wt': batch['input_ids_wt'].to(device),
                    'attention_mask_wt': batch['attention_mask_wt'].to(device),
                    'input_ids_mut': batch['input_ids_mut'].to(device),
                    'attention_mask_mut': batch['attention_mask_mut'].to(device),
                }
            else:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'token_type_ids': batch['token_type_ids'].to(device)
                }
            
            outputs = model(inputs)
            preds = outputs.squeeze().cpu().numpy().tolist()
            if isinstance(preds, float): preds = [preds]
            predictions.extend(preds)

    # Merge results
    final_df = processed_df.with_columns(pl.Series("predicted_fitness", predictions))

    # Save to file
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_df.write_csv(output_file)
    print(f"Predictions saved to: {output_file}")

    # Calculate statistics and display metadata
    calculate_metrics(final_df, output_dir if output_dir else ".", metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dataset path (.csv or .parquet)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--model_type", type=str, default="protbert", choices=["protbert", "esm"], help="Model type: protbert or esm")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(args.input, args.checkpoint, args.output, args.batch_size, device, args.model_type)