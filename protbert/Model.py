import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
import datetime  # For timestamping logs
from scipy.stats import pearsonr, spearmanr  # Import pearsonr for accuracy metric
import os

import wandb  # Import Weights & Biases


# Configuration
class Config:
    project_name = "protein-mutation-prediction-protbert"
    pretrained_model = "Rostlab/prot_bert"
    wandb_token = ""
    max_length = 512
    batch_size = 48
    learning_rate = 5e-5
    hidden_dropout_prob = 0.1
    epochs = 15
    freeze_layers = 2
    early_stopping_patience = 2
    early_stopping_delta = 0.001
    step_validation = 1500
    base_dir = "./"


# Custom Dataset with concatenated sequences
class ProteinMutationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if "[SEP]" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["[SEP]"])
            print("INFO: Token [SEP] was added to tokenizer")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wt_seq = ' '.join(list(row['fragment_255_org']))
        mut_aa = ' '.join(list(row['fragment_255_mut']))
        sep_token = self.tokenizer.sep_token
        combined_seq = f"{wt_seq} {sep_token} {mut_aa}"

        encoding = self.tokenizer(
            combined_seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(row['target'], dtype=torch.float)
        }


# Single BERT model with separation token
class SingleBertPredictor(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.tokenizer = tokenizer

        if len(tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(tokenizer))

        self.regressor_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        # Získání výstupů z BERTu je v pořádku
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)

        # --- Vektorizovaná logika pro nalezení a zpracování reprezentací ---

        # 1. Najdeme pozice [SEP] tokenů pro celou dávku najednou
        sep_token_id = self.tokenizer.sep_token_id
        sep_mask = (input_ids == sep_token_id)

        # argmax je trik, jak najít první výskyt. Pro jistotu převedeme na float.
        sep_indices = torch.argmax(sep_mask.float(), dim=1)

        # Pojistka: pokud v nějakém vzorku není [SEP], argmax vrátí 0.
        # V takovém případě použijeme jako fallback střed sekvence.
        no_sep_found = (sep_indices == 0) & ~sep_mask[:, 0]
        mid_points = torch.full_like(sep_indices, input_ids.size(1) // 2)
        sep_indices[no_sep_found] = mid_points[no_sep_found]

        # 2. Vytvoříme masky pro wild-type (WT) a mutantní (MUT) části
        seq_len = input_ids.size(1)
        arange_mask = torch.arange(seq_len, device=input_ids.device)[None, :].expand(input_ids.size(0), -1)

        # Maska pro WT: od pozice 1 (za [CLS]) až po [SEP]
        wt_padding_mask = (arange_mask > 0) & (arange_mask < sep_indices.unsqueeze(1))
        # Maska pro MUT: od pozice za [SEP] až do konce
        mut_padding_mask = (arange_mask > sep_indices.unsqueeze(1))

        # Zkombinujeme s původní attention_mask, abychom ignorovali padding
        wt_mask = wt_padding_mask & attention_mask.bool()
        mut_mask = mut_padding_mask & attention_mask.bool()

        # 3. Provedeme "masked average pooling"
        # Rozšíříme masky, aby měly stejnou dimenzi jako sequence_output
        wt_mask_expanded = wt_mask.unsqueeze(-1).expand_as(sequence_output)
        mut_mask_expanded = mut_mask.unsqueeze(-1).expand_as(sequence_output)

        # Vynulujeme hodnoty, které nechceme (kde je maska False)
        wt_sum = (sequence_output * wt_mask_expanded).sum(dim=1)
        mut_sum = (sequence_output * mut_mask_expanded).sum(dim=1)

        # Spočítáme počet platných tokenů pro průměrování (musíme se vyhnout dělení nulou)
        wt_count = wt_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mut_count = mut_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        wt_repr = wt_sum / wt_count
        mut_repr = mut_sum / mut_count

        # 4. Spojíme reprezentace a pošleme je do regresní hlavy
        combined_embeddings = torch.cat([wt_repr, mut_repr], dim=1)
        return self.regressor_head(combined_embeddings).squeeze()


# Loss function
def simple_mse(preds, targets, _):
    loss = (preds - targets) ** 2
    return loss.mean()


def loss_bert(preds, targets, sigma):
    return simple_mse(preds, targets, sigma)


def wand_init(config):
    wandb.login(key=config.wandb_token)
    run = wandb.init(
        project=config.project_name,
        config={
            "model": config.pretrained_model,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "dropout": config.hidden_dropout_prob,
            "epochs": config.epochs,
            "freeze_layers": config.freeze_layers,
            "architecture": "BERT_with_Deep_FC_Head",
            "dataset": "protein_fitness",
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_delta": config.early_stopping_delta
        }
    )
    return run


def calculate_and_log_metrics(val_preds, val_targets, total_val_loss, duration, log_prefix, step):
    """
    Calculates a comprehensive set of metrics, logs them to wandb, and prints a summary.
    """
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    avg_val_loss = total_val_loss / len(val_targets)

    # --- Core Metrics ---
    val_pearson_corr, pearson_p_value = pearsonr(val_preds, val_targets)
    val_spearman_corr, spearman_p_value = spearmanr(val_preds, val_targets)
    mae = mean_absolute_error(val_targets, val_preds)
    mse = mean_squared_error(val_targets, val_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(val_targets, val_preds)

    # --- Advanced Error Analysis ---
    max_error = np.max(np.abs(val_preds - val_targets))
    median_abs_error = np.median(np.abs(val_preds - val_targets))
    explained_variance = 1 - (np.var(val_preds - val_targets) / np.var(val_targets))

    # --- Residuals Analysis ---
    residuals = val_preds - val_targets
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    # --- Percentage Error (handle division by zero) ---
    mape = None
    if np.all(val_targets != 0):
        mape = mean_absolute_percentage_error(val_targets, val_preds) * 100

    # --- Logging to wandb ---
    metrics_log = {
        f"{log_prefix}/avg_val_loss": avg_val_loss,
        f"{log_prefix}/pearson_corr": val_pearson_corr,
        f"{log_prefix}/pearson_p_value": pearson_p_value,
        f"{log_prefix}/spearman_corr": val_spearman_corr,
        f"{log_prefix}/spearman_p_value": spearman_p_value,
        f"{log_prefix}/mae": mae,
        f"{log_prefix}/mse": mse,
        f"{log_prefix}/rmse": rmse,
        f"{log_prefix}/r2_score": r2,
        f"{log_prefix}/max_error": max_error,
        f"{log_prefix}/median_abs_error": median_abs_error,
        f"{log_prefix}/explained_variance": explained_variance,
        f"{log_prefix}/residual_mean": residual_mean,
        f"{log_prefix}/residual_std": residual_std,
        f"{log_prefix}/duration_seconds": duration.total_seconds(),
        "step": step
    }
    if mape is not None:
        metrics_log[f"{log_prefix}/mape"] = mape

    wandb.log(metrics_log)

    # --- Print Summary ---
    print(f"\n--- {log_prefix.replace('_', ' ').title()} Results ---")
    print(
        f"  Avg Loss: {avg_val_loss:.4f} | Pearson: {val_pearson_corr:.4f} | Spearman: {val_spearman_corr:.4f} | R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | Duration: {duration}")
    print("--------------------------------------------------")

    return avg_val_loss


def run_validation_step(model, validation_df, tokenizer, config, global_step):
    start_time = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_dataset = ProteinMutationDataset(validation_df, tokenizer, config.max_length)
    # Je důležité nastavit shuffle=False, aby se zachovalo pořadí dat pro správné spojení výsledků
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    total_val_loss = 0
    val_preds = []
    val_targets = []

    # Seznam pro ukládání dílčích DataFrame pro každou dávku
    results_dfs = []
    processed_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['targets'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = loss_bert(predictions, target, None)

            batch_size = len(target)
            total_val_loss += loss.item() * batch_size

            # Převedení predikcí a skutečných hodnot na CPU a do numpy pole
            batch_preds = np.atleast_1d(predictions.cpu().numpy())
            batch_targets = np.atleast_1d(target.cpu().numpy())

            val_preds.extend(batch_preds)
            val_targets.extend(batch_targets)

            # Získání odpovídající části původního DataFrame
            start_index = processed_samples
            end_index = start_index + batch_size
            batch_df = validation_df.iloc[start_index:end_index].copy()

            batch_df['predicted_fitness'] = batch_preds
            batch_df['actual_fitness'] = batch_targets

            results_dfs.append(batch_df)
            processed_samples += batch_size

    # Spojení všech dílčích DataFrame do jednoho výsledného
    results_df = pd.concat(results_dfs, ignore_index=True)

    log_interactive_dataframe_to_wandb(results_df, table_name=f"validation_results")

    duration = datetime.datetime.now() - start_time
    calculate_and_log_metrics(val_preds, val_targets, total_val_loss, duration, "validation_step",
                              global_step)  # Set the model back to training mode


def train_single_model(train_df, testing_df, config):
    run = wand_init(config)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
    model = SingleBertPredictor(config, tokenizer)
    wandb.watch(model, log="all", log_freq=100)

    print("Preparing data...")
    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    testing_dataset = ProteinMutationDataset(testing_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    testing_loader = DataLoader(testing_dataset, batch_size=config.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_steps = len(train_loader) * config.epochs
    print(f"Training will run for {total_steps} steps in total "
          f"({len(train_loader)} steps per epoch × {config.epochs} epochs).")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    global_step = 0
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        train_preds = []
        train_targets = []

        train_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                         desc=f"Epoch {epoch + 1}/{config.epochs} [Train]", leave=True)
        for step, batch in train_bar:

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fitness = batch['targets'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = loss_bert(predictions, fitness, None)
            loss.backward()

            total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            optimizer.step()

            global_step += 1

            if global_step > 0 and global_step % config.step_validation == 0:
                testing_subset_df = testing_df.sample(frac=0.1)
                model.eval()
                run_validation_step(model, testing_subset_df, tokenizer, config, global_step)
                model.train()

            total_train_loss += loss.item()
            train_preds.extend(predictions.detach().cpu().numpy())
            train_targets.extend(fitness.detach().cpu().numpy())

            wandb.log({
                "batch_loss": loss.item(),
                "gradient_norm": total_norm,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "step": global_step
            })
            train_bar.set_postfix(loss=loss.item())

        # --- End of Epoch Training Metrics ---
        avg_train_loss = total_train_loss / len(train_loader)
        train_pearson_corr, _ = pearsonr(train_preds, train_targets)
        wandb.log({
            "epoch_train/avg_loss": avg_train_loss,
            "epoch_train/pearson_corr": train_pearson_corr,
            "epoch": epoch + 1
        })
        print(
            f"\nEpoch {epoch + 1}/{config.epochs} Training Summary: Loss: {avg_train_loss:.4f}, Pearson: {train_pearson_corr:.4f}")

        # --- End of Epoch Validation on Whole Dataset ---
        val_start_time = datetime.datetime.now()
        model.eval()
        val_bar = tqdm(testing_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Full Val]", leave=True)

        total_val_loss = 0
        val_preds = []
        val_targets = []

        results_dfs = []
        processed_samples = 0

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fitness = batch['targets'].to(device)

                predictions = model(input_ids, attention_mask)
                loss = loss_bert(predictions, fitness, None)
                total_val_loss += loss.item() * len(fitness)  # Weighted by batch size

                batch_size = len(fitness)
                total_val_loss += loss.item() * batch_size

                # Převedení predikcí a skutečných hodnot na CPU a do numpy pole
                batch_preds = np.atleast_1d(predictions.cpu().numpy())
                batch_targets = np.atleast_1d(fitness.cpu().numpy())

                val_preds.extend(batch_preds)
                val_targets.extend(batch_targets)

                # Získání odpovídající části původního DataFrame
                start_index = processed_samples
                end_index = start_index + batch_size
                batch_df = testing_df.iloc[start_index:end_index].copy()

                batch_df['predicted_fitness'] = batch_preds
                batch_df['actual_fitness'] = batch_targets

                results_dfs.append(batch_df)
                processed_samples += batch_size

            # Spojení všech dílčích DataFrame do jednoho výsledného
        results_df = pd.concat(results_dfs, ignore_index=True)

        log_interactive_dataframe_to_wandb(results_df, table_name=f"epoch_val")

        val_duration = datetime.datetime.now() - val_start_time
        avg_val_loss = calculate_and_log_metrics(val_preds, val_targets, total_val_loss, val_duration, "epoch_val",
                                                 global_step)

        scheduler.step(avg_val_loss)

        print(f"Best Val Loss so far: {best_val_loss:.4f}")
        if avg_val_loss < best_val_loss - config.early_stopping_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_filename = f'best_model_epoch_{epoch + 1}.pth'
            best_model_path = os.path.join(run.dir, best_model_filename)
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path, policy="now")
            print(f"  >>> New best model found! Saved as {best_model_filename} with val loss {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  ! No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    wandb.finish()
    return model


def run_validation_worker(config, model, df):
    """
    Worker function that runs the parallel CPU validation and puts the result into a queue.
    This function is the target for our asynchronous multiprocessing.Process.
    """

    # selectin just random 20% of the dataframe for quick validation
    selected_values = df.sample(frac=config.val_fract_size, random_state=42).reset_index(drop=True)

    print(f"\n--- [Async Val] Process started. Loading model from GPU ---")
    results = {}

    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 60)

    print(f"📊 BASIC METRICS:")
    print(f"   Model Loss (avg):        {results['avg_val_loss']:.6f}")
    print(f"   Validation Duration:     {results['validation_duration']}")
    print(f"   Number of Samples:       {results['n_samples']}")

    print(f"\n📈 CORRELATION METRICS:")
    print(f"   Pearson Correlation:     {results['val_pearson_corr']:.6f} (p={results['pearson_p_value']:.2e})")
    print(f"   Spearman Correlation:    {results['val_spearman_corr']:.6f} (p={results['spearman_p_value']:.2e})")

    print(f"\n📏 ERROR METRICS:")
    print(f"   MAE (Mean Abs Error):    {results['mae']:.6f}")
    print(f"   MSE (Mean Sq Error):     {results['mse']:.6f}")
    print(f"   RMSE (Root Mean Sq Err): {results['rmse']:.6f}")
    print(f"   Max Absolute Error:      {results['max_error']:.6f}")
    print(f"   Median Absolute Error:   {results['median_abs_error']:.6f}")
    if results['mape'] is not None:
        print(f"   MAPE (Mean Abs % Error): {results['mape']:.2f}%")

    print(f"\n📊 GOODNESS OF FIT:")
    print(f"   R² (R-squared):          {results['r2_score']:.6f}")
    print(f"   Explained Variance:      {results['explained_variance']:.6f}")

    print(f"\n📉 RESIDUAL ANALYSIS:")
    print(f"   Residual Mean (bias):    {results['residual_mean']:.6f}")
    print(f"   Residual Std Dev:        {results['residual_std']:.6f}")

    print(
        f" Duration: {results['duration']} for {results["n_samples"]}, so with speed of {results['duration'] / results['n_samples']} per sample, Used batch size of {config.batch_size}")

    print("=" * 60)

    # Interpretation guide
    print(f"\n💡 QUICK INTERPRETATION:")
    val_pearson_corr = results['val_pearson_corr']
    r2 = results['r2_score']

    if val_pearson_corr > 0.8:
        print(f"   🟢 Strong linear correlation ({val_pearson_corr:.3f})")
    elif val_pearson_corr > 0.6:
        print(f"   🟡 Moderate linear correlation ({val_pearson_corr:.3f})")
    else:
        print(f"   🔴 Weak linear correlation ({val_pearson_corr:.3f})")

    if r2 > 0.8:
        print(f"   🟢 Excellent fit (R²={r2:.3f}) - model explains {r2 * 100:.1f}% of variance")
    elif r2 > 0.6:
        print(f"   🟡 Good fit (R²={r2:.3f}) - model explains {r2 * 100:.1f}% of variance")
    else:
        print(f"   🔴 Poor fit (R²={r2:.3f}) - model explains {r2 * 100:.1f}% of variance")

    print(f"--- [Async Val] Process finished. Results sent back. ---")


def log_interactive_dataframe_to_wandb(
        df: pd.DataFrame,
        table_name: str = "validation_results_js_interactive"
):
    """
    Generuje a loguje plně interaktivní, bezserverový HTML report.
    Obsahuje strukturálně opravenou a vizuálně vylepšenou matici záměn
    a tlačítko pro stažení dat jako CSV.
    """

    # --- 1. Příprava dat (beze změny) ---
    df = df.copy()
    if 'target' in df.columns and 'actual_fitness' not in df.columns:
        df.rename(columns={'target': 'actual_fitness'}, inplace=True)
    required_cols = ['actual_fitness', 'predicted_fitness']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CHYBA: V DataFrame chybí klíčové sloupce: {required_cols}.")
    if 'error' not in df.columns:
        df['error'] = (df['predicted_fitness'] - df['actual_fitness']).abs()
    POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD = 0.05, -0.05

    def classify_mutation(value):
        if pd.isna(value): return 'N/A'
        if value > POSITIVE_THRESHOLD: return 'Positive'
        if value < NEGATIVE_THRESHOLD: return 'Negative'
        return 'Neutral'

    df['actual_class'] = df['actual_fitness'].apply(classify_mutation)
    df['predicted_class'] = df['predicted_fitness'].apply(classify_mutation)
    df_clean = df.where(pd.notnull(df), None)
    df_json = df_clean.to_json(orient='records')

    # --- 2. Vytvoření HTML z finálně opravené šablony ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="cs">
    <head>
        <meta charset="UTF-8">
        <title>Interaktivní analýza výsledků</title>
        <script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine-dark.css" />
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                padding: 20px; background-color: #121212; color: #e0e0e0;
            }}
            .header-container {{
                display: flex; justify-content: space-between; align-items: center;
                border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 20px;
            }}
            h1 {{ color: #ffffff; margin: 0; }}
            h2 {{ color: #ffffff; }}

            #download-btn {{
                padding: 8px 15px; font-size: 14px; background-color: #4e9af1;
                color: white; border: none; border-radius: 5px; cursor: pointer;
                transition: background-color 0.2s;
            }}
            #download-btn:hover {{ background-color: #3a75c4; }}

            .metric-container {{
                display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px; margin-bottom: 30px;
            }}
            .metric-card {{
                padding: 15px; border: 1px solid #333; border-radius: 8px;
                background-color: #1e1e1e;
            }}
            .metric-title {{ font-weight: bold; color: #aaa; display: block; margin-bottom: 5px; }}
            .metric-value {{ font-size: 1.4em; color: #4e9af1; font-family: monospace; }}

            .confusion-matrix-container {{ margin-bottom: 30px; }}
            .confusion-matrix {{
                display: grid; grid-template-columns: 1.5fr repeat(3, 1fr); gap: 5px;
                text-align: center; max-width: 700px; background-color: #1e1e1e;
                padding: 15px; border-radius: 8px; border: 1px solid #333;
            }}
            .cm-cell {{
                padding: 12px; font-size: 1.1em; font-family: sans-serif;
                display: flex; align-items: center; justify-content: center;
                border-radius: 5px;
            }}
            .cm-header {{ background-color: #2a2a2a; font-weight: bold; }}
            .cm-axis-label {{ font-weight: bold; text-align: right; justify-content: flex-end; padding-right: 15px; }}
            .cm-data {{ font-family: monospace; font-size: 1.3em; }}
            .cm-correct {{ background-color: #1a5325; color: #a3e0b2; }}
            .cm-error {{ background-color: #6d1c23; color: #f5b5bc; }}
            .cm-empty {{ background-color: transparent; }}
            #myGrid {{ height: 600px; width: 100%; }}
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>Interaktivní analýza výsledků</h1>
            <button id="download-btn">Stáhnout filtrovaná data (CSV)</button>
        </div>

        <div class="metric-container">
            <!-- Metriky zůstávají stejné -->
            <div class="metric-card"><span class="metric-title">Zobrazeno vzorků:</span> <span id="samples-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">Platných pro výpočet:</span> <span id="valid-samples-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">MSE:</span> <span id="mse-value" class-value">-</span></div>
            <div class="metric-card"><span class="metric-title">MAE:</span> <span id="mae-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">RMSE:</span> <span id="rmse-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">R²:</span> <span id="r2-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">Pearson Corr:</span> <span id="pearson-value" class="metric-value">-</span></div>
        </div>

        <!-- === SKUTEČNĚ OPRAVENÁ HTML STRUKTURA MATICE ZÁMĚN === -->
        <div class="confusion-matrix-container">
            <h2>Matice záměn (dle filtru)</h2>
            <div class="confusion-matrix">
                <!-- Řádek 1: Hlavní nadpis pro sloupce -->
                <div class="cm-cell cm-empty"></div>
                <!-- OPRAVA ZDE: Použití 'grid-column' místo neplatného 'colspan' -->
                <div class="cm-cell cm-header" style="grid-column: 2 / 5;"><b>Skutečná třída (Actual)</b></div>

                <!-- Řádek 2: Konkrétní nadpisy sloupců -->
                <div class="cm-cell cm-empty"></div>
                <div class="cm-cell cm-header">Pozitivní</div>
                <div class="cm-cell cm-header">Neutrální</div>
                <div class="cm-cell cm-header">Negativní</div>

                <!-- Řádek 3: Predikce Pozitivní -->
                <div class="cm-cell cm-axis-label">Predikovaná: Pozitivní</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_pos-act_pos">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_pos-act_neu">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_pos-act_neg">0</div>

                <!-- Řádek 4: Predikce Neutrální -->
                <div class="cm-cell cm-axis-label">Predikovaná: Neutrální</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neu-act_pos">0</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_neu-act_neu">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neu-act_neg">0</div>

                <!-- Řádek 5: Predikce Negativní -->
                <div class="cm-cell cm-axis-label">Predikovaná: Negativní</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neg-act_pos">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neg-act_neu">0</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_neg-act_neg">0</div>
            </div>
        </div>
        <!-- ======================================================= -->

        <div id="myGrid" class="ag-theme-alpine-dark"></div>

        <script>
            // JavaScript zůstává beze změny, chyba byla čistě v HTML struktuře
            let gridApi;
            const rowData = {df_json};
            const columnDefs = Object.keys(rowData[0] || {{}}).map(key => ({{ field: key, sortable: true, resizable: true, filter: typeof rowData[0][key] === 'number' ? 'agNumberColumnFilter' : 'agTextColumnFilter' }}));
            const gridOptions = {{ columnDefs, rowData, defaultColDef: {{ flex: 1, minWidth: 150, filter: true, sortable: true, resizable: true, floatingFilter: true }}, onGridReady: (params) => {{ gridApi = params.api; }}, onFirstDataRendered: (params) => updateDashboard(params.api), onFilterChanged: (params) => updateDashboard(params.api) }};

            function downloadCSV(api) {{
                if (!api) return;
                const currentData = [];
                api.forEachNodeAfterFilter(node => currentData.push(node.data));
                if (currentData.length === 0) {{ alert("Nenalezena žádná data k exportu."); return; }}
                const headers = Object.keys(currentData[0]);
                const csvHeader = headers.join(',');
                const csvRows = currentData.map(row => headers.map(header => {{
                    let value = row[header];
                    if (value === null || value === undefined) return '';
                    let strValue = String(value);
                    if (strValue.includes(',') || strValue.includes('"') || strValue.includes('\\n')) {{
                        return `"${{strValue.replace(/"/g, '""')}}"`;
                    }}
                    return strValue;
                }}).join(','));
                const csvContent = [csvHeader, ...csvRows].join('\\n');
                const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                const link = document.createElement("a");
                const url = URL.createObjectURL(blob);
                link.setAttribute("href", url); link.setAttribute("download", "filtered_data.csv");
                link.style.visibility = 'hidden'; document.body.appendChild(link);
                link.click(); document.body.removeChild(link);
            }}
            function updateConfusionMatrix(data) {{
                const counts = {{ 'Positive': {{'Positive':0,'Neutral':0,'Negative':0}}, 'Neutral':{{'Positive':0,'Neutral':0,'Negative':0}}, 'Negative':{{'Positive':0,'Neutral':0,'Negative':0}} }};
                data.forEach(row => {{
                    const actual = row.actual_class, predicted = row.predicted_class;
                    if (counts[predicted] && counts[predicted][actual] !== undefined) counts[predicted][actual]++;
                }});
                document.getElementById('cm-pred_pos-act_pos').innerText = counts.Positive.Positive;
                document.getElementById('cm-pred_pos-act_neu').innerText = counts.Positive.Neutral;
                document.getElementById('cm-pred_pos-act_neg').innerText = counts.Positive.Negative;
                document.getElementById('cm-pred_neu-act_pos').innerText = counts.Neutral.Positive;
                document.getElementById('cm-pred_neu-act_neu').innerText = counts.Neutral.Neutral;
                document.getElementById('cm-pred_neu-act_neg').innerText = counts.Neutral.Negative;
                document.getElementById('cm-pred_neg-act_pos').innerText = counts.Negative.Positive;
                document.getElementById('cm-pred_neg-act_neu').innerText = counts.Negative.Neutral;
                document.getElementById('cm-pred_neg-act_neg').innerText = counts.Negative.Negative;
            }}
            function calculateMetrics(data) {{
                if (!data) return {{ samples: 0, valid: 0 }};
                const validPairs = data.map(row => ({{ t: parseFloat(row.actual_fitness), p: parseFloat(row.predicted_fitness) }})).filter(pair => !isNaN(pair.t) && !isNaN(pair.p));
                const n = validPairs.length; if (n < 2) return {{ samples: data.length, valid: n }};
                let sum_sq_err=0, sum_abs_err=0, sum_true=0, total_sum_sq=0, sum_xy=0, sum_x=0, sum_y=0, sum_x2=0, sum_y2=0;
                validPairs.forEach(pair => {{ sum_true += pair.t; sum_x += pair.t; sum_y += pair.p; sum_xy += pair.t * pair.p; sum_x2 += pair.t * pair.t; sum_y2 += pair.p * pair.p; }});
                const mean_true = sum_true / n;
                validPairs.forEach(pair => {{ sum_sq_err += (pair.p - pair.t)**2; sum_abs_err += Math.abs(pair.p - pair.t); total_sum_sq += (pair.t - mean_true)**2; }});
                const mse = sum_sq_err / n; const r2 = total_sum_sq < 1e-9 ? 1 : 1 - (sum_sq_err / total_sum_sq);
                const num = n * sum_xy - sum_x * sum_y; const den = Math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2));
                const pearson = den < 1e-9 ? 0 : num / den;
                return {{ samples: data.length, valid: n, mse, mae: sum_abs_err / n, rmse: Math.sqrt(mse), r2, pearson }};
            }}
            function updateMetrics(data) {{
                const metrics = calculateMetrics(data);
                document.getElementById('samples-value').innerText = metrics.samples;
                document.getElementById('valid-samples-value').innerText = metrics.valid;
                ['mse', 'mae', 'rmse', 'r2', 'pearson'].forEach(key => {{
                    const el = document.getElementById(key + '-value');
                    if (el) {{ const value = metrics[key]; el.innerText = (value === undefined || isNaN(value)) ? '-' : value.toFixed(6); }}
                }});
            }}
            function updateDashboard(api) {{
                const currentData = [];
                if (api) {{ api.forEachNodeAfterFilter(node => currentData.push(node.data)); }}
                updateMetrics(currentData);
                updateConfusionMatrix(currentData);
            }}
            document.addEventListener('DOMContentLoaded', () => {{
                const gridDiv = document.querySelector('#myGrid');
                agGrid.createGrid(gridDiv, gridOptions);
                document.getElementById('download-btn').addEventListener('click', () => downloadCSV(gridApi));
            }});
        </script>
    </body>
    </html>
    """

    # --- 3. Logování do W&B ---
    wandb.log({table_name: wandb.Html(html_content)})
    print(f"Interaktivní report '{table_name}' byl úspěšně zalogován do W&B.")
