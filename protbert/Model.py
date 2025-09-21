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

import multiprocessing
from functools import partial
from queue import Empty

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


# Custom Dataset with concatenated sequences
class ProteinMutationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if "[SEP]" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["[SEP]"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wt_seq = row['fragment_255_org']
        mut_aa = row['fragment_255_mut']

        wt_seq = ' '.join(list(wt_seq))
        mut_aa = ' '.join(list(mut_aa))

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
            'fitness': torch.tensor(row['normalized_fitness'], dtype=torch.float),
            'fitness_sigma': torch.tensor(row['normalized_fitness_sigma'], dtype=torch.float)
        }


# Single BERT model with separation token
class SingleBertPredictor(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.tokenizer = tokenizer

        if len(tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(tokenizer))

        if config.freeze_layers > 0:
            for i in range(config.freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

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
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        sep_token_id = self.tokenizer.sep_token_id
        wt_representations = []
        mut_representations = []

        for i in range(input_ids.size(0)):
            seq_tokens = input_ids[i]

            if sep_token_id is not None:
                sep_mask = (seq_tokens == sep_token_id)
                if sep_mask.any():
                    sep_positions = sep_mask.nonzero(as_tuple=True)[0]
                else:
                    sep_positions = torch.tensor([], device=seq_tokens.device)
            else:
                sep_positions = torch.tensor([], device=seq_tokens.device)

            if len(sep_positions) == 0:
                sep_pos = seq_tokens.size(0) // 2
            else:
                sep_pos = sep_positions[0].item()

            cls_representation = sequence_output[i, 0, :]
            wt_repr = cls_representation
            mut_repr = cls_representation

            try:
                if sep_pos > 1:
                    wt_repr = sequence_output[i, 1:sep_pos, :].mean(dim=0)
                if sep_pos < sequence_output.size(1) - 1:
                    mut_repr = sequence_output[i, sep_pos + 1:, :].mean(dim=0)
            except:
                pass

            wt_representations.append(wt_repr)
            mut_representations.append(mut_repr)

        wt_embeddings = torch.stack(wt_representations)
        mut_embeddings = torch.stack(mut_representations)
        combined_embeddings = torch.cat([wt_embeddings, mut_embeddings], dim=1)

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
    """
    Runs validation on a subset of the data and logs comprehensive metrics to wandb.
    """
    start_time = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_dataset = ProteinMutationDataset(validation_df, tokenizer, config.max_length)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    total_val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fitness = batch['fitness'].to(device)
            fitness_sigma = batch['fitness_sigma'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = loss_bert(predictions, fitness, fitness_sigma)
            total_val_loss += loss.item() * len(fitness)  # Weighted by batch size

            val_preds.extend(np.atleast_1d(predictions.cpu().numpy()))
            val_targets.extend(np.atleast_1d(fitness.cpu().numpy()))

    duration = datetime.datetime.now() - start_time
    calculate_and_log_metrics(val_preds, val_targets, total_val_loss, duration, "validation_step", global_step)

    model.train()  # Set the model back to training mode


def train_single_model(train_df, testing_df, config):
    run = wand_init(config)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
    model = SingleBertPredictor(config, tokenizer)
    wandb.watch(model, log="all", log_freq=100)

    print("Preparing data...")
    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    testing_dataset = ProteinMutationDataset(testing_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
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
            if global_step > 0 and global_step % config.step_validation == 0:
                testing_subset_df = testing_df.sample(frac=0.1)
                run_validation_step(model, testing_subset_df, tokenizer, config, global_step)

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fitness = batch['fitness'].to(device)
            fitness_sigma = batch['fitness_sigma'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = loss_bert(predictions, fitness, fitness_sigma)
            loss.backward()

            total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            optimizer.step()

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
            global_step += 1

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
        total_val_loss = 0
        val_preds = []
        val_targets = []
        val_bar = tqdm(testing_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Full Val]", leave=True)

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fitness = batch['fitness'].to(device)
                fitness_sigma = batch['fitness_sigma'].to(device)

                predictions = model(input_ids, attention_mask)
                loss = loss_bert(predictions, fitness, fitness_sigma)
                total_val_loss += loss.item() * len(fitness)  # Weighted by batch size

                val_preds.extend(np.atleast_1d(predictions.cpu().numpy()))
                val_targets.extend(np.atleast_1d(fitness.cpu().numpy()))
                val_bar.set_postfix(loss=loss.item())

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
            wandb.save(best_model_path, policy="now", base_path="/models/")
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
