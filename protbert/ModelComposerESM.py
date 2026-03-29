import json
import os
from dataclasses import dataclass, asdict
import torch.distributed
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Composer imports
from composer import Trainer, Callback, State, Logger, Evaluator
from composer.algorithms import GradientClipping
from composer.callbacks import EarlyStopper, LRMonitor, OptimizerMonitor, CheckpointSaver
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from torch.utils.data import Dataset, DataLoader
from composer.utils import dist

# Torchmetrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score, \
    MeanAbsolutePercentageError, MatthewsCorrCoef, F1Score

# Transformers imports
from transformers import AutoTokenizer, EsmModel


# --- Configuration for ESM2 ---
@dataclass
class ConfigESM:
    project_name: str = "protein-mutation-prediction-esm2"
    pretrained_model: str = "facebook/esm2_t33_650M_UR50D"  # 650M parameter model
    wandb_token: str = ""
    max_length: int = 1024
    batch_size: int = 24  # Reduced batch size (ESM2 is heavier than ProtBERT)
    learning_rate: float = 2e-6
    # INCREASED NUMBER OF EPOCHS FOR EARLY STOPPING
    epochs: int = 50
    early_stopping_patience: int = 3
    # NEW PARAMETER FOR FREQ_VAL EARLY STOPPER (must be greater than unfreeze callback patience)
    early_stopping_patience_freq: int = 10
    early_stopping_delta: float = 0.001
    base_dir: str = "./"
    seq_window_size: int = 511  # ESM2 context is larger, we can use larger windows
    save_folder: str = "checkpoints_esm2"


# --- Data Preparation Helper ---
def prepare_data_dynamic(df: pl.DataFrame, max_total_length: int = 1024, window_size: int = 511):
    """
    Standard data prep: Cuts sequences around the mutation.
    """
    if "target" in df.columns and "fitness" not in df.columns:
        df = df.rename({"target": "fitness"})
    if "mutation" not in df.columns and "mut_type" in df.columns:
        df = df.rename({"mut_type": "mutation"})

    if window_size == 255 and "fragment_255_org" in df.columns and "fragment_255_mut" in df.columns:
        print("INFO: Using pre-calculated fragments (fragment_255_org/mut).")
        df_processed = df.with_columns([
            pl.col("fragment_255_org").str.replace_all("[UZOB]", "X").alias("clean_wt"),
            pl.col("fragment_255_mut").str.replace_all("[UZOB]", "X").alias("clean_mut")
        ])
        return df_processed

    print(f"INFO: Calculating fragments dynamically (Window={window_size}).")

    if "original_seq_full" in df.columns:
        df = df.rename({"original_seq_full": "wt_sequence", "mutated_seq_full": "mut_sequence"})

    if "wt_sequence" not in df.columns:
        raise ValueError(f"Dataset missing 'wt_sequence'. Columns: {df.columns}")

    def get_mutation_idx(row):
        import re
        if row['mutation']:
            match = re.search(r'\d+', str(row['mutation']))
            if match:
                return int(match.group(0)) - 1
        s1, s2 = row['wt_sequence'], row['mut_sequence']
        for i in range(min(len(s1), len(s2))):
            if s1[i] != s2[i]:
                return i
        return 0

    target_window_size = window_size
    half_window = target_window_size // 2

    df_processed = df.with_columns([
        pl.struct(["wt_sequence", "mut_sequence", "mutation"])
        .map_elements(get_mutation_idx, return_dtype=pl.Int64)
        .alias("mut_idx"),
        pl.col("wt_sequence").str.len_chars().alias("seq_len")
    ]).with_columns([
        (pl.col("mut_idx") - half_window).clip(lower_bound=0).alias("start_idx")
    ]).with_columns([
        pl.when((pl.col("start_idx") + target_window_size) > pl.col("seq_len"))
        .then((pl.col("seq_len") - target_window_size).clip(lower_bound=0))
        .otherwise(pl.col("start_idx"))
        .alias("final_start")
    ]).with_columns([
        pl.col("wt_sequence").str.slice(pl.col("final_start"), target_window_size)
        .str.replace_all("[UZOB]", "X").alias("clean_wt"),
        pl.col("mut_sequence").str.slice(pl.col("final_start"), target_window_size)
        .str.replace_all("[UZOB]", "X").alias("clean_mut")
    ])

    return df_processed


# --- 1. Dataset ---
class ProteinMutationDatasetESM(Dataset):
    def __init__(self, processed_df: pl.DataFrame, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.wt_seqs = processed_df["clean_wt"].to_list()
        self.mut_seqs = processed_df["clean_mut"].to_list()
        self.targets = processed_df["fitness"].to_list()
        self.max_length = max_length
        self.ids = list(range(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq_wt = self.wt_seqs[idx]
        seq_mut = self.mut_seqs[idx]
        target = self.targets[idx]
        row_id = self.ids[idx]

        # Tokenize separately!
        inputs_wt = self.tokenizer(seq_wt, truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt')
        inputs_mut = self.tokenizer(seq_mut, truncation=True, max_length=self.max_length, padding='max_length',
                                    return_tensors='pt')

        return {
            'input_ids_wt': inputs_wt['input_ids'].squeeze(),
            'attention_mask_wt': inputs_wt['attention_mask'].squeeze(),
            'input_ids_mut': inputs_mut['input_ids'].squeeze(),
            'attention_mask_mut': inputs_mut['attention_mask'].squeeze(),
            'labels': torch.tensor(target, dtype=torch.float),
            'row_idx': torch.tensor(row_id, dtype=torch.long)
        }


class ESMProteinMutationCore(nn.Module):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()

        # Load base ESM model
        self.esm = EsmModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = tokenizer

        # --- DDP FIX: Remove unused layers ---
        if hasattr(self.esm, 'pooler') and self.esm.pooler is not None:
            del self.esm.pooler
            self.esm.pooler = None

        if hasattr(self.esm, 'contact_head') and self.esm.contact_head is not None:
            del self.esm.contact_head
            self.esm.contact_head = None

        # Resize if necessary (added tokens)
        if len(tokenizer) > self.esm.config.vocab_size:
            self.esm.resize_token_embeddings(len(tokenizer))

        hidden_size = self.esm.config.hidden_size

        # MLP Head - input is 3x hidden_size:
        # 1. WT Mean Pooled
        # 2. MUT Mean Pooled
        # 3. Difference (MUT - WT)
        input_dim = hidden_size * 3
        self.regressor_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean Pooling - Takes attention mask into account for correct averaging
        (Ignores padding tokens)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids_wt, attention_mask_wt, input_ids_mut, attention_mask_mut):
        # 1. DDP TRICK: Combine WT and MUT into a single batch
        combined_input_ids = torch.cat([input_ids_wt, input_ids_mut], dim=0)
        combined_attention_mask = torch.cat([attention_mask_wt, attention_mask_mut], dim=0)

        # 2. SINGLE MODEL PASS
        out_combined = self.esm(input_ids=combined_input_ids, attention_mask=combined_attention_mask)

        # 3. SPLIT BACK INTO WT AND MUT
        batch_size = input_ids_wt.size(0)
        wt_hidden_state = out_combined.last_hidden_state[:batch_size]
        mut_hidden_state = out_combined.last_hidden_state[batch_size:]

        # 4. MEAN POOLING (Ignoring padding tokens)
        wt_pooled = self.mean_pooling(wt_hidden_state, attention_mask_wt)
        mut_pooled = self.mean_pooling(mut_hidden_state, attention_mask_mut)

        # 5. Combine extracted features mathematically
        combined_embeddings = torch.cat([
            wt_pooled,
            mut_pooled,
            mut_pooled - wt_pooled  # Global difference
        ], dim=1)

        return self.regressor_head(combined_embeddings)


class ComposerProteinModelESM(ComposerModel):
    def __init__(self, pretrained_model_name, tokenizer, freeze_encoder=True):
        super().__init__()
        self.classification_threshold = 0.0
        self.model = ESMProteinMutationCore(pretrained_model_name, tokenizer)

        # --- 1. FREEZE BACKBONE (Linear Probing) ---
        if freeze_encoder:
            print(f"INFO: Freezing ESM encoder (Linear Probing). Only MLP head is training.")
            for param in self.model.esm.parameters():
                param.requires_grad = False

            # Ensure Regressor Head is unfrozen
            for param in self.model.regressor_head.parameters():
                param.requires_grad = True
        else:
            print(f"INFO: Full Fine-Tuning (ESM encoder is unfrozen).")
            # If doing full fine-tuning, enable gradient checkpointing to save memory
            self.model.esm.gradient_checkpointing_enable()

        # Freeze positional embeddings (always a good idea with ESM)
        for name, param in self.model.named_parameters():
            if "position_embeddings" in name:
                param.requires_grad = False

        self.criterion = nn.MSELoss()

        # --- Metrics ---
        self.train_metrics = nn.ModuleDict({
            'mse': MeanSquaredError(),
        })
        self.val_metrics = nn.ModuleDict({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'mape': MeanAbsolutePercentageError(),
            'pearson': PearsonCorrCoef(),
            'r2': R2Score(),
            'f1': F1Score(task="binary"),
            'mcc': MatthewsCorrCoef(task="binary"),
        })

    def forward(self, batch):
        # Extract only what the model needs.
        # ESM doesn't use token_type_ids, and our logic doesn't need them either.
        return self.model(
            input_ids_wt=batch['input_ids_wt'],
            attention_mask_wt=batch['attention_mask_wt'],
            input_ids_mut=batch['input_ids_mut'],
            attention_mask_mut=batch['attention_mask_mut']
        )

    def loss(self, outputs, batch):
        return self.criterion(outputs.squeeze(), batch["labels"])

    def get_metrics(self, is_train: bool = False):
        if is_train:
            return self.train_metrics
        return self.val_metrics

    def update_metric(self, batch, outputs, metric):
        targets = batch["labels"]
        predictions = outputs.squeeze()

        regression_metrics = (
            MeanSquaredError,
            MeanAbsoluteError,
            MeanAbsolutePercentageError,
            PearsonCorrCoef,
            R2Score
        )

        if isinstance(metric, regression_metrics):
            metric.update(predictions, targets)
        else:
            binary_preds = (predictions > self.classification_threshold).long()
            binary_targets = (targets > self.classification_threshold).long()
            metric.update(binary_preds, binary_targets)


# --- OUR NEW CUSTOM CALLBACK FOR REDUCELRONPLATEAU ---
class ReduceLROnPlateauCallback(Callback):
    """
    This callback monitors the specified metric after a specific evaluation
    (in our case 'frequent_val') and sends it to ReduceLROnPlateau.
    """

    def __init__(self, optimizer, monitor_metric='pearson', mode='max', factor=0.5, patience=1):
        self.optimizer = optimizer
        # Initialize standard PyTorch scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
        self.monitor_metric = monitor_metric
        # Save the last known LR for comparison
        self.last_lr = [group['lr'] for group in optimizer.param_groups]

    def eval_end(self, state: State, logger: Logger):
        # Called at the end of ANY evaluation.
        # Check if "frequent_val" just finished.
        if state.dataloader_label == "frequent_val":
            # Composer stores metrics in state.eval_metrics
            metrics = state.eval_metrics.get("frequent_val", {})
            if self.monitor_metric in metrics:
                # Get the currently computed metric value
                metric_val = metrics[self.monitor_metric].compute().item()

                # Perform .step() with the measured value
                self.scheduler.step(metric_val)

                # Check if the optimizer LR changed and log it
                current_lr = [group['lr'] for group in self.optimizer.param_groups]
                if current_lr != self.last_lr:
                    if dist.get_global_rank() == 0:
                        print(
                            f"\n[LR Scheduler] Metric '{self.monitor_metric}' is plateauing. Reducing Learning Rate from {self.last_lr[0]:.2e} to {current_lr[0]:.2e}!")
                    self.last_lr = current_lr

    # Important: implement state saving and loading,
    # so the scheduler resumes correctly from a checkpoint.
    def state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict(),
            "last_lr": self.last_lr
        }

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state["scheduler"])
        self.last_lr = state.get("last_lr", [0.0])


# --- NEW CALLBACK FOR SAVING BEST MODEL DURING FREQ VAL ---
class SaveBestFrequentCallback(Callback):
    """
    Monitors frequent_val metric and if the model improves, saves current weights
    to a specific file for the given epoch.
    """

    def __init__(self, save_dir: str, monitor_metric: str = 'pearson', mode: str = 'max'):
        self.save_dir = save_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        # Initial worst possible value for maximization is -infinity
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def eval_end(self, state: State, logger: Logger):
        if state.dataloader_label == "frequent_val":
            metrics = state.eval_metrics.get("frequent_val", {})
            if self.monitor_metric in metrics:
                current_metric = metrics[self.monitor_metric].compute().item()

                # Check if current metric is better than the best so far
                is_better = current_metric < self.best_metric if self.mode == 'min' else current_metric > self.best_metric

                if is_better:
                    self.best_metric = current_metric

                    # Dist rank check ensures we only save from the main process (for DDP/multi-GPU)
                    if dist.get_global_rank() == 0:
                        os.makedirs(self.save_dir, exist_ok=True)
                        epoch = int(state.timestamp.epoch)

                        # Filename includes epoch. During one epoch, this file will be overwritten
                        # by the best model achieved during frequent_val.
                        filename = os.path.join(self.save_dir, f"best_frequent_epoch_{epoch}.pt")

                        print(
                            f"\n[SaveBestFrequentCallback] New record for {self.monitor_metric}: {current_metric:.4f}! Saving model to {filename}")

                        # Save the model's state_dict
                        torch.save(state.model.state_dict(), filename)

    # Implementation to keep state upon resumption from checkpoint
    def state_dict(self):
        return {"best_metric": self.best_metric}

    def load_state_dict(self, state):
        self.best_metric = state["best_metric"]


# --- NEW CALLBACK FOR DYNAMIC ESM MODEL UNFREEZING ---
class UnfreezeOnPlateauCallback(Callback):
    """
    Monitors validation (default full_val). If the metric does not improve
    after 'patience' steps, unfreezes the ESM model for Full Fine-Tuning.
    """

    def __init__(self, monitor_metric='pearson', mode='max', patience=1, dataloader_label='full_val'):
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.dataloader_label = dataloader_label
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.bad_epochs = 0
        self.is_unfrozen = False

    def eval_end(self, state: State, logger: Logger):
        # If the model is already unfrozen, do nothing
        if self.is_unfrozen:
            return

        if state.dataloader_label == self.dataloader_label:
            metrics = state.eval_metrics.get(self.dataloader_label, {})
            if self.monitor_metric in metrics:
                current_metric = metrics[self.monitor_metric].compute().item()

                is_better = current_metric < self.best_metric if self.mode == 'min' else current_metric > self.best_metric

                if is_better:
                    self.best_metric = current_metric
                    self.bad_epochs = 0
                else:
                    self.bad_epochs += 1

                if self.bad_epochs >= self.patience:
                    if dist.get_global_rank() == 0:
                        print(
                            f"\n[UnfreezeCallback] No improvement for {self.patience} validations on {self.dataloader_label}. UNFREEZING ESM MODEL!")

                    # Handle potential DDP wrapper (get direct access to ComposerModel)
                    base_model = state.model.module if hasattr(state.model, 'module') else state.model

                    # Unfreeze ESM layer parameters
                    for param in base_model.model.esm.parameters():
                        param.requires_grad = True

                    # Enable gradient checkpointing (saves VRAM when training such a large model)
                    base_model.model.esm.gradient_checkpointing_enable()

                    self.is_unfrozen = True

    def state_dict(self):
        return {
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "is_unfrozen": self.is_unfrozen
        }

    def load_state_dict(self, state):
        self.best_metric = state["best_metric"]
        self.bad_epochs = state["bad_epochs"]
        self.is_unfrozen = state["is_unfrozen"]


# --- 3. HTML Report Generator ---
import wandb


def log_interactive_report_polars(df: pl.DataFrame, table_name: str, step: int = None):
    df_export = df.select([
        "wt_sequence", "mut_sequence", "mutation",
        "cath_class", "cath_arch", "cath_topology", "cath_homology", "data_source",
        "predicted_fitness", "actual_fitness"
    ]).with_columns(
        (pl.col("predicted_fitness") - pl.col("actual_fitness")).abs().alias("diff")
    )

    json_data = df_export.write_json()
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnt', 'html_templates',
                                 'interactive_report_template.html')

    if not os.path.exists(template_path):
        print(f"WARNING: Template not found at {template_path}")
        return

    with open(template_path, 'r') as f:
        html_template_str = f.read()

    html_content = html_template_str.format(json_data=json_data, table_name=table_name)
    log_payload = {table_name: wandb.Html(html_content)}

    if step is not None:
        wandb.log(log_payload, step=step)
    else:
        wandb.log(log_payload)


class InteractiveReportCallbackESM(Callback):
    def __init__(self, log_function, val_original_df: pl.DataFrame):
        self.log_func = log_function
        try:
            self.val_original_df = val_original_df.with_row_index(name="row_idx")
        except AttributeError:
            self.val_original_df = val_original_df.with_row_count(name="row_idx")
        self.preds = []
        self.indices = []

    def eval_batch_end(self, state: State, logger: Logger):
        if state.dataloader_label == "full_val":
            outputs = state.outputs.detach().float().cpu().numpy().flatten().tolist()
            batch_indices = state.batch['row_idx'].detach().cpu().numpy().flatten().tolist()
            self.preds.extend(outputs)
            self.indices.extend(batch_indices)

    def eval_end(self, state: State, logger: Logger):
        all_preds = [None for _ in range(dist.get_world_size())]
        all_indices = [None for _ in range(dist.get_world_size())]

        torch.distributed.all_gather_object(all_preds, self.preds)
        torch.distributed.all_gather_object(all_indices, self.indices)

        if dist.get_global_rank() == 0:
            full_preds = [item for sublist in all_preds for item in sublist]
            full_indices = [item for sublist in all_indices for item in sublist]

            if len(full_preds) > 0:
                preds_df = pl.DataFrame({
                    "row_idx": full_indices,
                    "predicted_fitness": full_preds
                }).with_columns(pl.col("row_idx").cast(pl.UInt32))

                self.val_original_df = self.val_original_df.with_columns(pl.col("row_idx").cast(pl.UInt32))
                final_df = self.val_original_df.join(preds_df, on="row_idx", how="inner")

                if "fitness" in final_df.columns and "actual_fitness" not in final_df.columns:
                    final_df = final_df.rename({"fitness": "actual_fitness"})

                table_name = f"val_results_epoch_{int(state.timestamp.epoch)}"
                current_step = int(state.timestamp.batch)

                try:
                    self.log_func(final_df, table_name=table_name, step=current_step)
                    print(f"Interactive Report generated with ALL {len(final_df)} samples.")
                except Exception as e:
                    print(f"Error generating report: {e}")

        self.preds = []
        self.indices = []


# --- 4. Main Training Function for ESM2 ---
def train_esm_model(train_df_raw, val_df_raw, config: ConfigESM, num_workers=1, epochs=5):
    print(f"Train samples: {len(train_df_raw)}, Validation samples: {len(val_df_raw)}")

    config_dict = asdict(config)
    if "wandb_token" in config_dict:
        del config_dict["wandb_token"]
    print(f"The config: {json.dumps(config_dict)}")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    print(f"Preprocessing Training Data (Window Size: {config.seq_window_size})...")
    train_df = prepare_data_dynamic(train_df_raw, max_total_length=config.max_length,
                                    window_size=config.seq_window_size)

    print(f"Preprocessing Validation Data (Window Size: {config.seq_window_size})...")
    validation_df = prepare_data_dynamic(val_df_raw, max_total_length=config.max_length,
                                         window_size=config.seq_window_size)

    if isinstance(validation_df, pd.DataFrame):
        validation_df = pl.from_pandas(validation_df)

    composer_model = ComposerProteinModelESM(config.pretrained_model, tokenizer)

    train_dataset = ProteinMutationDatasetESM(train_df, tokenizer, config.max_length)
    train_sampler = dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=num_workers)

    val_dataset = ProteinMutationDatasetESM(validation_df, tokenizer, config.max_length)

    val_sampler_subset = dist.get_sampler(val_dataset, shuffle=True, drop_last=True)
    val_loader_subset = DataLoader(val_dataset,
                                   batch_size=config.batch_size,
                                   sampler=val_sampler_subset,
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=num_workers)

    val_sampler_full = dist.get_sampler(val_dataset, shuffle=False, drop_last=False)
    val_loader_full = DataLoader(val_dataset,
                                 batch_size=config.batch_size,
                                 sampler=val_sampler_full,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=num_workers)

    eval_frequent = Evaluator(
        label="frequent_val",
        dataloader=val_loader_subset,
        metric_names=['mse', 'pearson'],
        subset_num_batches=20,
        eval_interval="200ba"
    )

    eval_full = Evaluator(
        label="full_val",
        dataloader=val_loader_full,
        eval_interval="1ep"
    )

    optimizer = DecoupledAdamW(composer_model.parameters(), lr=config.learning_rate)

    # Setup ReduceLROnPlateau for Pearson correlation (looking for maximum)
    lr_plateau_callback = ReduceLROnPlateauCallback(
        optimizer=optimizer,
        monitor_metric='pearson',
        mode='max',
        factor=0.5,
        patience=1  # Changed from 2 to 1
    )

    # Add callback to save best model according to frequent_val (by Pearson)
    save_path = os.path.join(config.base_dir, config.save_folder)
    save_best_freq_callback = SaveBestFrequentCallback(
        save_dir=save_path,
        monitor_metric='pearson',
        mode='max'
    )

    # Callback to unfreeze ESM model after 3 validations without improvement on frequent_val
    unfreeze_callback = UnfreezeOnPlateauCallback(
        monitor_metric='pearson',
        mode='max',
        patience=3,
        dataloader_label='frequent_val'
    )

    html_callback = InteractiveReportCallbackESM(
        log_function=log_interactive_report_polars,
        val_original_df=validation_df
    )

    # ORIGINAL EARLY STOPPER (Monitors the whole epoch - full_val)
    early_stopper_full = EarlyStopper(
        monitor="mcc",  # Can also be changed to pearson in the future if mcc is not optimal
        dataloader_label="full_val",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_delta
    )

    # NEW EARLY STOPPER (Monitors frequent_val every 200ba)
    early_stopper_freq = EarlyStopper(
        monitor="pearson",
        dataloader_label="frequent_val",
        patience=config.early_stopping_patience_freq,  # Set to 10
        min_delta=config.early_stopping_delta
    )

    checkpoint_saver = CheckpointSaver(
        folder=save_path,
        filename="model_epoch_{epoch}.pt",
        save_interval="1ep",
        overwrite=False
    )

    # Added the second early stopper
    callbacks = [
        LRMonitor(),
        OptimizerMonitor(),
        html_callback,
        early_stopper_full,
        early_stopper_freq,
        checkpoint_saver,
        lr_plateau_callback,
        save_best_freq_callback,
        unfreeze_callback
    ]

    gc = GradientClipping(clipping_type='norm', clipping_threshold=1.0)

    wandb_logger = WandBLogger(
        project=config.project_name,
        init_kwargs={"config": config_dict},
        log_artifacts=False,
        rank_zero_only=True
    )

    # Update the print statement to reflect the newly set epochs
    print(f"=== Starting ESM2 Training ({config.epochs} epochs) ===")

    load_path = getattr(config, 'load_path', None)
    if load_path:
        print(f"Resuming training from: {load_path}")

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=[eval_frequent, eval_full],
        max_duration=f"{config.epochs}ep",  # Now takes the parameter from config.epochs (e.g., 50)
        optimizers=optimizer,
        algorithms=[gc],
        seed=42,
        load_path=load_path,
        callbacks=callbacks,
        loggers=[wandb_logger],
        device="gpu",
        precision="amp_bf16",
        device_train_microbatch_size="auto"
    )

    trainer.fit()
    print("=== ESM2 Training Finished ===")
    trainer.close()
    print("ESM2 Training finished successfully!")