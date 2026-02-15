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
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from torch.utils.data import Dataset, DataLoader
from composer.utils import dist

# Torchmetrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score, \
    MeanAbsolutePercentageError, MatthewsCorrCoef, F1Score

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --- Configuration for ESM2 ---
@dataclass
class ConfigESM:
    project_name: str = "protein-mutation-prediction-esm2"
    pretrained_model: str = "facebook/esm2_t33_650M_UR50D"  # 650M parameter model
    wandb_token: str = ""
    max_length: int = 1024
    batch_size: int = 24  # Reduced batch size (ESM2 is heavier than ProtBERT)
    learning_rate: float = 2e-5
    epochs: int = 15
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.001
    base_dir: str = "./"
    seq_window_size: int = 511  # ESM2 context is larger, we can use larger windows
    save_folder: str = "checkpoints_esm2"


# --- Data Preparation Helper ---
def prepare_data_dynamic(df: pl.DataFrame, max_total_length: int = 1024, window_size: int = 511):
    """
    Standard data prep: Cuts sequences around the mutation.
    """
    # 1. Standardize column names
    if "target" in df.columns and "fitness" not in df.columns:
        df = df.rename({"target": "fitness"})
    if "mutation" not in df.columns and "mut_type" in df.columns:
        df = df.rename({"mut_type": "mutation"})

    # 2. Priority: Use pre-calculated fragments
    if window_size == 255 and "fragment_255_org" in df.columns and "fragment_255_mut" in df.columns:
        print("INFO: Using pre-calculated fragments (fragment_255_org/mut).")
        df_processed = df.with_columns([
            pl.col("fragment_255_org").str.replace_all("[UZOB]", "X").alias("clean_wt"),
            pl.col("fragment_255_mut").str.replace_all("[UZOB]", "X").alias("clean_mut")
        ])
        return df_processed

    # 3. Dynamic calculation
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

        # Tokenize as a pair. ESM tokenizer will handle special tokens automatically.
        # Usually: <s> WT </s> </s> MUT </s>
        inputs = self.tokenizer(
            seq_wt,
            seq_mut,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            # ESM usually doesn't use token_type_ids, but we'll grab them if they exist
            'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(),
            'labels': torch.tensor(target, dtype=torch.float),
            'row_idx': torch.tensor(row_id, dtype=torch.long)
        }


class ComposerProteinModelESM(ComposerModel):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()

        self.classification_threshold = 0.0

        # Načtení modelu
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=1,
            problem_type="regression"
        )

        self.model.gradient_checkpointing_enable()

        # Zkusíme najít base model (obvykle uložen jako .esm)
        base_model = getattr(self.model, "esm", None)

        if base_model is not None:
            # Odstranění Contact Head (pokud existuje)
            if hasattr(base_model, "contact_head"):
                base_model.contact_head = None

            # Odstranění Pooleru (pokud existuje)
            # EsmForSequenceClassification používá vlastní hlavičku nad hidden_states,
            # takže původní pooler je zbytečný a způsobuje DDP chybu.
            if hasattr(base_model, "pooler"):
                base_model.pooler = None

        # Resize embeddings
        if len(tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(tokenizer))

        self.criterion = nn.HuberLoss(delta=0.2)

        # --- Metriky ---
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
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids')
        )
        return outputs.logits

    def loss(self, outputs, batch):
        return self.criterion(outputs.squeeze(), batch["labels"])

    def get_metrics(self, is_train: bool = False):
        if is_train:
            return {}
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

    # Path to template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'html_templates',
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
def train_esm_model(train_df_raw, val_df_raw, config: ConfigESM, num_workers=8, epochs=5):
    print(f"Train samples: {len(train_df_raw)}, Validation samples: {len(val_df_raw)}")

    config_dict = asdict(config)
    if "wandb_token" in config_dict:
        del config_dict["wandb_token"]
    print(f"The config: {json.dumps(config_dict)}")

    # 1. Tokenizer
    # AutoTokenizer is safer for ESM than EsmTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    print(f"Preprocessing Training Data (Window Size: {config.seq_window_size})...")
    train_df = prepare_data_dynamic(train_df_raw, max_total_length=config.max_length,
                                    window_size=config.seq_window_size)

    print(f"Preprocessing Validation Data (Window Size: {config.seq_window_size})...")
    validation_df = prepare_data_dynamic(val_df_raw, max_total_length=config.max_length,
                                         window_size=config.seq_window_size)

    if isinstance(validation_df, pd.DataFrame):
        validation_df = pl.from_pandas(validation_df)

    # 2. Model
    composer_model = ComposerProteinModelESM(config.pretrained_model, tokenizer)

    # NOTE: No freezing here. Full fine-tuning.

    # 3. DataLoaders
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

    # 4. Evaluators
    eval_frequent = Evaluator(
        label="frequent_val",
        dataloader=val_loader_subset,
        metric_names=['mse', 'pearson'],
        subset_num_batches=20,
        eval_interval="2000ba"
    )

    eval_full = Evaluator(
        label="full_val",
        dataloader=val_loader_full,
        eval_interval="1ep"
    )

    optimizer = DecoupledAdamW(composer_model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup="0.20dur", alpha_f=0.01)

    # 5. Callbacks
    html_callback = InteractiveReportCallbackESM(
        log_function=log_interactive_report_polars,
        val_original_df=validation_df
    )

    early_stopper = EarlyStopper(
        monitor="mcc",
        dataloader_label="full_val",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_delta
    )

    save_path = os.path.join(config.base_dir, config.save_folder)
    checkpoint_saver = CheckpointSaver(
        folder=save_path,
        filename="model_epoch_{epoch}.pt",
        save_interval="1ep",
        overwrite=False
    )

    callbacks = [LRMonitor(), html_callback, early_stopper, checkpoint_saver]
    gc = GradientClipping(clipping_type='norm', clipping_threshold=1.0)

    wandb_logger = WandBLogger(
        project=config.project_name,
        init_kwargs={"config": config_dict},
        log_artifacts=False,
        rank_zero_only=True
    )

    # 6. Trainer
    print(f"=== Starting ESM2 Training ({epochs} epochs) ===")
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=[eval_frequent, eval_full],
        max_duration=f"{epochs}ep",
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=[gc],
        seed=42,
        parallelism_config={'ddp': {'find_unused_parameters': True}},
        callbacks=callbacks,
        loggers=[wandb_logger],
        device="gpu",
        precision="amp_bf16"
    )

    trainer.fit()
    print("=== ESM2 Training Finished ===")
    trainer.close()
    print("ESM2 Training finished successfully!")
