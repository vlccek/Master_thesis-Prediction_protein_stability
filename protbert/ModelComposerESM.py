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
from composer.algorithms import GradientClipping, StochasticDepth
from composer.callbacks import EarlyStopper, LRMonitor, OptimizerMonitor, CheckpointSaver
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
import torch_optimizer as optim
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from torch.utils.data import Dataset, DataLoader
# Torchmetrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score, MeanAbsolutePercentageError, MatthewsCorrCoef, F1Score
from transformers import EsmTokenizer, EsmModel, AutoConfig, AutoTokenizer
from transformers import DataCollatorWithPadding

# Configuration for ESM2
@dataclass
class ConfigESM:
    project_name: str = "protein-mutation-prediction-esm2"
    pretrained_model: str = "facebook/esm2_t33_650M_UR50D" # Specific ESM2 model
    wandb_token: str = ""
    max_length: int = 1024 # ESM2 models often have specific max lengths
    batch_size: int = 48
    learning_rate: float = 5e-5
    hidden_dropout_prob: float = 0.1
    epochs: float = 15
    freeze_layers: int = 3
    early_stopping_patience: int = 2
    early_stopping_delta: float = 0.001
    step_validation: int = 1500
    base_dir: str = "./"
    seq_window_size: int = 255 # Should be compatible with ESM2's max length if truncation happens
    save_folder: str = "checkpoints_esm2"
    stochastic_depth_drop_rate: float = 0.2


# === Data Preparation (Copied from ModelComposter.py, assumed generic) ===
def prepare_data_dynamic(df: pl.DataFrame, max_total_length: int = 1024, window_size: int = 511):
    """
    Přijme DF, přidá sloupce 'clean_wt' a 'clean_mut' (ořezané).
    Logika: Vytvoří okno o velikosti `window_size` kolem mutace.
    """

    # 1. Standardizace názvů sloupců
    if "target" in df.columns and "fitness" not in df.columns:
        df = df.rename({"target": "fitness"})
    if "mutation" not in df.columns and "mut_type" in df.columns:
        df = df.rename({"mut_type": "mutation"})

    # 2. Priorita: Použít předvypočítané fragmenty (POUZE pokud sedí velikost okna)
    # Předpokládáme, že sloupce 'fragment_255_org' odpovídají oknu 255.
    if window_size == 255 and "fragment_255_org" in df.columns and "fragment_255_mut" in df.columns:
        print("INFO: Using pre-calculated fragments (fragment_255_org/mut).")
        df_processed = df.with_columns([
            pl.col("fragment_255_org").str.replace_all("[UZOB]", "X").alias("clean_wt"),
            pl.col("fragment_255_mut").str.replace_all("[UZOB]", "X").alias("clean_mut")
        ])
        return df_processed

    # 3. Dynamický výpočet (pro homology dataset nebo jinou délku okna)
    print(f"INFO: Calculating fragments dynamically (Window={window_size}).")

    # Sjednocení názvů vstupních sekvencí
    if "original_seq_full" in df.columns:
        df = df.rename({"original_seq_full": "wt_sequence", "mutated_seq_full": "mut_sequence"})

    if "wt_sequence" not in df.columns:
        raise ValueError(f"Dataset missing 'wt_sequence'. Columns: {df.columns}")

    # Funkce pro nalezení indexu mutace (prioritně z popisu 'mutation')
    def get_mutation_idx(row):
        import re
        # Pokusíme se parsovat číslo z "A168V"
        if row['mutation']:
            match = re.search(r'\d+', str(row['mutation']))
            if match:
                return int(match.group(0)) - 1

        # Fallback: Porovnání sekvencí
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


# --- 1. Dataset (Adapted for ESM tokenizer) ---
class ProteinMutationDatasetESM(Dataset):
    def __init__(self, processed_df: pl.DataFrame, tokenizer, max_length=1024):
        """
        Args:
            processed_df: Výstup z funkce prepare_data_with_polars
            tokenizer: HuggingFace tokenizer (ESM compatible)
        """
        self.tokenizer = tokenizer

        # ESM tokenizers usually don't require spaces between amino acids
        self.wt_seqs = processed_df["clean_wt"].to_list()
        self.mut_seqs = processed_df["clean_mut"].to_list()
        self.targets = processed_df["fitness"].to_list()

        self.max_length = max_length
        self.ids = list(range(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    # Removed add_spaces, as ESM tokenizers handle sequences directly
    # def add_spaces(self, seq):
    #     return " ".join(seq)

    def __getitem__(self, idx):
        seq_wt = self.wt_seqs[idx]
        seq_mut = self.mut_seqs[idx]
        target = self.targets[idx]
        row_id = self.ids[idx]

        # Tokenize directly, ESM tokenizers typically don't need token_type_ids for single sequences
        # When concatenating two sequences like this, ESM tokenizers usually handle the separation.
        inputs = self.tokenizer(
            seq_wt,
            seq_mut,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt' # Return PyTorch tensors directly
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            # ESM models typically don't use token_type_ids for protein sequences,
            # especially not in the same way BERT does for sentence pair tasks.
            # Removing it for now, can be added back if a specific ESM model requires it.
            # 'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.float),
            'row_idx': torch.tensor(row_id, dtype=torch.long)
        }


# --- 2. Model (Adapted for ESM2 Architecture) ---
class ProteinMutationModelESM(nn.Module):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()

        # 1. Zkusíme vypnout pooler přes konfiguraci
        self.esm_model = EsmModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        self.tokenizer = tokenizer

        # Odstranění Pooleru (pokud ho konfig neodstranil)
        if hasattr(self.esm_model, 'pooler') and self.esm_model.pooler is not None:
            del self.esm_model.pooler

        # Odstranění Contact Head (častý viník u ESM modelů)
        if hasattr(self.esm_model, 'contact_head') and self.esm_model.contact_head is not None:
            del self.esm_model.contact_head

        # Resize token embeddings if tokenizer vocabulary is larger
        if len(tokenizer) > self.esm_model.config.vocab_size:
            self.esm_model.resize_token_embeddings(len(tokenizer))

        # ESM2 models typically have a 'last_hidden_state' from the main model output
        input_dim = self.esm_model.config.hidden_size * 3 # WT, MUT, DIFF

        self.regressor_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
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

    def forward(self, input_ids, attention_mask): # Removed token_type_ids
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # ESM models usually return a BaseModelOutputWithPoolingAndCrossAttentions
        sequence_output = outputs.last_hidden_state

        sep_token_id = self.tokenizer.sep_token_id
        # FIX: Pokud není sep_token definován, použijeme eos_token (u ESM je to obvykle </s>)
        if sep_token_id is None:
            sep_token_id = self.tokenizer.eos_token_id

        # Find the index of the first SEP token
        # This assumes the tokenizer places `seq_wt [SEP] seq_mut [SEP]`
        # We need to find the SEP that separates WT from MUT.
        first_sep_mask = (input_ids == sep_token_id)

        # Get the index of the first SEP token (which separates WT and MUT)
        # Using argmax here for the first occurrence of sep_token_id
        # FIX: Převedeme boolean masku na long (0/1), aby argmax fungoval správně
        sep_indices = torch.argmax(first_sep_mask.long(), dim=1)


        # Create masks for WT and MUT based on the separator index
        # WT sequence is from index 1 (after <cls>) to sep_indices - 1
        # MUT sequence is from sep_indices + 1 to the second sep_indices - 1 (if multiple) or end
        arange_mask = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)

        # WT mask: from after CLS (index 1) up to the first SEP token
        wt_mask = (arange_mask > 0) & (arange_mask < sep_indices.unsqueeze(1)) & attention_mask.bool()
        
        # MUT mask: from after the first SEP token to the end of the sequence
        # This needs careful handling. If tokenizer concatenates as 'seq_wt <sep> seq_mut <sep>',
        # then the second sep is at the very end.
        # Let's find the second SEP token's index if it exists, or use attention_mask for end.
        # A simpler approach might be to identify the start of the second sequence.
        
        # For simplicity, assuming `seq_wt [SEP] seq_mut [SEP]` structure,
        # where the first SEP divides them. The second SEP is typically at the end of the input.
        # We need to find the start of the mut_sequence which is `sep_indices + 1`.
        # The end of mut_sequence would be where attention_mask ends, or before the *next* SEP if it's a batch.
        
        # Let's refine the mut_mask to start after the first SEP.
        # It should end before the final SEP token added by the tokenizer if any.
        
        # For ESM tokenization: `<s> Sequence1 </s> Sequence2 </s>`
        # `<s>` is typically ID 0, `</s>` is ID 2.
        # `sep_token_id` is 2.
        # `sep_indices` will give the index of the first `</s>`.
        
        # wt_mask: tokens between `<s>` and first `</s>`
        # mut_mask: tokens between first `</s>` and second `</s>`
        
        # To get the second SEP index more robustly, we can find all sep tokens
        all_sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)
        
        # For each batch item, get the index of the first and second SEP.
        # If there's only one, then it's `seq_wt [SEP] seq_mut`.
        # If there are two, it's `seq_wt [SEP] seq_mut [SEP]`.
        
        # This part is crucial and dependent on how the tokenizer handles two sequences.
        # Let's assume the common `seq_wt [SEP] seq_mut [SEP]` pattern where the first SEP divides them.
        
        # A more robust way to get the start of the second sequence (mut_sequence)
        # is to check where token_type_ids would typically switch if ESM used them,
        # or rely on the tokenizer's specific output for `seq_wt, seq_mut`.
        
        # Let's try to simulate the BERT-like segment averaging, assuming the first sep divides.
        # WT representation: average of tokens from index 1 (after CLS/bos) up to first SEP.
        # MUT representation: average of tokens from after first SEP up to second SEP (or end).

        # For ESM, usually tokens are <cls> AA AA ... <sep> BB BB ... <sep>
        # The first sep_indices correctly points to the end of seq_wt.
        # The second sequence starts at sep_indices + 1.
        # The end of the second sequence is usually the last token before the *last* sep token,
        # or simply the end of the attention mask if the tokenizer didn't add a trailing sep.

        # Let's verify the ESM tokenizer behavior:
        # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # tokenized = tokenizer("AAAA", "BBBB", return_tensors="pt")
        # print(tokenized)
        # outputs:
        # {'input_ids': tensor([[0, 6, 6, 6, 6, 2, 6, 6, 6, 6, 2]]),
        #  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
        # IDs: 0=<s>, 2=</s> (SEP), 6=A.
        # So it's <s> AAAA </s> BBBB </s>.
        # First SEP is at index 5. Second SEP is at index 10.

        # Correcting masks based on ESM tokenizer behavior:
        
        # Find all SEP token indices for each item in the batch
        # This will be tricky with torch.argmax if there are multiple.
        # Let's get the indices directly.
        
        # We need the index of the first `</s>` (which is `sep_token_id`) to define end of WT
        # And the index of the last `</s>` to define end of MUT (if it exists)
        
        wt_start = 1 # After <s>
        
        # Find first sep token for each batch item (end of WT)
        # Using a loop for clarity, can be vectorized
        first_sep_indices = []
        for i in range(input_ids.size(0)):
            # Find all occurrences of sep_token_id in this sequence
            sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                first_sep_indices.append(sep_positions[0].item())
            else:
                # Fallback: if no sep token found, assume whole sequence is WT
                first_sep_indices.append(input_ids.size(1))
        first_sep_indices = torch.tensor(first_sep_indices, device=input_ids.device)

        # Find last sep token for each batch item (end of MUT)
        last_sep_indices = []
        for i in range(input_ids.size(0)):
            sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 1: # If there's at least a second sep
                last_sep_indices.append(sep_positions[-1].item())
            elif len(sep_positions) == 1: # Only one sep means mut goes to end of attention_mask
                 last_sep_indices.append(attention_mask[i].sum().item())
            else: # No sep tokens at all
                last_sep_indices.append(attention_mask[i].sum().item()) # Mut covers remaining attention
        last_sep_indices = torch.tensor(last_sep_indices, device=input_ids.device)

        wt_mask = (arange_mask >= wt_start) & (arange_mask < first_sep_indices.unsqueeze(1)) & attention_mask.bool()
        mut_start_after_first_sep = first_sep_indices + 1
        mut_mask = (arange_mask >= mut_start_after_first_sep.unsqueeze(1)) & (arange_mask < last_sep_indices.unsqueeze(1)) &  attention_mask.bool()
                   
        wt_sum = (sequence_output * wt_mask.unsqueeze(-1).float()).sum(dim=1)
        mut_sum = (sequence_output * mut_mask.unsqueeze(-1).float()).sum(dim=1)

        wt_count = wt_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mut_count = mut_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        wt_repr = wt_sum / wt_count
        mut_repr = mut_sum / mut_count

        diff = mut_repr - wt_repr

        combined_embeddings = torch.cat([wt_repr, mut_repr, diff], dim=1)
        return self.regressor_head(combined_embeddings)


# --- 3. Composer Wrapper (Copied as is, assumed generic) ---
class ComposerProteinModelESM(ComposerModel):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()

        self.classification_threshold = 0.0

        self.model = ProteinMutationModelESM(pretrained_model_name, tokenizer)

        self.criterion = nn.HuberLoss(delta=0.2)

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
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            # token_type_ids is removed for ESM
            # token_type_ids=batch.get('token_type_ids')
        )

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


# --- 4. Helper function for freezing layers (Adapted for ESM) ---
def freeze_esm_layers(model, num_layers_to_freeze):
    """
    Zmrazí embeddingy a prvních N vrstev encoderu ESM modelu.
    """
    if num_layers_to_freeze == 0:
        return

    print(f"INFO: Freezing embeddings and first {num_layers_to_freeze} ESM encoder layers.")

    # ESM embeddings are usually under `esm_model.embeddings`
    for param in model.esm_model.embeddings.parameters():
        param.requires_grad = False

    # ESM encoder layers are usually under `esm_model.encoder.layer`
    for i in range(num_layers_to_freeze):
        if i < len(model.esm_model.encoder.layer):
            for param in model.esm_model.encoder.layer[i].parameters():
                param.requires_grad = False
        else:
            print(f"WARNING: Cannot freeze layer {i}, ESM model has only {len(model.esm_model.encoder.layer)} layers.")


# --- HTML Report Generator (Copied as is) ---
import wandb
from composer.utils import dist

def log_interactive_report_polars(df: pl.DataFrame, table_name: str, step: int = None):
    df_export = df.select([
        "wt_sequence", "mut_sequence", "mutation",
        "cath_class", "cath_arch", "cath_topology", "cath_homology", "data_source",
        "predicted_fitness", "actual_fitness"
    ]).with_columns(
        (pl.col("predicted_fitness") - pl.col("actual_fitness")).abs().alias("diff")
    )

    json_data = df_export.write_json()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'html_templates', 'interactive_report_template.html')
    # Make sure 'interactive_report_template.html' exists in '../html_templates/'
    try:
        with open(template_path, 'r') as f:
            html_template_str = f.read()
    except FileNotFoundError:
        print(f"WARNING: interactive_report_template.html not found at {template_path}. Skipping HTML report.")
        return "" # Return empty string or handle error appropriately

    html_content = html_template_str.format(json_data=json_data, table_name=table_name)
    log_payload = {table_name: wandb.Html(html_content)}
    if step is not None:
        wandb.log(log_payload, step=step)
    else:
        wandb.log(log_payload)
    return html_content


class InteractiveReportCallbackESM(Callback): # Renamed for clarity
    def __init__(self, log_function, val_original_df: pl.DataFrame):
        self.log_func = log_function
        try:
            self.val_original_df = val_original_df.with_row_index(name="row_idx")
        except AttributeError:
            self.val_original_df = val_original_df.with_row_count(name="row_idx")
        self.preds = []
        self.indices = []
        self.targets = []

    def eval_batch_end(self, state: State, logger: Logger):
        if state.dataloader_label == "full_val":
            outputs = state.outputs.detach().float().cpu().numpy().flatten().tolist()
            targets = state.batch['labels'].detach().float().cpu().numpy().flatten().tolist()
            batch_indices = state.batch['row_idx'].detach().cpu().numpy().flatten().tolist()
            self.preds.extend(outputs)
            self.targets.extend(targets)
            self.indices.extend(batch_indices)

    def eval_end(self, state: State, logger: Logger):
        all_preds = [None for _ in range(dist.get_world_size())]
        all_indices = [None for _ in range(dist.get_world_size())]
        all_targets = [None for _ in range(dist.get_world_size())]

        torch.distributed.all_gather_object(all_preds, self.preds)
        torch.distributed.all_gather_object(all_indices, self.indices)
        torch.distributed.all_gather_object(all_targets, self.targets)

        if dist.get_global_rank() == 0:
            full_preds = [item for sublist in all_preds for item in sublist]
            full_indices = [item for sublist in all_indices for item in sublist]

            if len(full_preds) > 0:
                preds_df = pl.DataFrame({
                    "row_idx": full_indices,
                    "predicted_fitness": full_preds
                })

                preds_df = preds_df.with_columns(pl.col("row_idx").cast(pl.UInt32))
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


# --- 6. Main Training Function for ESM2 ---
def train_esm_model(train_df_raw, val_df_raw, config: ConfigESM, num_workers=8, epochs=5):
    print(f"Train samples: {len(train_df_raw)}, Validation samples: {len(val_df_raw)}")

    config_dict = asdict(config)
    if "wandb_token" in config_dict:
        del config_dict["wandb_token"]
    print(f"The config: {json.dumps(config_dict)}")

    # 1. Tokenizer
    # ESM tokenizers do not require `do_lower_case=False`
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

    # === Apply layer freezing ===
    freeze_depth = getattr(config, 'freeze_layers', 0)
    freeze_esm_layers(composer_model.model, freeze_depth)

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

    # 6. Callbacks
    html_callback = InteractiveReportCallbackESM( # Changed to ESM specific callback
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

    callbacks = [LRMonitor(), OptimizerMonitor(), html_callback, early_stopper, checkpoint_saver]

    gc = GradientClipping(clipping_type='norm', clipping_threshold=1.0)

    wandb_logger = WandBLogger(
        project=config.project_name,
        init_kwargs={"config": config_dict},
        log_artifacts=False,
        rank_zero_only=True
    )

    # 7. Trainer
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

