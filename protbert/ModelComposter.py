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
from composer.callbacks import EarlyStopper, LRMonitor, OptimizerMonitor
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from torch.utils.data import Dataset, DataLoader
# Torchmetrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef, R2Score, \
    MeanAbsolutePercentageError, MatthewsCorrCoef, F1Score
from transformers import BertModel, BertTokenizer
from transformers import DataCollatorWithPadding


# Configuration
@dataclass
class Config:
    project_name: str = "protein-mutation-prediction-protbert"
    pretrained_model: str = "Rostlab/prot_bert"
    wandb_token: str = ""
    max_length: int = 1024
    batch_size: int = 48
    learning_rate: float = 5e-5
    hidden_dropout_prob: float = 0.1
    epochs: float = 15
    freeze_layers: int = 3
    early_stopping_patience: int = 2
    early_stopping_delta: float = 0.001
    step_validation: int = 1500
    base_dir: str = "./"
    seq_window_size: int = 255


def prepare_data_dynamic(df: pl.DataFrame, max_total_length: int = 1024, window_size: int = 255):
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


# --- 1. Dataset ---
class ProteinMutationDataset(Dataset):
    def __init__(self, processed_df: pl.DataFrame, tokenizer, max_length=1024):
        """
        Args:
            processed_df: Výstup z funkce prepare_data_with_polars
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

        self.wt_seqs = processed_df["clean_wt"].to_list()
        self.mut_seqs = processed_df["clean_mut"].to_list()
        self.targets = processed_df["fitness"].to_list()

        self.max_length = max_length
        self.ids = list(range(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def add_spaces(self, seq):
        return " ".join(seq)

    def __getitem__(self, idx):
        # 1. Vytažení stringů (bleskové)
        seq_wt = self.add_spaces(self.wt_seqs[idx])
        seq_mut = self.add_spaces(self.mut_seqs[idx])
        target = self.targets[idx]
        row_id = self.ids[idx]

        # 2. Tokenizace
        # Zde už nepoužíváme truncation ani složité logiky, data jsou připravena.
        # Používáme padding=False pro Dynamic Padding v Collatoru.
        inputs = self.tokenizer(
            seq_wt,
            seq_mut,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # Změna: Statický padding jako v Model.py
            return_token_type_ids=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.float),
            'row_idx': torch.tensor(row_id, dtype=torch.long)
        }

# --- 2. Model (Correct Architecture) ---
class ProteinMutationModel(nn.Module):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        self.tokenizer = tokenizer

        if len(tokenizer) > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(len(tokenizer))

        # hidden_size * 2 (Jen WT a MUT, bez CLS, přesně jako v Model.py)
        input_dim = self.bert.config.hidden_size * 2

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

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state

        sep_token_id = self.tokenizer.sep_token_id
        sep_mask = (input_ids == sep_token_id)
        sep_indices = torch.argmax(sep_mask.float(), dim=1)

        arange_mask = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)

        wt_mask = (arange_mask > 0) & (arange_mask < sep_indices.unsqueeze(1)) & attention_mask.bool()
        mut_mask = (arange_mask > sep_indices.unsqueeze(1)) & attention_mask.bool()

        wt_sum = (sequence_output * wt_mask.unsqueeze(-1).float()).sum(dim=1)
        mut_sum = (sequence_output * mut_mask.unsqueeze(-1).float()).sum(dim=1)

        wt_count = wt_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mut_count = mut_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        wt_repr = wt_sum / wt_count
        mut_repr = mut_sum / mut_count

        combined_embeddings = torch.cat([wt_repr, mut_repr], dim=1)
        return self.regressor_head(combined_embeddings)


# --- 3. Composer Wrapper ---
class ComposerProteinModel(ComposerModel):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()

        self.classification_threshold = 0.0

        self.model = ProteinMutationModel(pretrained_model_name, tokenizer)

        # Používáme HuberLoss místo MSELoss - je méně citlivá na odlehlé hodnoty (šum v datech)
        self.criterion = nn.HuberLoss(delta=1.0)

        self.val_metrics = nn.ModuleDict({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'mape': MeanAbsolutePercentageError(),  # Přidáno MAPE
            'pearson': PearsonCorrCoef(),
            'spearman': SpearmanCorrCoef(),
            'r2': R2Score(),
            'f1': F1Score(task="binary"),
            'mcc': MatthewsCorrCoef(task="binary"),
        })

    def forward(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids')
        )

    def loss(self, outputs, batch):
        # Squeeze je důležitý pro srovnání rozměrů [B, 1] vs [B]
        return self.criterion(outputs.squeeze(), batch["labels"])

    def get_metrics(self, is_train: bool = False):
        if is_train:
            return {}
        return self.val_metrics

    def update_metric(self, batch, outputs, metric):
        """
        Zde se děje magie: Rozlišíme, zda jde o regresi nebo klasifikaci.
        """
        targets = batch["labels"]

        predictions = outputs.squeeze()  # Zajistíme tvar [Batch_Size]

        regression_metrics = (
            MeanSquaredError,
            MeanAbsoluteError,
            MeanAbsolutePercentageError,
            PearsonCorrCoef,
            SpearmanCorrCoef,
            R2Score
        )

        # 3. Rozdělení logiky
        if isinstance(metric, regression_metrics):
            # Regrese -> posíláme přímo čísla
            metric.update(predictions, targets)
        else:
            # Vše ostatní (F1, MCC, Accuracy) -> Binarizujeme na 0 a 1
            binary_preds = (predictions > self.classification_threshold).long()
            binary_targets = (targets > self.classification_threshold).long()
            metric.update(binary_preds, binary_targets)


# --- 4. Helper funkce ---

# A) Funkce pro zmrazení vrstev (z původního skriptu)
def freeze_bert_layers(model, num_layers_to_freeze):
    """
    Zmrazí embeddingy a prvních N vrstev encoderu.
    """
    if num_layers_to_freeze == 0:
        return

    print(f"INFO: Freezing embeddings and first {num_layers_to_freeze} BERT layers.")

    # 1. Zmrazit Embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # 2. Zmrazit Encoder vrstvy
    for i in range(num_layers_to_freeze):
        if i < len(model.bert.encoder.layer):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        else:
            print(f"WARNING: Cannot freeze layer {i}, model has only {len(model.bert.encoder.layer)} layers.")


# B) HTML Report Generator
import wandb


def log_interactive_report_polars(df: pl.DataFrame, table_name: str, step: int = None):
    """
    Generuje HTML report z Polars DataFrame.
    Umožňuje měnit thresholdy v prohlížeči a dynamicky počítat F1/MCC.
    """

    # 1. PŘÍPRAVA DAT (Select + Rename pro konzistenci s JS)
    # Přejmenujeme sloupce na malá písmena, aby seděly s logikou v JavaScriptu (row.predicted_fitness)
    df_export = df.select([
        "wt_sequence", "mut_sequence", "mutation",
        "cath_class", "cath_arch", "cath_topology", "cath_homology", "data_source",
        "predicted_fitness", "actual_fitness"
    ]).with_columns(
        (pl.col("predicted_fitness") - pl.col("actual_fitness")).alias("diff")
    )

    json_data = df_export.write_json()

    # 2. HTML Šablona
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine-dark.css" />
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px; }}
            .container {{ max-width: 100%; margin: 0 auto; }}

            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            h2 {{ margin: 0; color: #fff; }}

            .controls-panel {{ background-color: #1e1e1e; padding: 15px; margin-bottom: 20px; border-radius: 8px; display: flex; gap: 20px; align-items: center; border: 1px solid #333; }}

            .input-group {{ display: flex; align-items: center; gap: 10px; }}
            input[type=number] {{ background: #333; color: white; border: 1px solid #555; padding: 5px; border-radius: 4px; width: 70px; }}

            .btn {{ padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s; color: white; }}
            .btn-recalc {{ background-color: #2196F3; }}
            .btn-recalc:hover {{ background-color: #1976D2; }}
            .btn-download {{ background-color: #4CAF50; }}
            .btn-download:hover {{ background-color: #388E3C; }}

            .metric-container {{ display: flex; gap: 15px; flex-wrap: wrap; justify-content: space-between; width: 100%; }}
            .metric-card {{ background-color: #252525; padding: 10px 15px; border-radius: 5px; min-width: 100px; text-align: center; border: 1px solid #333; flex-grow: 1; }}
            .metric-label {{ font-size: 0.8em; color: #aaa; margin-bottom: 5px; display: block; }}
            .metric-val {{ font-size: 1.3em; color: #4e9af1; font-weight: bold; font-family: monospace; }}

            #myGrid {{ height: 650px; width: 100%; border-radius: 8px; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Interactive Validation Report</h2>
                <button class="btn btn-download" onclick="downloadCSV()">Download CSV</button>
            </div>

            <div class="controls-panel">
                <div class="input-group">
                    <label>Negative (≤):</label>
                    <input type="number" id="neg-input" step="0.01" value="-0.05">
                </div>
                <div class="input-group">
                    <label>Positive (>):</label>
                    <input type="number" id="pos-input" step="0.01" value="0.05">
                </div>
                <button class="btn btn-recalc" onclick="recalcAll()">Recalculate Metrics</button>
            </div>

            <div class="controls-panel">
                <div class="metric-container">
                    <div class="metric-card"><span class="metric-label">MSE</span><div id="val-mse" class="metric-val">-</div></div>
                    <div class="metric-card"><span class="metric-label">Pearson</span><div id="val-pearson" class="metric-val">-</div></div>
                    <div class="metric-card"><span class="metric-label">Pos F1</span><div id="val-f1-pos" class="metric-val">-</div></div>
                    <div class="metric-card"><span class="metric-label">Pos MCC</span><div id="val-mcc-pos" class="metric-val">-</div></div>
                    <div class="metric-card"><span class="metric-label">Accuracy</span><div id="val-acc" class="metric-val">-</div></div>
                </div>
            </div>

            <div id="myGrid" class="ag-theme-alpine-dark"></div>
        </div>

        <script>
            const rawData = {json_data};
            let gridApi;

            // --- 1. Definice Sloupců ---
            let columnDefs = [];
            if (rawData.length > 0) {{
                const keys = Object.keys(rawData[0]);

                // Prioritní sloupce (čísla)
                const priority = ['predicted_fitness', 'actual_fitness', 'fitness'];

                priority.forEach(key => {{
                    if (keys.includes(key)) {{
                        columnDefs.push({{ field: key, filter: 'agNumberColumnFilter', sortable: true, resizable: true, width: 110 }});
                    }}
                }});

                // Klasifikační sloupce (obarvené)
                columnDefs.push({{ 
                    field: "predicted_class", headerName: "Pred Class", width: 120, cellStyle: params => {{
                        if (params.value === 'Positive') return {{color: '#4caf50', fontWeight: 'bold'}};
                        if (params.value === 'Negative') return {{color: '#f44336'}};
                        return {{color: '#ff9800'}};
                    }}
                }});
                columnDefs.push({{ 
                    field: "actual_class", headerName: "Act Class", width: 120, cellStyle: params => {{
                        if (params.value === 'Positive') return {{color: '#4caf50', fontWeight: 'bold'}};
                        if (params.value === 'Negative') return {{color: '#f44336'}};
                        return {{color: '#ff9800'}};
                    }}
                }});

                // Ostatní textové sloupce
                keys.forEach(key => {{
                    if (!priority.includes(key) && key !== 'row_idx' && key !== 'predicted_class' && key !== 'actual_class') {{
                        columnDefs.push({{ field: key, filter: 'agTextColumnFilter', sortable: true, resizable: true }});
                    }}
                }});
            }}

            function recalcAll() {{
                if (!gridApi) return;

                const negThresh = parseFloat(document.getElementById('neg-input').value);
                const posThresh = parseFloat(document.getElementById('pos-input').value);

                let tp=0, tn=0, fp=0, fn=0, correct=0;
                let sumSq=0, sumXY=0, sumX=0, sumY=0, sumX2=0, sumY2=0;

                // Iterace přes data a update tříd "in-place"
                rawData.forEach(row => {{
                    const pred = row.predicted_fitness;
                    const act = row.actual_fitness;

                    // Regrese
                    sumSq += (pred-act)**2;
                    sumX += act; sumY += pred;
                    sumXY += act*pred;
                    sumX2 += act**2; sumY2 += pred**2;

                    // Klasifikace
                    const getCls = v => v > posThresh ? 'Positive' : (v <= negThresh ? 'Negative' : 'Neutral');
                    const pCls = getCls(pred);
                    const aCls = getCls(act);

                    // Update hodnot v objektu pro Grid
                    row.predicted_class = pCls;
                    row.actual_class = aCls;

                    if(pCls === aCls) correct++;

                    // MCC Metrics (Positive vs Rest)
                    const pP = pCls === 'Positive';
                    const aP = aCls === 'Positive';
                    if(pP && aP) tp++;
                    if(pP && !aP) fp++;
                    if(!pP && !aP) tn++;
                    if(!pP && aP) fn++;
                }});

                // Výpočet statistik
                const n = rawData.length;
                const mse = sumSq/n;
                const num = n*sumXY - sumX*sumY;
                const den = Math.sqrt((n*sumX2 - sumX**2)*(n*sumY2 - sumY**2));
                const pearson = den===0 ? 0 : num/den;

                const precision = tp+fp>0 ? tp/(tp+fp) : 0;
                const recall = tp+fn>0 ? tp/(tp+fn) : 0;
                const f1 = precision+recall>0 ? 2*precision*recall/(precision+recall) : 0;

                const mccDen = Math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
                const mcc = mccDen===0 ? 0 : (tp*tn - fp*fn)/mccDen;

                // Update UI
                document.getElementById('val-mse').innerText = mse.toFixed(4);
                document.getElementById('val-pearson').innerText = pearson.toFixed(3);
                document.getElementById('val-f1-pos').innerText = f1.toFixed(3);
                document.getElementById('val-mcc-pos').innerText = mcc.toFixed(3);
                document.getElementById('val-acc').innerText = (correct/n*100).toFixed(1)+'%';

                // --- CRITICAL FIX: Refresh Cells místo setRowData ---
                // Toto zajistí, že se změny v 'predicted_class' a 'actual_class' projeví v tabulce
                gridApi.refreshCells({{ force: true }});
            }}

            function downloadCSV() {{
                if (gridApi) {{
                    gridApi.exportDataAsCsv({{ fileName: 'validation_report.csv' }});
                }}
            }}

            const gridOptions = {{
                rowData: rawData,
                columnDefs: columnDefs,
                defaultColDef: {{ sortable: true, filter: true, resizable: true }},
                onGridReady: params => {{ 
                    gridApi = params.api; 
                    recalcAll(); // První výpočet hned po startu
                }}
            }};

            document.addEventListener('DOMContentLoaded', () => {{
                agGrid.createGrid(document.querySelector('#myGrid'), gridOptions);
            }});
        </script>
    </body>
    </html>
    """

    log_payload = {table_name: wandb.Html(html_content)}
    if step is not None:
        wandb.log(log_payload, step=step)
    else:
        wandb.log(log_payload)


from composer.utils import dist


class InteractiveReportCallback(Callback):
    def __init__(self, log_function, val_original_df: pl.DataFrame):
        self.log_func = log_function

        # Přidání row_idx do originálního DF (jak jsme řešili minule)
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
        # 1. Musíme sesbírat data ze všech GPU
        # Vytvoříme seznamy, kam se uloží data od ostatních
        all_preds = [None for _ in range(dist.get_world_size())]
        all_indices = [None for _ in range(dist.get_world_size())]
        all_targets = [None for _ in range(dist.get_world_size())]

        # 2. Synchronizace (poslání dat)
        # Toto je náročná operace, posílá data přes síť mezi kartami
        torch.distributed.all_gather_object(all_preds, self.preds)
        torch.distributed.all_gather_object(all_indices, self.indices)
        torch.distributed.all_gather_object(all_targets, self.targets)

        # 3. Zpracování pouze na Rank 0 (Master)
        if dist.get_global_rank() == 0:
            # Sloučení seznamů listů do jednoho dlouhého listu
            # all_preds je nyní např: [[gpu0_data], [gpu1_data], ...]
            full_preds = [item for sublist in all_preds for item in sublist]
            full_indices = [item for sublist in all_indices for item in sublist]

            if len(full_preds) > 0:
                # Vytvoření DF se VŠEMI daty
                preds_df = pl.DataFrame({
                    "row_idx": full_indices,
                    "predicted_fitness": full_preds
                })

                # Typová konverze pro join
                preds_df = preds_df.with_columns(pl.col("row_idx").cast(pl.UInt32))
                self.val_original_df = self.val_original_df.with_columns(pl.col("row_idx").cast(pl.UInt32))

                # JOIN s původními daty
                final_df = self.val_original_df.join(preds_df, on="row_idx", how="inner")

                if "fitness" in final_df.columns and "actual_fitness" not in final_df.columns:
                    final_df = final_df.rename({"fitness": "actual_fitness"})

                # Logování
                table_name = f"val_results_epoch_{int(state.timestamp.epoch)}"
                current_step = int(state.timestamp.batch)

                try:
                    self.log_func(final_df, table_name=table_name, step=current_step)
                    print(f"Interactive Report generated with ALL {len(final_df)} samples.")
                except Exception as e:
                    print(f"Error generating report: {e}")

        # Vyčištění paměti na všech GPU
        self.preds = []
        self.indices = []


# --- 6. Hlavní funkce ---
def train_full_model(train_df_raw, val_df_raw, config, num_workers=8):
    print(f"Train samples: {len(train_df_raw)}, Validation samples: {len(val_df_raw)}")

    config_dict = asdict(config)

    if "wandb_token" in config_dict:
        del config_dict["wandb_token"]

    print(f"The config: {json.dumps(config_dict)}")
    # 1. Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)

    print(f"Preprocessing Training Data (Window Size: {config.seq_window_size})...")
    train_df = prepare_data_dynamic(train_df_raw, max_total_length=config.max_length, window_size=config.seq_window_size)

    print(f"Preprocessing Validation Data (Window Size: {config.seq_window_size})...")
    validation_df = prepare_data_dynamic(val_df_raw, max_total_length=config.max_length, window_size=config.seq_window_size)

    if isinstance(validation_df, pd.DataFrame):
        validation_df = pl.from_pandas(validation_df)

    # 2. Model
    composer_model = ComposerProteinModel(config.pretrained_model, tokenizer)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Odstraněno pro shodu s Model.py

    # === Aplikace zmrazení vrstev (z původního skriptu) ===
    freeze_depth = getattr(config, 'freeze_layers', 0)
    freeze_bert_layers(composer_model.model, freeze_depth)

    # 3. DataLoaders
    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    train_sampler = dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  # collate_fn=data_collator, # Odstraněno
                                  pin_memory=True,
                                  num_workers=num_workers)

    val_dataset = ProteinMutationDataset(validation_df, tokenizer, config.max_length)

    val_sampler_subset = dist.get_sampler(val_dataset, shuffle=True, drop_last=True)
    val_loader_subset = DataLoader(val_dataset,
                                   batch_size=config.batch_size,
                                   sampler=val_sampler_subset,
                                   drop_last=True,
                                   # collate_fn=data_collator, # Odstraněno
                                   pin_memory=True,
                                   num_workers=num_workers)

    val_sampler_full = dist.get_sampler(val_dataset, shuffle=False, drop_last=False)
    val_loader_full = DataLoader(val_dataset,
                                 batch_size=config.batch_size,
                                 sampler=val_sampler_full,
                                 drop_last=False,
                                 # collate_fn=data_collator, # Odstraněno
                                 pin_memory=True,
                                 num_workers=num_workers)

    # 4. Evaluators
    eval_frequent = Evaluator(
        label="frequent_val",
        dataloader=val_loader_subset,
        metric_names=['mse', 'pearson'],
        subset_num_batches=20,
        eval_interval="500ba"
    )

    eval_full = Evaluator(
        label="full_val",
        dataloader=val_loader_full,
        eval_interval="1ep"
    )

    # 5. Optimizer & Scheduler
    optimizer = DecoupledAdamW(composer_model.parameters(), lr=config.learning_rate)
    # Změna: Ještě delší warmup (20%) - model se déle učí s rostoucím LR, což oddálí stagnaci
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup="0.20dur", alpha_f=0.01)

    # 6. Callbacks
    html_callback = InteractiveReportCallback(
        log_function=log_interactive_report_polars,
        val_original_df=validation_df
    )

    early_stopper = EarlyStopper(
        monitor="mcc",
        dataloader_label="full_val",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_delta
    )

    callbacks = [LRMonitor(), OptimizerMonitor(), html_callback, early_stopper]

    gc = GradientClipping(clipping_type='norm', clipping_threshold=1.0)

    save_path = os.path.join(config.base_dir, 'checkpoints')

    # 7. Trainer
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=[eval_frequent, eval_full],
        max_duration=f"{config.epochs}ep",
        optimizers=optimizer,
        schedulers=scheduler,

        algorithms=[gc],
        # ==============================================

        # === Seed ===
        seed=42,

        save_folder=save_path,

        parallelism_config={'ddp': {'find_unused_parameters': True}},
        callbacks=callbacks,
        loggers=[WandBLogger(project=config.project_name, init_kwargs={"config": config_dict})],

        save_filename='model_epoch_homology_{epoch}.pt',
        save_latest_filename="latest",
        save_overwrite=True,
        save_interval="1ep",

        device="gpu",
        precision="amp_bf16"
    )

    print(f"Starting training (Freezing {freeze_depth} layers, Grad Clip 1.0, Seed 42)...")
    trainer.fit()
    print("Training finished successfully!")
