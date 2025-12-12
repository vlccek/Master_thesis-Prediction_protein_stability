import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb  # Nutné pro logování HTML

# Composer imports
from composer import Trainer, Callback, State, Logger
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from composer.callbacks import EarlyStopper, LRMonitor, OptimizerMonitor
from composer.loggers import WandBLogger
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef, R2Score
from composer.utils import dist
from composer.optim.scheduler import LinearWithWarmupScheduler, CosineAnnealingWithWarmupScheduler


# --- 1. Dataset ---
class ProteinMutationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

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


# --- 2. Model (Correct Architecture) ---
class ProteinMutationModel(nn.Module):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()
        # add_pooling_layer=False je klíčové pro DDP, aby se nevytvářely nepoužité parametry
        self.bert = BertModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        sep_token_id = self.tokenizer.sep_token_id
        sep_mask = (input_ids == sep_token_id)
        sep_indices = torch.argmax(sep_mask.float(), dim=1)

        no_sep_found = (sep_indices == 0) & ~sep_mask[:, 0]
        mid_points = torch.full_like(sep_indices, input_ids.size(1) // 2)
        sep_indices[no_sep_found] = mid_points[no_sep_found]

        seq_len = input_ids.size(1)
        arange_mask = torch.arange(seq_len, device=input_ids.device)[None, :].expand(input_ids.size(0), -1)

        wt_padding_mask = (arange_mask > 0) & (arange_mask < sep_indices.unsqueeze(1))
        mut_padding_mask = (arange_mask > sep_indices.unsqueeze(1))

        wt_mask = wt_padding_mask & attention_mask.bool()
        mut_mask = mut_padding_mask & attention_mask.bool()

        wt_mask_expanded = wt_mask.unsqueeze(-1).expand_as(sequence_output)
        mut_mask_expanded = mut_mask.unsqueeze(-1).expand_as(sequence_output)

        wt_sum = (sequence_output * wt_mask_expanded).sum(dim=1)
        mut_sum = (sequence_output * mut_mask_expanded).sum(dim=1)

        wt_count = wt_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mut_count = mut_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        wt_repr = wt_sum / wt_count
        mut_repr = mut_sum / mut_count

        combined_embeddings = torch.cat([wt_repr, mut_repr], dim=1)
        return self.regressor_head(combined_embeddings).squeeze()


# --- 3. Composer Wrapper ---
class ComposerProteinModel(ComposerModel):
    def __init__(self, pretrained_model_name, tokenizer):
        super().__init__()
        self.model = ProteinMutationModel(pretrained_model_name, tokenizer)

        # Přidány zpět všechny metriky
        self.val_metrics = nn.ModuleDict({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'pearson': PearsonCorrCoef(),
            'spearman': SpearmanCorrCoef(),
            'r2': R2Score()
        })

    def forward(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

    def loss(self, outputs, batch):
        return F.mse_loss(outputs, batch['targets'])

    def get_metrics(self, is_train: bool = False):
        if is_train:
            return {}
        return self.val_metrics

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs, batch['targets'])


# --- 4. HTML Report Generator (Tvá funkce) ---
def log_interactive_dataframe_to_wandb(df: pd.DataFrame, table_name: str = "validation_results_js_interactive", step: int = None):
    """
    Generuje a loguje plně interaktivní HTML report.
    """
    df = df.copy()
    # Přejmenování sloupců pokud je potřeba
    if 'target' in df.columns and 'actual_fitness' not in df.columns:
        df.rename(columns={'target': 'actual_fitness'}, inplace=True)

    # Výpočet chyby pro vizualizaci
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

    # Nahrazení NaN pro JSON
    df_clean = df.where(pd.notnull(df), None)
    df_json = df_clean.to_json(orient='records')

    # HTML Šablona
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
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; padding: 20px; background-color: #121212; color: #e0e0e0; }}
            .header-container {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
            h1 {{ color: #ffffff; margin: 0; }}
            h2 {{ color: #ffffff; }}
            #download-btn {{ padding: 8px 15px; font-size: 14px; background-color: #4e9af1; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.2s; }}
            #download-btn:hover {{ background-color: #3a75c4; }}
            .metric-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .metric-card {{ padding: 15px; border: 1px solid #333; border-radius: 8px; background-color: #1e1e1e; }}
            .metric-title {{ font-weight: bold; color: #aaa; display: block; margin-bottom: 5px; }}
            .metric-value {{ font-size: 1.4em; color: #4e9af1; font-family: monospace; }}
            .confusion-matrix-container {{ margin-bottom: 30px; }}
            .confusion-matrix {{ display: grid; grid-template-columns: 1.5fr repeat(3, 1fr); gap: 5px; text-align: center; max-width: 700px; background-color: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; }}
            .cm-cell {{ padding: 12px; font-size: 1.1em; font-family: sans-serif; display: flex; align-items: center; justify-content: center; border-radius: 5px; }}
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
            <div class="metric-card"><span class="metric-title">Zobrazeno vzorků:</span> <span id="samples-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">Platných pro výpočet:</span> <span id="valid-samples-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">MSE:</span> <span id="mse-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">MAE:</span> <span id="mae-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">RMSE:</span> <span id="rmse-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">R²:</span> <span id="r2-value" class="metric-value">-</span></div>
            <div class="metric-card"><span class="metric-title">Pearson Corr:</span> <span id="pearson-value" class="metric-value">-</span></div>
        </div>
        <div class="confusion-matrix-container">
            <h2>Matice záměn (dle filtru)</h2>
            <div class="confusion-matrix">
                <div class="cm-cell cm-empty"></div>
                <div class="cm-cell cm-header" style="grid-column: 2 / 5;"><b>Skutečná třída (Actual)</b></div>
                <div class="cm-cell cm-empty"></div>
                <div class="cm-cell cm-header">Pozitivní</div>
                <div class="cm-cell cm-header">Neutrální</div>
                <div class="cm-cell cm-header">Negativní</div>
                <div class="cm-cell cm-axis-label">Predikovaná: Pozitivní</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_pos-act_pos">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_pos-act_neu">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_pos-act_neg">0</div>
                <div class="cm-cell cm-axis-label">Predikovaná: Neutrální</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neu-act_pos">0</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_neu-act_neu">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neu-act_neg">0</div>
                <div class="cm-cell cm-axis-label">Predikovaná: Negativní</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neg-act_pos">0</div>
                <div class="cm-cell cm-data cm-error"   id="cm-pred_neg-act_neu">0</div>
                <div class="cm-cell cm-data cm-correct" id="cm-pred_neg-act_neg">0</div>
            </div>
        </div>
        <div id="myGrid" class="ag-theme-alpine-dark"></div>
        <script>
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
                    if (strValue.includes(',') || strValue.includes('"') || strValue.includes('\\n')) {{ return `"${{strValue.replace(/"/g, '""')}}"`; }}
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

    log_payload = {table_name: wandb.Html(html_content)}

    if step is not None:
        wandb.log(log_payload, step=step)
    else:
        wandb.log(log_payload)


# --- 5. Callback pro sběr dat a volání HTML generátoru ---
class InteractiveReportCallback(Callback):
    def __init__(self, log_function):
        self.log_func = log_function
        self.preds = []
        self.targets = []

    def eval_batch_end(self, state: State, logger: Logger):
        # Sběr dat z aktuálního batche
        # Přesuneme na CPU a převedeme na numpy
        outputs = state.outputs.detach().cpu().numpy()
        targets = state.batch['targets'].detach().cpu().numpy()

        # V DDP režimu by správně měl sbírat data jen rank 0,
        # nebo bychom museli dělat all_gather.
        # Pro vizualizaci stačí, když rank 0 zaloguje "svou část" validace,
        # nebo (pokud je validace malá) to necháme takto a rank 0 posbírá co vidí.
        # Zde sbíráme lokálně na každém procesu, ale logovat budeme jen na ranku 0.
        self.preds.extend(outputs)
        self.targets.extend(targets)

    def eval_end(self, state: State, logger: Logger):
        # Logujeme pouze na hlavním procesu (Rank 0)
        if dist.get_global_rank() == 0:
            if len(self.preds) > 0:
                df = pd.DataFrame({
                    'predicted_fitness': self.preds,
                    'actual_fitness': self.targets
                })

                # Název tabulky podle epochy
                table_name = f"val_results_epoch_{int(state.timestamp.epoch)}"

                try:
                    self.log_func(df, table_name=table_name, step=int(state.timestamp.epoch))
                    print(f"HTML Report '{table_name}' generated and logged.")
                except Exception as e:
                    print(f"Error generating HTML report: {e}")

            # Vyčištění listů pro další epochu
            self.preds = []
            self.targets = []


# --- 6. Hlavní funkce ---
def train_full_model(train_df, validation_df, config):
    print(f"Train samples: {len(train_df)}, Validation samples: {len(validation_df)}")

    # 1. Tokenizer & [SEP]
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
    if "[SEP]" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[SEP]"])
        print("INFO: Token [SEP] was added to tokenizer")

    # 2. Model
    model = ComposerProteinModel(config.pretrained_model, tokenizer)

    # 3. DataLoaders
    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    val_dataset = ProteinMutationDataset(validation_df, tokenizer, config.max_length)

    train_sampler = dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = dist.get_sampler(val_dataset, shuffle=False, drop_last=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=4,
        drop_last=False
    )

    # 4. Optimizer
    optimizer = DecoupledAdamW(model.parameters(), lr=config.learning_rate)

    # 5. Callbacks
    # Přidán náš nový callback
    html_callback = InteractiveReportCallback(log_interactive_dataframe_to_wandb)

    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup="0.1dur",  # 10% Warmup
        alpha_f=0.01  # Klesne na 1% původní LR
    )

    callbacks = [
        LRMonitor(),
        OptimizerMonitor(),
        html_callback
    ]

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        max_duration=f"{config.epochs}ep",
        optimizers=optimizer,
        schedulers=scheduler,
        parallelism_config={
            'ddp': {
                'find_unused_parameters': True
            }
        },

        callbacks=callbacks,
        loggers=[WandBLogger(project=config.project_name)],
        eval_interval="1ep",

        save_folder=getattr(config, 'save_folder', './checkpoints'),
        save_filename='model_epoch_{epoch}.pt',
        save_overwrite=True,

        device="gpu",
        precision="amp_fp16"
    )

    print("Starting training with CORRECT architecture & HTML Reports...")
    trainer.fit()
    print("Training finished successfully!")