import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from composer import Evaluator

# 1. Importy z Composeru a Torchmetrics
from composer import Trainer, Callback, State, Logger
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from composer.callbacks import EarlyStopper, LRMonitor, OptimizerMonitor
from composer.loggers import WandBLogger
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef, R2Score, \
    ExplainedVariance
from composer.utils import dist

# Import našich definic z ostatních souborů
from Model import Config, SingleBertPredictor, ProteinMutationDataset, log_interactive_dataframe_to_wandb




class ValidationTableCallback(Callback):

    def __init__(self, validation_df: pd.DataFrame):
        self.validation_df = validation_df

    def eval_end(self, state: State, logger: Logger):
        if dist.get_global_rank() == 0:
            print("INFO: Generating validation table...")

            try:
                all_preds = []

                if not hasattr(state, 'outputs') or state.outputs is None:
                    print("WARNING: state.outputs is None or does not exist")
                    return

                # Zpracování state.outputs
                for batch_output in state.outputs:
                    if batch_output is None:
                        continue

                    if isinstance(batch_output, torch.Tensor):
                        # Pokud je tenzor skalár (0-dimenzionální), převedeme ho na 1D tenzor
                        if batch_output.dim() == 0:
                            batch_output = batch_output.unsqueeze(0)
                        all_preds.append(batch_output.detach().cpu())
                    elif isinstance(batch_output, (list, tuple)):
                        for item in batch_output:
                            if isinstance(item, torch.Tensor):
                                # Stejně zpracujeme skalární tenzory v seznamech
                                if item.dim() == 0:
                                    item = item.unsqueeze(0)
                                all_preds.append(item.detach().cpu())
                    else:
                        print(f"WARNING: Unexpected type in batch_output: {type(batch_output)}")

                if len(all_preds) == 0:
                    print("WARNING: No predictions collected from state.outputs")
                    return

                # Spojíme všechny predikce - nyní by všechny měly být alespoň 1D
                try:
                    preds_tensor = torch.cat(all_preds)
                except Exception as e:
                    print(f"ERROR concatenating tensors: {e}")
                    print(f"Shapes of tensors: {[t.shape for t in all_preds]}")
                    return

                # Zkontrolujeme, zda máme dostatek vzorků
                num_results = min(len(preds_tensor), len(self.validation_df))

                if num_results == 0:
                    print("WARNING: No results to display in validation table")
                    return

                results_df = self.validation_df.iloc[:num_results].copy()
                results_df['predicted_fitness'] = preds_tensor.numpy()[:num_results]

                log_interactive_dataframe_to_wandb(
                    df=results_df,
                    table_name=f"validation_results_epoch_{state.timestamp.epoch.value}"
                )
                print(f"INFO: Validation table with {num_results} samples logged to W&B")

            except Exception as e:
                print(f"ERROR creating validation table: {e}")
                import traceback
                traceback.print_exc()


class ComposerBertModel(ComposerModel):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.module = SingleBertPredictor(config, tokenizer)

        # Metriky pro trénování
        self.metrics_train = nn.ModuleDict({
            'pearson_corr': PearsonCorrCoef()
        })

        # Metriky pro validaci
        self.metrics_val = nn.ModuleDict({
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'pearson_corr': PearsonCorrCoef(),
            'spearman_corr': SpearmanCorrCoef(),
            'r2_score': R2Score(),
            'explained_variance': ExplainedVariance(),
            'rmse': MeanSquaredError(squared=False),
        })

    def forward(self, batch):
        return self.module(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    def loss(self, outputs, batch):
        return F.mse_loss(outputs, batch['targets'])

    def get_metrics(self, is_train: bool = False):
        return self.metrics_train if is_train else self.metrics_val

    def update_metric(self, batch, outputs, metric):
        """Aktualizuje konkrétní metriku"""
        targets = batch['targets']
        metric.update(outputs, targets)

    def eval_forward(self, batch, outputs=None):
        """Pouze inference - bez aktualizace metrik"""
        return self.forward(batch)


def train_single_model(train_df, testing_df, config):
    print(f"Train samples: {len(train_df)}, Test samples: {len(testing_df)}")

    if len(testing_df) == 0:
        print("WARNING: No validation data provided!")
        return

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
    composer_model = ComposerBertModel(config, tokenizer)

    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    train_sampler = dist.get_sampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  sampler=train_sampler,
                                  num_workers=6,
                                  drop_last=True
                                  )

    testing_dataset = ProteinMutationDataset(testing_df, tokenizer, config.max_length)
    testing_sampler = dist.get_sampler(testing_dataset, shuffle=False, drop_last=False)

    testing_dataloader = DataLoader(testing_dataset,
                                    batch_size=config.batch_size,
                                    sampler=testing_sampler,
                                    num_workers=6,
                                    drop_last=False
                                    )

    loggers = []
    wandb_logger = WandBLogger(project="protein-mutation-prediction-protbert")
    # loggers.append(wandb_logger)

    print(f"INFO: Preparing to freeze the first {config.freeze_layers} layers of BERT.")

    frozen_prefixes = []
    if config.freeze_layers > 0:
        for i in range(config.freeze_layers):
            frozen_prefixes.append(f'module.bert.encoder.layer.{i}.')

    params_to_optimize = []
    total_param_count = 0
    trainable_param_count = 0

    print("--- Parameter Freeze/Train Status ---")
    for name, param in composer_model.named_parameters():
        total_param_count += param.numel()
        is_frozen = any(name.startswith(prefix) for prefix in frozen_prefixes)

        if is_frozen:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_optimize.append(param)
            trainable_param_count += param.numel()

    print(f"Total parameters:      {total_param_count}")
    print(f"Trainable parameters:  {trainable_param_count}")
    if total_param_count == trainable_param_count and (config.freeze_layers > 0 or len(frozen_prefixes) > 0):
        print("\n!!! WARNING: Parameter filtering FAILED. Check prefixes. !!!\n")
    else:
        print(f"Successfully frozen:   {total_param_count - trainable_param_count} parameters.")
    print("------------------------------------")

    optimizer = DecoupledAdamW(params_to_optimize, lr=config.learning_rate)

    partial_evaluator = Evaluator(
        label='partial',
        dataloader=testing_dataloader,
        metric_names=['mse', 'mae', 'pearson_corr', 'spearman_corr', 'r2_score'],
        subset_num_batches=max(int(0.1 * len(testing_dataloader)), config.batch_size),
        eval_interval='10ba'
    )

    # Evaluator pro úplnou validaci (kompletní)
    full_evaluator = Evaluator(
        label='full',
        dataloader=testing_dataloader,
        metric_names=['mse', 'mae', 'pearson_corr', 'spearman_corr', 'r2_score'],
        eval_interval='1ep'
    )

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=[partial_evaluator, full_evaluator],
        max_duration=f"{config.epochs}ep",
        optimizers=optimizer,
        parallelism_config={
            'ddp': {
                'find_unused_parameters': True,
                'static_graph': True,
            }
        },
        callbacks=[
            LRMonitor(),
            OptimizerMonitor(),
            ValidationTableCallback(validation_df=testing_df),
            # EarlyStopper s opraveným názvem metriky
            EarlyStopper(
                monitor='pearson_corr',  # Kompletní název metriky
                dataloader_label='eval',
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_delta,
            )
        ],
        loggers=loggers,
        save_folder='checkpoints',
        save_filename='best_model_epoch_{epoch}.pt',
        save_overwrite=True,
        save_latest_filename="latest_checkpoint.pt",
        device_train_microbatch_size='auto',
        device="gpu" if torch.cuda.is_available() else 'cpu',
    )

    print("Starting training with Composer...")
    trainer.fit()
    print("Training finished.")
