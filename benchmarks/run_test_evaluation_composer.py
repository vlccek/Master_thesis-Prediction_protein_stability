# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2026-05-06

import argparse
import polars as pl
import torch
import os
import numpy as np
import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, f1_score, matthews_corrcoef, accuracy_score, r2_score

from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Composer imports
from composer import Trainer, Callback, State, Logger
from composer.utils import dist
from composer.models import ComposerModel

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
    ProteinMutationDatasetESM as DatasetESM,
    ComposerProteinModelESM
)

from protbert.ModelComposerESM_meanpooling import (
    ComposerProteinModelESM as ComposerProteinModelESMMeanPool
)

class InferenceCallback(Callback):
    def __init__(self, output_dir, original_df, model_type):
        self.output_dir = output_dir
        self.model_type = model_type
        # Add row index to ensure we can join back correctly
        try:
            self.original_df = original_df.with_row_index(name="row_idx")
        except AttributeError:
            self.original_df = original_df.with_row_count(name="row_idx")
            
        self.preds = []
        self.indices = []

    def eval_batch_end(self, state: State, logger: Logger):
        # state.outputs is [Batch, 1]
        outputs = state.outputs.detach().float().cpu().numpy().flatten().tolist()
        # batch['row_idx'] was added in the Dataset classes if we modify them, 
        # but let's assume we use a similar logic to InteractiveReportCallback
        batch_indices = state.batch['row_idx'].detach().cpu().numpy().flatten().tolist()
        
        self.preds.extend(outputs)
        self.indices.extend(batch_indices)

    def eval_end(self, state: State, logger: Logger):
        # Gather all predictions from all GPUs
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
                })
                
                # Deduplicate just in case (DistributedSampler might have padding)
                preds_df = preds_df.unique(subset=["row_idx"])
                
                preds_df = preds_df.with_columns(pl.col("row_idx").cast(pl.UInt32))
                self.original_df = self.original_df.with_columns(pl.col("row_idx").cast(pl.UInt32))
                
                final_df = self.original_df.join(preds_df, on="row_idx", how="inner")
                
                output_file = os.path.join(self.output_dir, "test_predictions.csv")
                final_df.write_csv(output_file)
                print(f"Predictions saved to: {output_file}")
                
                # Calculate metrics
                self.calculate_metrics(final_df)

    def calculate_metrics(self, df: pl.DataFrame):
        print("\n--- Calculating Statistics ---")
        y_true = df["fitness"].to_numpy()
        y_pred = df["predicted_fitness"].to_numpy()

        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        pos_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)
        binary_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

        def to_ternary(values):
            bins = []
            for v in values:
                if v < -0.15: bins.append(0)
                elif v > 0.15: bins.append(2)
                else: bins.append(1)
            return np.array(bins)

        y_true_ter = to_ternary(y_true)
        y_pred_ter = to_ternary(y_pred)
        ternary_mcc = matthews_corrcoef(y_true_ter, y_pred_ter)
        ternary_acc = accuracy_score(y_true_ter, y_pred_ter)

        stats_output = (
            f"--- EVALUATION RESULTS ---\n"
            f"Model Type:           {self.model_type}\n"
            f"Pearson Correlation:  {pearson_corr:.4f}\n"
            f"Spearman Correlation: {spearman_corr:.4f}\n"
            f"MSE:                  {mse:.4f}\n"
            f"R2 Score:             {r2:.4f}\n"
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
        with open(os.path.join(self.output_dir, "test_metrics.txt"), "w") as f:
            f.write(stats_output)

def run_evaluation_composer(test_file, checkpoint_path, output_dir, batch_size, model_type):
    if dist.get_global_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Configuration based on model type
    if model_type.lower() == "esm":
        config = ConfigESM()
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        prep_fn = prepare_data_esm
        dataset_cls = DatasetESM
        model_cls = ComposerProteinModelESM
    elif model_type.lower() == "esm_meanpool":
        config = ConfigESM()
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        prep_fn = prepare_data_esm
        dataset_cls = DatasetESM
        model_cls = ComposerProteinModelESMMeanPool
    else:
        config = ConfigProtbert()
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
        prep_fn = prepare_data_protbert
        dataset_cls = DatasetProtbert
        model_cls = ComposerProteinModel

    # Load and prep data
    df_raw = pl.read_csv(test_file)
    if "target" in df_raw.columns and "fitness" not in df_raw.columns:
        df_raw = df_raw.rename({"target": "fitness"})
    
    processed_df = prep_fn(df_raw, max_total_length=config.max_length, window_size=config.seq_window_size)
    processed_df = processed_df.drop_nulls(["clean_wt", "clean_mut"])

    dataset = dataset_cls(processed_df, tokenizer, max_length=config.max_length)
    
    # Composer distributed sampler
    sampler = dist.get_sampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, pin_memory=True)

    # Initialize model
    composer_model = model_cls(config.pretrained_model, tokenizer)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    # Map location to CPU first to avoid OOM, Trainer will move it to GPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint.get('state', {}).get('model', checkpoint.get('state_dict', checkpoint))
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    # Load into the underlying torch model
    composer_model.model.load_state_dict(state_dict)

    # Inference Callback
    inf_callback = InferenceCallback(output_dir, processed_df, model_type)

    # Trainer for inference
    trainer = Trainer(
        model=composer_model,
        eval_dataloader=dataloader,
        device="gpu",
        precision="amp_bf16" if torch.cuda.is_bf16_supported() else "amp_fp16",
        callbacks=[inf_callback]
    )

    print(f"Starting inference on {dist.get_world_size()} GPUs...")
    trainer.eval()
    print("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="protbert")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    
    run_evaluation_composer(
        args.test_file, 
        args.checkpoint, 
        args.output_dir, 
        args.batch_size, 
        args.model_type
    )
