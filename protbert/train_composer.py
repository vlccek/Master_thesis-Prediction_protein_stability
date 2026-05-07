# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2025-09-21

import argparse
import polars as pl
from ModelComposter import Config
import os

from ModelComposter import train_full_model as train_model

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Train ProtBERT model on protein dataset")

parser.add_argument("--limit", type=int, default=0,
                    help="Number of rows to use from the dataset")
parser.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate")
parser.add_argument("--epochs", type=int, default=5,
                    help="Number of training epochs on full dataset")
parser.add_argument("--batch_size", type=int, default=46,
                    help="Batch size")
# Removed --smart_batch as it's not used in ModelComposerESM and was not explicitly used in ModelComposter
parser.add_argument("--project_name", type=str, default="protein-mutation-prediction-esm2", # Changed default
                    help="WandB project name")
parser.add_argument("--datasets_prefix", type=str, default="dataset_255w_")
parser.add_argument("--base_dir", type=str, default="./", )
parser.add_argument("--step_validation", type=int, default=1500)
parser.add_argument("--model_name", type=str, default="Rostlab/prot_bert")
parser.add_argument("--seq_window_size", type=int, default=255,
                    help="Size of the sequence window centered around mutation (default: 255)")
parser.add_argument("--freezed_layers", type=int, default=3, )
parser.add_argument("--num_workers", type=int, default=2,
                    help="Number of data loader workers (default: 2)")
parser.add_argument("--save_folder", type=str, default="protbert_composer_checkpoints", )
parser.add_argument("--stochastic_depth_drop_rate", type=float, default=0.2,
                    help="Stochastic depth drop rate")
args = parser.parse_args()

print(f"the datasets_base_dir {args.base_dir} and datasets_prefix {args.datasets_prefix}")

TRAIN_PATH = os.path.join(args.base_dir, f"{args.datasets_prefix}train.csv")
TEST_PATH = os.path.join(args.base_dir, f"{args.datasets_prefix}test.csv")

# test if files exist
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

# --- Load dataset ---
df = pl.read_csv(TRAIN_PATH)
df_test = pl.read_csv(TEST_PATH)

if args.limit > 0:
    print(f"Limiting dataset to {args.limit} rows")
    df = df[:args.limit]
    df_test = df_test[:int(max(2, args.limit * 0.1))]

# --- Config setup ---
config = Config()
config.batch_size = args.batch_size
config.wandb_token = "c72619d4978c2953476cc5cf60d9ac0fac32b809"
config.learning_rate = args.lr
config.project_name = args.project_name
config.step_validation = args.step_validation
config.batch_size = args.batch_size
config.eval_interval = args.step_validation
config.save_folder = args.save_folder
config.base_dir = args.base_dir
config.pretrained_model = args.model_name
config.seq_window_size = args.seq_window_size
config.freeze_layers = args.freezed_layers
config.stochastic_depth_drop_rate = args.stochastic_depth_drop_rate
config.epochs = args.epochs

print(f"Training with config just started :happy:")

train_model(df, df_test, config, num_workers=args.num_workers, epochs=args.epochs)
