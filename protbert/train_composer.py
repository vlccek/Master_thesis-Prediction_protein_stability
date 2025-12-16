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
parser.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=46,
                    help="Batch size")
parser.add_argument("--smart_batch", type=bool, default=False,
                    help="Batch size")
parser.add_argument("--project_name", type=str, default="protein-mutation-prediction-protbert",
                    help="WandB project name")
parser.add_argument("--datasets_prefix", type=str, default="dataset_255w_")
parser.add_argument("--base_dir", type=str, default="./", )
parser.add_argument("--step_validation", type=int, default=1500)
parser.add_argument("--model_name", type=str, default="Rostlab/prot_bert")


args = parser.parse_args()

print(f"the datasets_base_dir {args.base_dir} and datasets_prefix {args.datasets_prefix}")

TRAIN_PATH = f"{args.base_dir}/{args.datasets_prefix}train.csv"
TEST_PATH = f"{args.base_dir}/{args.datasets_prefix}test.csv"

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
config.epochs = args.epochs
config.batch_size = args.batch_size
config.wandb_token = "c72619d4978c2953476cc5cf60d9ac0fac32b809"
config.learning_rate = args.lr
config.project_name = args.project_name
config.step_validation = args.step_validation
config.batch_size = args.batch_size
config.eval_interval = args.step_validation
config.save_folder = "protbert_composer_checkpoints"
config.base_dir = args.base_dir
config.model_name = args.model_name


print(f"Training with config just started :happy:")

train_model(df, df_test, config)
