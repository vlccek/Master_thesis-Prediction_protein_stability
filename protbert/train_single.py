import argparse
import pandas as pd
from Model import Config, train_single_model
import torch

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
parser.add_argument("--smart_batch", type=bool, default=True,
                    help="Batch size")
parser.add_argument("--project_name", type=str, default="protein-mutation-prediction-protbert",
                    help="WandB project name")

parser.add_argument("--step_validation", type=int, default=1500)
args = parser.parse_args()

# --- Load dataset ---
df = pd.read_csv("dataset_255w_train.csv")
df_test = pd.read_csv("dataset_255w_test.csv")


if args.limit > 0:
    df = df[:args.limit]
    df_test = df_test[:int(max(2, args.limit*0.1))]

gpu_memory = 0

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_memory = props.total_memory / 1024 ** 3

    if gpu_memory < 41:
        args.batch_size = 25
    elif gpu_memory < 47:
        args.batch_size = 30
    elif gpu_memory < 86:
        args.batch_size = 84
    else:
        args.batch_size = 90
else:
    print("CUDA není dostupná")
    args.batch_size = 2






# --- Config setup ---
config = Config()
config.epochs = args.epochs
config.batch_size = args.batch_size
config.wandb_token = "c72619d4978c2953476cc5cf60d9ac0fac32b809"
config.learning_rate = args.lr
config.project_name = args.project_name
config.step_validation = config.step_validation

print(f"Starting training with name {args.project_name}... "
      f"with {args.limit} samples, lr={args.lr}, "
      f"epochs={args.epochs}, batch_size={args.batch_size}")

# --- Start training ---

model = train_single_model(df, df_test, config)
