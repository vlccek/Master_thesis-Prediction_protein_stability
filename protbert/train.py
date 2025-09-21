from DualoModel import DualProtBERTPredictor, DualConfig, train_dual_model

import polars as pl

df = pl.read_csv('dataset_512w_protbert.csv')

# Add this to your config
config = DualConfig()
config.epochs = 20


print("Starting training")
# Start training
model = train_dual_model(df, config)
