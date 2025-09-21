import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
    Sequence Understanding: ProtBERT captures complex patterns in protein sequences

    Comparative Architecture: Processes both wild-type and mutant sequences

    Uncertainty Awareness: Loss function accounts for measurement variability

    Transfer Learning: Leverages pre-trained protein knowledge

    Flexibility: Handles both substitutions and STOP mutations


Enchacements:

    Gradual Unfreezing: Start with frozen ProtBERT, gradually unfreeze layers

    Learning Rate Scheduling: Use cosine annealing or reduce-on-plateau

    Data Augmentation: Create synthetic mutations for underrepresented residues

    Ensemble Methods: Train multiple models with different seeds

    Attention Visualization: Analyze which residues ProtBERT focuses on
"""


# Configuration
class DualConfig:
    pretrained_model = "Rostlab/prot_bert"
    max_length = 512
    batch_size = 8  # Reduced due to larger model
    learning_rate = 2e-5
    wt_learning_rate = 2e-5  # Separate LR for WT encoder
    mut_learning_rate = 2e-5  # Separate LR for mutant encoder
    hidden_dropout_prob = 0.1
    epochs = 15
    freeze_layers = 6  # Freeze first 6 layers of each encoder


# Custom Dataset
class ProteinMutationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx)
        wt_seq = row['aa_seq']

        # Create mutant sequence
        position = row['position'] - 1  # Convert to 0-based indexing
        mut_aa = row['mut_aa']

        # Handle STOP codon (convert to '*')
        if mut_aa == 'STOP' or row['STOP']:
            mut_seq = wt_seq[:position] + '*' + wt_seq[position + 1:]
        else:
            mut_seq = wt_seq[:position] + mut_aa + wt_seq[position + 1:]

        # Tokenize both sequences
        wt_encoding = self.tokenizer(
            ' '.join(list(wt_seq)),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        mut_encoding = self.tokenizer(
            ' '.join(list(mut_seq)),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'wt_input_ids': wt_encoding['input_ids'].flatten(),
            'wt_attention_mask': wt_encoding['attention_mask'].flatten(),
            'mut_input_ids': mut_encoding['input_ids'].flatten(),
            'mut_attention_mask': mut_encoding['attention_mask'].flatten(),
            'fitness': torch.tensor(row['normalized_fitness'], dtype=torch.float),
            'fitness_sigma': torch.tensor(row['normalized_fitness_sigma'], dtype=torch.float)
        }


class DualProtBERTPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Two separate ProtBERT encoders with different weights
        self.wt_bert = BertModel.from_pretrained(config.pretrained_model)
        self.mut_bert = BertModel.from_pretrained(config.pretrained_model)

        # Freeze initial layers if needed
        if config.freeze_layers > 0:
            for i in range(config.freeze_layers):
                for param in self.wt_bert.encoder.layer[i].parameters():
                    param.requires_grad = False
                for param in self.mut_bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Attention mechanism to weight important positions
        self.position_attention = nn.MultiheadAttention(
            embed_dim=self.wt_bert.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Regression head with separate pathways
        self.wt_regressor = nn.Sequential(
            nn.Linear(self.wt_bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.mut_regressor = nn.Sequential(
            nn.Linear(self.mut_bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # Combined regression
        self.combined_regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Weight for combining the two representations
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight

    def forward(self, wt_input_ids, wt_attention_mask, mut_input_ids, mut_attention_mask, mutation_positions=None):
        # Process wild-type sequence
        wt_outputs = self.wt_bert(
            input_ids=wt_input_ids,
            attention_mask=wt_attention_mask
        )
        wt_embeddings = wt_outputs.last_hidden_state

        # Process mutant sequence
        mut_outputs = self.mut_bert(
            input_ids=mut_input_ids,
            attention_mask=mut_attention_mask
        )
        mut_embeddings = mut_outputs.last_hidden_state

        # Apply position-aware attention if mutation positions provided
        if mutation_positions is not None:
            # Create position masks
            position_mask = torch.zeros_like(wt_attention_mask)
            for i, pos in enumerate(mutation_positions):
                if pos < wt_attention_mask.size(1):  # Ensure within bounds
                    position_mask[i, pos] = 1

            # Apply attention to focus on mutation region
            wt_attended, _ = self.position_attention(
                wt_embeddings, wt_embeddings, wt_embeddings,
                key_padding_mask=~position_mask.bool()
            )
            mut_attended, _ = self.position_attention(
                mut_embeddings, mut_embeddings, mut_embeddings,
                key_padding_mask=~position_mask.bool()
            )

            # Use [CLS] token and attended mutation position
            wt_representation = torch.cat([
                wt_embeddings[:, 0, :],  # [CLS] token
                wt_attended.mean(dim=1)  # Average of attended positions
            ], dim=1)

            mut_representation = torch.cat([
                mut_embeddings[:, 0, :],  # [CLS] token
                mut_attended.mean(dim=1)  # Average of attended positions
            ], dim=1)
        else:
            # Use only [CLS] token representations
            wt_representation = wt_embeddings[:, 0, :]
            mut_representation = mut_embeddings[:, 0, :]

        # Process through separate regressors
        wt_features = self.wt_regressor(wt_representation)
        mut_features = self.mut_regressor(mut_representation)

        # Combine with learnable weight
        combined = torch.cat([
            self.alpha * wt_features,
            (1 - self.alpha) * mut_features
        ], dim=1)

        # Final prediction
        return self.combined_regressor(combined).squeeze()


def uncertainty_aware_mse(preds, targets, sigma):
    loss = ((preds - targets) ** 2) / (2 * sigma ** 2 + 1e-6)
    loss += torch.log(sigma ** 2 + 1e-6)
    return loss.mean()


from tqdm import tqdm
import time


def train_dual_model(df, config):
    print("🚀 Starting training process...")
    print(f"📊 Configuration: {config.__dict__}")

    # Initialize tokenizer and model
    print("🔄 Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model, do_lower_case=False)
    model = DualProtBERTPredictor(config)
    print("✅ Tokenizer and model loaded successfully")

    # Prepare data
    print("📝 Preparing data...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📈 Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_dataset = ProteinMutationDataset(train_df, tokenizer, config.max_length)
    val_dataset = ProteinMutationDataset(val_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    print(f"📦 DataLoaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Using device: {device}")
    if torch.cuda.is_available():
        print(
            f"🎮 GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    model.to(device)

    # Separate optimizers for each encoder
    wt_params = list(model.wt_bert.parameters())
    mut_params = list(model.mut_bert.parameters())
    other_params = list(model.position_attention.parameters()) + \
                   list(model.wt_regressor.parameters()) + \
                   list(model.mut_regressor.parameters()) + \
                   list(model.combined_regressor.parameters()) + \
                   [model.alpha]

    optimizer = AdamW([
        {'params': wt_params, 'lr': config.wt_learning_rate},
        {'params': mut_params, 'lr': config.mut_learning_rate},
        {'params': other_params, 'lr': config.learning_rate}
    ])

    print(f"⚙️ Optimizer configured with learning rates:")
    print(f"   - WT encoder: {config.wt_learning_rate}")
    print(f"   - MUT encoder: {config.mut_learning_rate}")
    print(f"   - Other params: {config.learning_rate}")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    training_history = []

    print(f"\n🎯 Starting training for {config.epochs} epochs...")
    print("=" * 80)

    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        print(f"\n📘 Epoch {epoch + 1}/{config.epochs}")
        print("-" * 50)

        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}",
                          unit="batch", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Extract mutation positions for attention
            mutation_positions = batch.get('position', None)

            # Forward pass
            predictions = model(
                batch['wt_input_ids'],
                batch['wt_attention_mask'],
                batch['mut_input_ids'],
                batch['mut_attention_mask'],
                mutation_positions
            )

            # Calculate loss with uncertainty weighting
            loss = uncertainty_aware_mse(
                predictions,
                batch['fitness'],
                batch['fitness_sigma']
            )

            # Add regularization to encourage different representations
            reg_loss = 0.01 * (1 - torch.cosine_similarity(
                model.wt_regressor[0].weight.flatten(),
                model.mut_regressor[0].weight.flatten(),
                dim=0
            ))
            total_loss = loss + reg_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        train_pbar.close()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0

        print("🔍 Running validation...")
        val_pbar = tqdm(val_loader, desc="Validating", unit="batch", leave=False)

        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                mutation_positions = batch.get('position', None)

                predictions = model(
                    batch['wt_input_ids'],
                    batch['wt_attention_mask'],
                    batch['mut_input_ids'],
                    batch['mut_attention_mask'],
                    mutation_positions
                )

                loss = uncertainty_aware_mse(
                    predictions,
                    batch['fitness'],
                    batch['fitness_sigma']
                )

                total_val_loss += loss.item()
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        val_pbar.close()
        avg_val_loss = total_val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_dual_model.pth')
            save_status = "💾 BEST MODEL SAVED"
        else:
            save_status = ""

        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   ⏱️  Time: {epoch_time:.1f}s")
        print(f"   📉 Train Loss: {avg_train_loss:.4f}")
        print(f"   📈 Val Loss: {avg_val_loss:.4f}")
        print(f"   🎯 Alpha (WT weight): {model.alpha.item():.3f}")
        print(f"   📐 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   {save_status}")
        print("-" * 50)

        # Store history for potential plotting
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'alpha': model.alpha.item(),
            'time': epoch_time
        })

    print("=" * 80)
    print("🎉 Training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Best model saved as: 'best_dual_model.pth'")

    return model, training_history