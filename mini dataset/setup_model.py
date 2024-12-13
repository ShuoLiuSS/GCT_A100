# /content/drive/MyDrive/GCT/setup_model.py
import torch
import pickle
import os
from config import Config
from Models import Transformer

# Set working directory
base_dir = '/content/drive/MyDrive/GCT'
os.chdir(base_dir)

# Load vocabulary to get vocab size
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Create config and add vocab sizes
config = Config()
config.src_vocab_size = len(vocab)
config.trg_vocab_size = len(vocab)

# Save configuration
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)

# Load prepared data
prepared_data = torch.load('prepared_tensors.pt')

# Initialize the transformer model
model = Transformer(
    opt=config,
    src_vocab=config.src_vocab_size,
    trg_vocab=config.trg_vocab_size
)

print("Model architecture:")
print(model)

# Load data
padded_sequences = prepared_data['padded_sequences']
conditions = prepared_data['conditions']
src_mask = prepared_data['src_mask']
trg_mask = prepared_data['trg_mask']

# Setup optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr,
    betas=(config.lr_beta1, config.lr_beta2)
)

# Save initial model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'initial_model.pt')

print("\nModel initialized and saved")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
