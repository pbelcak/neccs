import torch
torch.set_default_dtype(torch.float64)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")