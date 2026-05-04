import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# config
LATENT_DIM = 64        # size of random noise input to generator
BATCH_SIZE = 64        # samples per training batch
EPOCHS = 50            # training rounds
LR = 0.0002            # learning rate
TRAIN_SIZE = 5000      # use subset so training is fast
SAVE_DIR = "outputs"   #  save results

os.makedirs(SAVE_DIR, exist_ok=True)

# device
device = torch.device("mps" if torch.backends.mps.is_available() 
                       else "cpu")
print(f"Using device: {device}")