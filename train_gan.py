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

# data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

# split into member (training) and non-member (holdout) sets
member_indices = list(range(TRAIN_SIZE))
nonmember_indices = list(range(TRAIN_SIZE, TRAIN_SIZE * 2))

member_dataset = Subset(full_dataset, member_indices)
nonmember_dataset = Subset(full_dataset, nonmember_indices)

train_loader = DataLoader(member_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Member samples: {len(member_dataset)}")
print(f"Non-member samples: {len(nonmember_dataset)}")

# generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),  # 28x28 = 784
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)