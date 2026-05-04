import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# config
LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
TRAIN_SIZE = 5000
SAVE_DIR = "outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

member_indices = list(range(TRAIN_SIZE))
nonmember_indices = list(range(TRAIN_SIZE, TRAIN_SIZE * 2))

member_dataset = Subset(full_dataset, member_indices)
nonmember_dataset = Subset(full_dataset, nonmember_indices)

train_loader = DataLoader(member_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Member samples: {len(member_dataset)}")
print(f"Non-member samples: {len(nonmember_dataset)}")

# generator (takes random noise, outputs fake image)
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
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

# discriminator(takes image, outputs confidence it's real)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        flat = img.view(-1, 784)
        return self.model(flat)

# initialize networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# loss and optimizers
criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# training loop
print("\nStarting GAN training...")
for epoch in range(EPOCHS):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        
        # labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # train discriminator
        opt_d.zero_grad()
        real_loss = criterion(discriminator(real_imgs), real_labels)
        
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = generator(z)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_d.step()
        
        # train generator
        opt_g.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        opt_g.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# save models
torch.save(discriminator.state_dict(), f"{SAVE_DIR}/discriminator.pt")
torch.save(generator.state_dict(), f"{SAVE_DIR}/generator.pt")
print("\nModels saved.")