import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import copy
import os

LATENT_DIM = 64
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.0002
TRAIN_SIZE = 5000
SAVE_DIR = "outputs"
EPSILON = 10.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
member_dataset = Subset(full_dataset, list(range(TRAIN_SIZE)))
train_loader = DataLoader(member_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img.view(-1, 784))

generator = Generator().to(device)
discriminator = Discriminator().to(device)


shadow_discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

privacy_engine = PrivacyEngine()
discriminator, opt_d, train_loader = privacy_engine.make_private_with_epsilon(
    module=discriminator,
    optimizer=opt_d,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
print(f"DP enabled - Target epsilon: {EPSILON}")

print("\nStarting DP-GAN training...")
for epoch in range(EPOCHS):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # discriminator step
        opt_d.zero_grad()
        with torch.no_grad():
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_imgs_d = generator(z)
        d_loss = (criterion(discriminator(real_imgs), real_labels) +
                  criterion(discriminator(fake_imgs_d), fake_labels))
        d_loss.backward()
        opt_d.step()

        
        shadow_discriminator.load_state_dict(
            discriminator._module.state_dict()
        )

        # generator step
        opt_g.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs_g = generator(z)
        g_loss = criterion(shadow_discriminator(fake_imgs_g), real_labels)
        g_loss.backward()
        opt_g.step()

    if (epoch + 1) % 10 == 0:
        eps = privacy_engine.get_epsilon(DELTA)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"D_loss: {d_loss.item():.4f}  "
              f"G_loss: {g_loss.item():.4f}  "
              f"ε spent: {eps:.2f}")

torch.save(discriminator._module.state_dict(), f"{SAVE_DIR}/discriminator_dp.pt")
torch.save(generator.state_dict(), f"{SAVE_DIR}/generator_dp.pt")
print(f"\nDP models saved. Final ε = {privacy_engine.get_epsilon(DELTA):.2f}")