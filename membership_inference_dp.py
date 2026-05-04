import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os

SAVE_DIR = "outputs"
TRAIN_SIZE = 5000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(f"{SAVE_DIR}/discriminator_dp.pt",
                                          map_location=device))
discriminator.eval()
print("DP Discriminator loaded.")

# same data split
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
member_loader = DataLoader(Subset(full_dataset, list(range(TRAIN_SIZE))),
                           batch_size=256, shuffle=False)
nonmember_loader = DataLoader(Subset(full_dataset, list(range(TRAIN_SIZE, TRAIN_SIZE * 2))),
                               batch_size=256, shuffle=False)

def get_scores(loader):
    scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            output = discriminator(imgs)
            scores.extend(output.cpu().numpy())
    return np.array(scores).flatten()

print("Running membership inference attack on DP model...")
member_scores = get_scores(member_loader)
nonmember_scores = get_scores(nonmember_loader)

print(f"\n--- DP Model Results ---")
print(f"Member avg confidence:     {member_scores.mean():.4f}")
print(f"Non-member avg confidence: {nonmember_scores.mean():.4f}")
print(f"Gap:                       {member_scores.mean() - nonmember_scores.mean():.4f}")

member_correct = (member_scores > 0.5).sum()
nonmember_correct = (nonmember_scores <= 0.5).sum()
total = len(member_scores) + len(nonmember_scores)
accuracy = (member_correct + nonmember_correct) / total * 100

print(f"\nAttack accuracy: {accuracy:.2f}%")
print(f"(Random guessing would give 50.00%)")
print(f"(No-DP baseline was 51.93%)")

# visualize score distributions
plt.figure(figsize=(10, 5))
plt.hist(member_scores, bins=50, alpha=0.6,
         label='Members (training data)', color='red')
plt.hist(nonmember_scores, bins=50, alpha=0.6,
         label='Non-members (unseen data)', color='blue')
plt.xlabel('Discriminator Confidence Score')
plt.ylabel('Count')
plt.title('Membership Inference Attack: DP Model Privacy Leakage')
plt.legend()
plt.savefig(f"{SAVE_DIR}/membership_inference_dp.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {SAVE_DIR}/membership_inference_dp.png")