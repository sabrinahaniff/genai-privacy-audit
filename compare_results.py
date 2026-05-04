import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

SAVE_DIR = "outputs"
TRAIN_SIZE = 5000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DiscriminatorWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.model(img.view(-1, 784))

class DiscriminatorNoDropout(nn.Module):
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

# load both discriminators
disc_nodp = DiscriminatorWithDropout().to(device)
disc_nodp.load_state_dict(torch.load(f"{SAVE_DIR}/discriminator.pt",
                                      map_location=device))
disc_nodp.eval()

disc_dp = DiscriminatorNoDropout().to(device)
disc_dp.load_state_dict(torch.load(f"{SAVE_DIR}/discriminator_dp.pt",
                                    map_location=device))
disc_dp.eval()

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
full_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
member_loader = DataLoader(Subset(full_dataset, list(range(TRAIN_SIZE))),
                           batch_size=256, shuffle=False)
nonmember_loader = DataLoader(Subset(full_dataset,
                               list(range(TRAIN_SIZE, TRAIN_SIZE * 2))),
                               batch_size=256, shuffle=False)

def get_scores(discriminator, loader):
    scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            output = discriminator(imgs)
            scores.extend(output.cpu().numpy())
    return np.array(scores).flatten()

# get the scores for both models
print("Computing scores...")
nodp_member = get_scores(disc_nodp, member_loader)
nodp_nonmember = get_scores(disc_nodp, nonmember_loader)
dp_member = get_scores(disc_dp, member_loader)
dp_nonmember = get_scores(disc_dp, nonmember_loader)

# compute the metrics
nodp_gap = nodp_member.mean() - nodp_nonmember.mean()
dp_gap = dp_member.mean() - dp_nonmember.mean()

nodp_acc = ((nodp_member > 0.5).sum() + 
            (nodp_nonmember <= 0.5).sum()) / (len(nodp_member) + 
            len(nodp_nonmember)) * 100
dp_acc = ((dp_member > 0.5).sum() + 
          (dp_nonmember <= 0.5).sum()) / (len(dp_member) + 
          len(dp_nonmember)) * 100

# plot
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# plot 1: no dp histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(nodp_member, bins=50, alpha=0.6, color='red', 
         label='Members')
ax1.hist(nodp_nonmember, bins=50, alpha=0.6, color='blue',
         label='Non-members')
ax1.set_title('No DP — Privacy Leakage Visible', fontsize=13, fontweight='bold')
ax1.set_xlabel('Discriminator Confidence Score')
ax1.set_ylabel('Count')
ax1.legend()
ax1.text(0.05, 0.95, f'Gap: {nodp_gap:.4f}\nAttack Acc: {nodp_acc:.2f}%',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plot 2: dp histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(dp_member, bins=50, alpha=0.6, color='red',
         label='Members')
ax2.hist(dp_nonmember, bins=50, alpha=0.6, color='blue',
         label='Non-members')
ax2.set_title(f'With DP (ε=10) — Leakage Eliminated', fontsize=13, 
              fontweight='bold')
ax2.set_xlabel('Discriminator Confidence Score')
ax2.set_ylabel('Count')
ax2.legend()
ax2.text(0.05, 0.95, f'Gap: {dp_gap:.4f}\nAttack Acc: {dp_acc:.2f}%',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plot 3: gap comparison bar chart
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(['No DP', f'With DP\n(ε=10)'], 
               [nodp_gap, dp_gap],
               color=['#e74c3c', '#2ecc71'], width=0.4)
ax3.set_title('Privacy Leakage Gap Comparison', fontsize=13, fontweight='bold')
ax3.set_ylabel('Member vs Non-member Gap')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
for bar, val in zip(bars, [nodp_gap, dp_gap]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# plot 4: attack accuracy comparison
ax4 = fig.add_subplot(gs[1, 1])
bars2 = ax4.bar(['No DP', f'With DP\n(ε=10)', 'Random\nGuessing'],
                [nodp_acc, dp_acc, 50.0],
                color=['#e74c3c', '#2ecc71', '#95a5a6'], width=0.4)
ax4.set_title('Attack Accuracy Comparison', fontsize=13, fontweight='bold')
ax4.set_ylabel('Attack Accuracy (%)')
ax4.set_ylim(48, 54)
ax4.axhline(y=50, color='black', linestyle='--', linewidth=0.8,
            label='Random guessing baseline')
for bar, val in zip(bars2, [nodp_acc, dp_acc, 50.0]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

fig.suptitle('Privacy Audit: GAN vs DP-GAN\nMembership Inference Attack Results',
             fontsize=15, fontweight='bold', y=1.02)

plt.savefig(f"{SAVE_DIR}/comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Comparison figure saved to {SAVE_DIR}/comparison.png")
print(f"\nSummary:")
print(f"No-DP  → Gap: {nodp_gap:.4f}, Attack accuracy: {nodp_acc:.2f}%")
print(f"DP     → Gap: {dp_gap:.4f}, Attack accuracy: {dp_acc:.2f}%")