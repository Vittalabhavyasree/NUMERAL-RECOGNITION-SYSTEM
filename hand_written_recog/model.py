import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

# --- Capsule Network Utils ---
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, dim_capsules, kernel_size, stride):
        super().__init__()
        self.capsules = nn.Conv2d(in_channels, out_channels * dim_capsules, kernel_size, stride)
        self.out_channels = out_channels
        self.dim_capsules = dim_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x)
        u = u.view(batch_size, self.out_channels, self.dim_capsules, -1)
        u = u.permute(0, 3, 1, 2).contiguous()
        u = u.view(batch_size, -1, self.dim_capsules)
        return squash(u)

class DigitCapsules(nn.Module):
    def __init__(self, num_caps_in, dim_caps_in, num_caps_out, dim_caps_out):
        super().__init__()
        self.num_caps_out = num_caps_out
        self.dim_caps_out = dim_caps_out
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps_in, num_caps_out, dim_caps_out, dim_caps_in))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x).squeeze(4)
        b_ij = torch.zeros(1, u_hat.size(1), u_hat.size(2), 1, device=x.device)
        for _ in range(3):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, dim=-1)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)
        return v_j.squeeze(1)

class CapsuleLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, outputs, labels):
        left = F.relu(self.m_pos - outputs) ** 2
        right = F.relu(outputs - self.m_neg) ** 2
        labels_one_hot = F.one_hot(labels, num_classes=outputs.size(1)).float()
        loss = labels_one_hot * left + self.lambda_ * (1.0 - labels_one_hot) * right
        return loss.sum(dim=1).mean()

# --- Capsule Model ---
class MnistCapsuleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=9, stride=1),
            nn.ReLU(inplace=True)
        )
        self.primary_caps = PrimaryCapsules(256, 32, 8, kernel_size=9, stride=2)
        self.digit_caps = DigitCapsules(1152, 8, num_classes, 16)

    def forward(self, x):
        x = self.features(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        probs = torch.norm(x, dim=-1)
        return probs

# --- Albumentations Dataset ---
class AlbumentationsMNIST(Dataset):
    def __init__(self, train=True):
        self.data = datasets.MNIST('./data', train=train, download=True)
        self.train = train
        self.transform = A.Compose([
    A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=30.0, p=0.3),
    A.Rotate(limit=25, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.3),

    A.Resize(28, 28),  # ✅ Add this line

    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2()
])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(image=img)["image"][0].unsqueeze(0)
        return img, label

# --- Train/Test ---
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return running_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(AlbumentationsMNIST(train=True), batch_size=64, shuffle=True)
    test_loader = DataLoader(AlbumentationsMNIST(train=False), batch_size=1000, shuffle=False)

    model = MnistCapsuleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = CapsuleLoss()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(1, 6):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

    torch.save(model.state_dict(), "mnist_capsule.pt")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
