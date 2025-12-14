#git clone https://github.com/akamaster/pytorch_resnet_cifar10.git
from pytorch_resnet_cifar10.resnet import resnet20

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# from resnet import resnet20   # akamaster 코드

# -----------------------
# Config
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 100
batch_size = 128
epochs = 150
lr = 0.05
weight_decay = 5e-4
momentum = 0.9

# -----------------------
# Dataset
# -----------------------
mean = (0.5071, 0.4867, 0.4408)
std  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = torchvision.datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_set = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0
)

# -----------------------
# Model
# -----------------------
model = resnet20(num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[75, 120],
    gamma=0.1
)

# -----------------------
# Train / Eval functions
# -----------------------
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return 100. * correct / total


# -----------------------
# Training loop
# -----------------------
best_acc = 0.0

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    test_acc = evaluate(model, test_loader)

    scheduler.step()
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "resnet20_cifar100_best.pth")

    print(
        f"[{epoch+1:03d}/{epochs}] "
        f"loss={train_loss:.4f} "
        f"train_acc={train_acc:.2f}% "
        f"test_acc={test_acc:.2f}% "
        f"lr={scheduler.get_last_lr()[0]:.4f}"
    )

print(f"Best Test Accuracy: {best_acc:.2f}%")
