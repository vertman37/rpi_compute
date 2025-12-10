import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0-1] -> [-1-1]
])

train_set = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_set = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

#================================save images
imgs = []
labels = []

for batch_x, batch_y in test_loader:   # 이미 Normalize 처리된 tensor
    # batch_x: (N,1,28,28), float32, [-1,1]
    imgs.append(batch_x.numpy())       # ndarray로 변환
    labels.append(batch_y.numpy())

import numpy as np
imgs = np.concatenate(imgs, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez("fmnist_test_normalized.npz", images=imgs, labels=labels)

data = np.load("fmnist_test_normalized.npz")
imgs = data["images"]        # (N,1,28,28)
labels = data["labels"]
print(labels)
print(labels.shape)
#native np. we don't need it..
# from torchvision import datasets
# raw_set = datasets.FashionMNIST(root='./data', train=False, download=True)

# raw_imgs = []
# for img, label in raw_set:
#     arr = np.array(img, dtype=np.float32) / 255.0
#     arr = (arr - 0.5) / 0.5
#     raw_imgs.append(arr)

# raw_imgs = np.array(raw_imgs).reshape(-1, 1, 28, 28)
# np.save("fmnist_test_normalized.npy", raw_imgs)


exit()



# define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)#28x28->14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#14x14->7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # dropout for MLP
        x = self.fc2(x)
        return x


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

#============================save
model = SimpleCNN()
model.load_state_dict(torch.load('fashion_mnist_cnn.pth'))
model.eval()

state = model.state_dict()
for name, param in state.items():
    print(name, param.shape)

# conv1.weight torch.Size([32, 1, 3, 3])
# conv1.bias torch.Size([32])
# conv2.weight torch.Size([64, 32, 3, 3])
# conv2.bias torch.Size([64])
# fc1.weight torch.Size([128, 3136])
# fc1.bias torch.Size([128])
# fc2.weight torch.Size([10, 128])
# fc2.bias torch.Size([10])

# print(model.conv1.weight.shape)
# print(model.conv1.bias.shape)
# torch.Size([32, 1, 3, 3])
# torch.Size([32])

w = model.conv1.weight.detach().cpu().numpy()
b = model.conv1.bias.detach().cpu().numpy()
print(type(w), w.shape)
print(type(b), b.shape)
# <class 'numpy.ndarray'> (32, 1, 3, 3)
# <class 'numpy.ndarray'> (32,)


params_np = {name: p.detach().cpu().numpy()
             for name, p in model.state_dict().items()}

for k, v in params_np.items():
    print(k, type(v), v.shape)
    # print(v.dtype)


# conv1.weight <class 'numpy.ndarray'> (32, 1, 3, 3)
# conv1.bias <class 'numpy.ndarray'> (32,)
# conv2.weight <class 'numpy.ndarray'> (64, 32, 3, 3)
# conv2.bias <class 'numpy.ndarray'> (64,)
# fc1.weight <class 'numpy.ndarray'> (128, 3136)
# fc1.bias <class 'numpy.ndarray'> (128,)
# fc2.weight <class 'numpy.ndarray'> (10, 128)
# fc2.bias <class 'numpy.ndarray'> (10,)

# np.savez("params.npz",
#          conv1_w=w,
#          conv1_b=b)

# data = np.load("params.npz")
# w = data["conv1_w"]
# b = data["conv1_b"]

#conv1.weight -> conv1_weight
clean_params = {k.replace('.', '_'): v for k, v in params_np.items()}

import numpy as np
np.savez("params.npz", **clean_params)

#finally done to export numpy!
print('saved')

#copy this code to load parameters!
data = np.load("params.npz")
for k in data.files:
    arr = data[k]
    print(k, arr.shape, arr.dtype)

exit()





#train setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

#train loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # avg loss for epoch
    avg_train_loss = train_loss / len(train_loader)
    
    # loss for test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# final test
final_loss, final_acc = evaluate(model, test_loader, criterion, device)
print(f"\nfinal test accuracy: {final_acc:.2f}%")

#save model
torch.save(model.state_dict(), 'fashion_mnist_cnn.pth')
print("the model is saved: fashion_mnist_cnn.pth")