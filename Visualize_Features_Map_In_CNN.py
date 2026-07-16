import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_data=datasets.CIFAR10(root="./data",transform=transform,train=False,download=True)
test_loader=DataLoader(test_data,batch_size=1000)

class FeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*8*8,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        return self.fc1(x)

model=FeatureCNN()
model.eval()
image,label=next(iter(test_loader))

with torch.no_grad():
    features_maps=F.relu(model.conv1(image))
fig,axes=plt.subplots(1,6,figsize=(15,3))
for i in range(6):
    axes[i].imshow(features_maps[0,i].cpu(),cmap='viridis')
    axes[i].axis("off")
plt.suptitle("feature maps from conv1")
plt.show()