import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import os
from PIL import Image

# Define model architecture with single perceptron for contrastive head
class ContrastiveHead(nn.Module):
    def __init__(self, in_features):
        super(ContrastiveHead, self).__init__()
        self.fc = nn.Linear(in_features, 1)  # Single perceptron layer for contrastive head

    def forward(self, x):
        return self.fc(x)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        self.targets = []

        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if os.path.isdir(cls_path):
                images_folder = os.path.join(cls_path, 'images')
                if os.path.isdir(images_folder):
                    cls_images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.png')]
                    for img_path in cls_images:
                        self.images.append(img_path)
                        self.targets.append(self.classes.index(cls))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, target = self.images[index], self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, target

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomDataset(root_dir='C:\\Users\\hp\\Downloads\\chestxr\\COVID-19_Radiography_Dataset', transform=transform)
print("Total number of samples in dataset:", len(dataset))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Number of samples in train dataset:", len(train_dataset))
print("Number of samples in test dataset:", len(test_dataset))

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define backbone model (e.g., ResNet18)
backbone_model = resnet18(pretrained=True)  # Load pre-trained weights
backbone_model.fc = nn.Identity()  # Replace the fully connected layer with an identity layer

# Define contrastive head for supervised contrastive learning
contrastive_head = ContrastiveHead(in_features=512)  # Assuming the output of ResNet18's backbone is 512 features

# Define optimizer
optimizer = optim.Adam(contrastive_head.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    contrastive_head.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        features = backbone_model(inputs)
        outputs = contrastive_head(features)
        target = labels.unsqueeze(1).float()  # Reshape labels for supervised contrastive learning
        loss = nn.BCEWithLogitsLoss()(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
