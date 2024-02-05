import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import os
from PIL import Image

def contrastive_head(features, in_features):
    fc = nn.Linear(in_features, 1)
    return fc(features)

def contrastive_loss(outputs, target, margin=1.0):
    # Calculate contrastive loss
    loss = nn.MarginRankingLoss(margin=margin)(outputs, target, torch.ones_like(target))
    return loss

def load_custom_dataset(root_dir, transform=None):
    images = []
    targets = []
    classes = os.listdir(root_dir)

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            images_folder = os.path.join(cls_path, 'images')
            if os.path.isdir(images_folder):
                cls_images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.png')]
                for img_path in cls_images:
                    with open(img_path, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                    if transform:
                        img = transform(img)
                    images.append(img)
                    targets.append(classes.index(cls))

    return images, targets

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

images, targets = load_custom_dataset('C:\\Users\\hp\\Downloads\\chestxr\\COVID-19_Radiography_Dataset', transform=transform)

dataset = list(zip(images, targets))
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

backbone_model = resnet18(pretrained=True)
backbone_model.fc = nn.Identity()

optimizer = optim.Adam(backbone_model.parameters(), lr=0.001)

num_epochs = 1
for epoch in range(num_epochs):
    backbone_model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        features = backbone_model(inputs)
        outputs = contrastive_head(features, in_features=512)
        target = labels.unsqueeze(1).float()
        loss = contrastive_loss(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
