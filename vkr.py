import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры обучения
num_epochs = 20
batch_size = 32
lr = 1e-5

# Общие преобразования для всех датасетов
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def train_and_save(data_dir, out_name):
    # 1) Датасеты и загрузчики
    train_ds = ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
    val_ds   = ImageFolder(os.path.join(data_dir, 'val'),   transform=data_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 2) Модель ResNet18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    # 3) Критерий и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) Обучение
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[{out_name}] Epoch {epoch+1}/{num_epochs}, '
              f'Loss: {running_loss/len(train_loader):.4f}')

    # 5) Валидация
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f'[{out_name}] Val Accuracy: {100*correct/total:.2f}%')

    # 6) Сохранение
    torch.save(model.state_dict(), f'{out_name}.pth')
    print(f'Model saved as {out_name}.pth\n')

# Переобучение биологической модели
#train_and_save(data_dir='dataset', out_name='resnet18_biology')
#train_and_save(data_dir='dataset2', out_name='resnet18_people')
train_and_save(data_dir='dataset1', out_name='resnet18_cars')
