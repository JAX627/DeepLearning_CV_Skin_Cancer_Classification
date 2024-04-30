import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


#init file path
train_dir = '/4486/Topic_5_Data/ISIC84by84/Train/'
test_dir = '/4486/Topic_5_Data/ISIC84by84/Test/'

model_dir = 'model_weights_v2'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

results_dir = 'results_v2'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    

#init dataset
data_tsf  = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

img_ds = {
    'train': datasets.ImageFolder(train_dir, data_tsf['train']),
    'val': datasets.ImageFolder(test_dir, data_tsf['val'])
}

classes = img_ds['train'].classes
class_counts = [0] * len(classes)

for img_path, _ in img_ds['train'].imgs:
    class_label = os.path.basename(os.path.dirname(img_path))
    class_index = classes.index(class_label)
    class_counts[class_index] += 1

class_weights = [1.0 / count for count in class_counts]
class_weights = torch.FloatTensor(class_weights)

dl = {x: DataLoader(img_ds[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
d_size = {x: len(img_ds[x]) for x in ['train', 'val']}
c_name = img_ds['train'].classes

#init model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 8)
device = torch.device("cuda:0")
model = model.to(device)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 20

#init result
train_losses = []
train_accs = []
val_losses = []
val_accs = []

#training
for epoch in range(num_epochs):
    for dset_name in ['train', 'val']:
        #train
        if dset_name == 'train':
            model.train()
            progress_bar = tqdm(dl[dset_name], unit="batch")
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    t, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / d_size[dset_name]
            epoch_acc = running_corrects.double() / d_size[dset_name]

            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
        #val
        else:
            model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dl[dset_name]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    t, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / d_size[dset_name]
            epoch_acc = running_corrects.double() / d_size[dset_name]

            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc)

    torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))

#test

model.load_state_dict(torch.load(os.path.join(model_dir, 'model_epoch_20.pth')))

y_true = []
y_pred = []

for inputs, labels in dl['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        t, preds = torch.max(outputs, 1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
print(f'Test Accuracy: {acc:.4f}')
print(f'Test F1-Score: {f1:.4f}')
