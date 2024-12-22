import os
import json
import torch
from PIL import Image
from torchvision import datasets, transforms
from PytorchWildlife.models import classification as pw_classification
from tqdm import tqdm
from torch.utils.data import random_split
from src.dataset import NACTIAnnotationDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Transform for classification
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classification_model = pw_classification.AI4GAmazonRainforest(device=device)

# number of classes
num_classes = 36 # the default number of classes in AI4GAmazonRainforest model

# split the dataset into training and validation part 80% and 20%
dataset = NACTIAnnotationDataset(
    image_dir=r"F:\DATASET\NACTI\images\nacti_part0",
    json_path=r"E:\result\json\detection\part0output.json",
    csv_path=r"F:\DATASET\NACTI\meta\nacti_metadata_part0.csv",
    transforms=transform
)

def collate_fn_remove_none(batch):
    # batch [(image, target), (image, target), ...]
    # if image or target is None, remove it
    filtered_batch  = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return None, None
    imgs, tgts = zip(*filtered_batch )
    return list(imgs), list(tgts)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          collate_fn=collate_fn_remove_none)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn_remove_none)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                         collate_fn=collate_fn_remove_none)

# fine-tune the model
# set the model to training mode and define the optimizer and loss function
classification_model.to(device)
optimizer = optim.AdamW(classification_model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# training loop
num_epochs = 10
writer = SummaryWriter()
global_step = 0

for epoch in range(num_epochs):
    #################
    # training phase
    #################
    classification_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_acc = 0
    batch_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        if batch[0] is None:
            # Skip batch if it is empty
            continue
        global_step += 1
        images, targets = batch

        images = torch.stack(images).to(device)
        # get the labels from the target
        # the target is a dictionary with keys: boxes, labels, common_name
        labels = [t["labels"][0] for t in targets]
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = classification_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # get the predicted class
        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        batch_loss = loss.item()
        batch_acc = (predicted == labels).float().mean().item()
        # # add the loss and accuracy to tensorboard for every 5 batches
        if batch % 5 == 0:
            writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)
        # print(f"Batch Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}")


    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    print(f"[Train] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    #################
    # validation phase
    #################
    classification_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            if batch[0] is None:
                continue
            images, targets = batch

            images = list(images)
            targets = list(targets)

            images = torch.stack(images).to(device)
            labels = [t["labels"][0] for t in targets]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = classification_model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_loss / val_total if val_total > 0 else 0.0
    val_epoch_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"[Val]   Epoch {epoch+1}/{num_epochs}, Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

# testing phase
classification_model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        if batch[0] is None:
            continue

        images, targets = batch
        images = list(images)
        targets = list(targets)

        images = torch.stack(images).to(device)
        labels = [t["labels"][0] for t in targets]
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        outputs = classification_model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_epoch_loss = test_loss / test_total if test_total > 0 else 0.0
test_epoch_acc = test_correct / test_total if test_total > 0 else 0.0
print(f"[Test]  Loss: {test_epoch_loss:.4f}, Acc: {test_epoch_acc:.4f}")
