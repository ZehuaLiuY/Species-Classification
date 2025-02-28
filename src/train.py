import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer
from model import MultimodalLongTailClassifier
from mm_dataset import MultimodalNACTIDataset, multimodal_collate_fn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch, log_interval=10):
    model.train()
    running_loss = 0.0
    running_samples = 0
    correct = 0
    total = 0

    for batch_idx, (crops, tokenized_texts, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Train]")):
        # --- to device ---
        batch_crops = torch.stack(crops, dim=0).to(device)
        batch_labels = torch.tensor(labels, dtype=torch.long, device=device)
        input_ids = torch.stack([t["input_ids"] for t in tokenized_texts]).to(device)
        attention_mask = torch.stack([t["attention_mask"] for t in tokenized_texts]).to(device)

        # --- forward pass ---
        optimizer.zero_grad()
        outputs = model(batch_crops, input_ids, attention_mask)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_labels.size(0)
        running_samples += batch_labels.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            batch_loss = loss.item()
            batch_acc = (predicted == batch_labels).float().mean().item()
            writer.add_scalar('Train/Loss_batch', batch_loss, epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, epoch * len(dataloader) + batch_idx)

    epoch_loss = running_loss / running_samples if running_samples > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    val_running_loss = 0.0
    val_running_samples = 0
    val_correct = 0
    total = 0

    all_val_preds = []
    all_val_labels = []
    all_val_probs = []

    with torch.no_grad():
        for batch_idx, (crops, tokenized_texts, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Val]")):
            batch_crops = torch.stack(crops, dim=0).to(device)
            batch_labels = torch.tensor(labels, dtype=torch.long, device=device)
            input_ids = torch.stack([t["input_ids"] for t in tokenized_texts]).to(device)
            attention_mask = torch.stack([t["attention_mask"] for t in tokenized_texts]).to(device)

            outputs = model(batch_crops, input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)
            val_running_loss += loss.item() * batch_labels.size(0)
            val_running_samples += batch_labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

            all_val_preds.extend(predicted.cpu().tolist())
            all_val_labels.extend(batch_labels.cpu().tolist())
            all_val_probs.extend(outputs.softmax(dim=1).cpu().tolist())

    val_loss = val_running_loss / val_running_samples if val_running_samples > 0 else 0
    val_acc = val_correct / total if total > 0 else 0

    val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

    num_classes = outputs.size(1) if 'outputs' in locals() else 1
    all_val_labels_one_hot = torch.nn.functional.one_hot(torch.tensor(all_val_labels), num_classes=num_classes).cpu().numpy()
    all_val_probs_np = np.array(all_val_probs)
    appeared_class = np.where(all_val_labels_one_hot.sum(axis=0) > 0)[0]
    mAP = 0.0
    if len(appeared_class) > 0:
        filtered_labels = all_val_labels_one_hot[:, appeared_class]
        filtered_probs = all_val_probs_np[:, appeared_class]
        try:
            mAP = average_precision_score(filtered_labels, filtered_probs, average="weighted")
        except ValueError:
            print("Error calculating mAP.")
    else:
        print("No appeared class in validation.")

    writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
    writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
    writer.add_scalar('Val/Precision_epoch', val_precision, epoch)
    writer.add_scalar('Val/Recall_epoch', val_recall, epoch)
    writer.add_scalar('Val/F1_epoch', val_f1, epoch)
    writer.add_scalar('Val/mAP_epoch', mAP, epoch)

    metrics = {
        'loss': val_loss,
        'acc': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'mAP': mAP,
    }
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 定义 crop_transform
    crop_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # --- dataset ---
    print("Initializing dataloaders...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalNACTIDataset(
        image_dir=r"F:\DATASET\NACTI\images",
        json_path=r"E:\result\json\detection\detection_filtered.json",
        csv_path=r"F:/DATASET/NACTI/meta/nacti_metadata_balanced.csv",
        text_csv_path=r"F:/DATASET/NACTI/meta/description.csv",
        transforms=None,
        allow_empty=False,
        tokenizer=tokenizer,
        max_length=128,
        crop_transform=crop_transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    remaining = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, remaining])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=multimodal_collate_fn)
    print("Dataloaders initialised.")

    # --- model ---
    print("Initializing model...")
    model = MultimodalLongTailClassifier(num_classes=args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    print("Model initialised.")

    # --- training ---
    writer = SummaryWriter()
    num_epochs = args.epochs
    patience = args.patience
    no_improvements = 0
    delta = args.delta
    best_recall = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        print(f"[Train] Epoch {epoch}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)

        val_metrics = validate(model, val_loader, criterion, device, writer, epoch)
        print(f"[Validation] Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | mAP: {val_metrics['mAP']:.4f}")

        # save best model based on recall
        if val_metrics['recall'] > best_recall + delta:
            best_recall = val_metrics['recall']
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with recall: {best_recall:.4f} at Epoch: {epoch}")
            no_improvements = 0
        else:
            no_improvements += 1
            if no_improvements >= patience:
                print(f"Early stopping at Epoch: {epoch}")
                break

        # save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Final model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Long Tail Classifier with Pre-cropped Regions")
    parser.add_argument("--num_classes", type=int, default=48, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--delta", type=float, default=0.005, help="Minimum improvement delta to reset patience")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
