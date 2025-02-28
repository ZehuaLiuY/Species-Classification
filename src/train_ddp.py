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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

world_size = int(os.getenv('SLURM_NTASKS'))
rank = int(os.getenv('SLURM_PROCID'))
local_rank = int(os.getenv('SLURM_LOCALID'))


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))

def train_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        writer,
        epoch,
        global_step_start=0,
        log_interval=10
):
    model.train()
    training_loss = torch.zeros(len(dataloader), dtype=torch.float32, device=device)
    running_samples = torch.tensor(0, dtype=torch.int32, device=device)
    correct = torch.tensor(0, dtype=torch.int32, device=device)
    total = torch.tensor(0, dtype=torch.int32, device=device)
    global_step = global_step_start

    for batch_idx, (crops, tokenized_texts, labels) in enumerate(dataloader):
        # --- to device ---
        batch_crops = torch.stack(crops, dim=0).to(device)
        batch_labels = torch.tensor(labels, dtype=torch.long, device=device)

        input_ids = torch.stack([t["input_ids"] for t in tokenized_texts]).to(device)
        attention_mask = torch.stack([t["attention_mask"] for t in tokenized_texts]).to(device)

        # --- forward pass ---
        optimizer.zero_grad()
        outputs = model(batch_crops, input_ids, attention_mask)
        loss = criterion(outputs, batch_labels)

        batch_loss = loss.item()

        loss.backward()
        optimizer.step()

        training_loss[batch_idx] = batch_loss
        running_samples += batch_labels.size(0)

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            batch_acc = (predicted == batch_labels).float().mean().item()
            writer.add_scalar('Train/Loss_batch', batch_loss, global_step)
            writer.add_scalar('Train/Accuracy_batch', batch_acc, epoch * global_step)

        global_step += 1

    torch.cuda.synchronize()
    dist.reduce(training_loss, 0, op=dist.ReduceOp.SUM)
    dist.reduce(running_samples, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total, 0, op=dist.ReduceOp.SUM)

    if rank ==0:
        epoch_loss = training_loss / running_samples if running_samples > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        return epoch_loss, epoch_acc, global_step

    return None, None, global_step

def validate(
        model,
        dataloader,
        criterion,
        device,
        writer,
        epoch,
):
    torch.cuda.synchronize()
    model.eval()
    val_running_loss = 0.0
    val_running_samples = 0
    val_correct = 0
    total = 0

    validation_loss = torch.zeros(len(dataloader), dtype=torch.float32, device=device)
    val_running_total = torch.tensor(0, dtype=torch.int32, device=device)
    val_correct = torch.tensor(0, dtype=torch.int32, device=device)
    val_total = torch.tensor(0, dtype=torch.int32, device=device)


    all_val_preds = []
    all_val_labels = []
    all_val_probs = []

    local_preds = []
    local_labels = []

    with torch.no_grad():
        for batch_idx, (crops, tokenized_texts, labels) in enumerate(dataloader):
            batch_crops = torch.stack(crops, dim=0).to(device)
            batch_labels = torch.tensor(labels, dtype=torch.long, device=device)
            input_ids = torch.stack([t["input_ids"] for t in tokenized_texts]).to(device)
            attention_mask = torch.stack([t["attention_mask"] for t in tokenized_texts]).to(device)

            outputs = model(batch_crops, input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)

            validation_loss[batch_idx] = loss.item()
            val_running_total += batch_labels.size(0)

            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

            local_preds.append(predicted.cpu().tolist())
            local_labels.append(batch_labels.cpu().tolist())

    torch.cuda.synchronize()
    dist.reduce(validation_loss, 0, op=dist.ReduceOp.SUM)
    dist.reduce(val_running_total, 0, op=dist.ReduceOp.SUM)
    dist.reduce(val_correct, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total, 0, op=dist.ReduceOp.SUM)

    local_preds = np.concatenate(local_preds, axis=0) if len(local_labels) > 0 else np.array([])
    local_labels = np.concatenate(local_labels, axis=0) if len(local_labels) > 0 else np.array([])

    gather_list_preds = [None for _ in range(world_size)]
    gather_list_labels = [None for _ in range(world_size)]

    dist.all_gather_object(gather_list_preds, local_preds)
    dist.all_gather_object(gather_list_labels, local_labels)

    if rank == 0:
        val_running_loss = validation_loss.sum().item()
        val_loss = val_running_loss / max(val_running_total, 1)
        val_acc =val_correct / max(total, 1)

        global_preds = np.concatenate([arr for arr in gather_list_preds if arr is not None])
        global_labels = np.concatenate([arr for arr in gather_list_labels if arr is not None])

        val_precision = precision_score(global_labels, global_preds,  average='weighted' , zero_division=0)
        val_recall = recall_score(global_labels, global_preds,  average='weighted' , zero_division=0)
        val_f1 = f1_score(global_labels, global_preds,  average='weighted' , zero_division=0)


        metrics = {
            'loss': val_loss,
            'acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
        }

        if writer is not None:
            writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
            writer.add_scalar('Val/Precision_epoch', val_precision, epoch)
            writer.add_scalar('Val/Recall_epoch', val_recall, epoch)
            writer.add_scalar('Val/F1_epoch', val_f1, epoch)

        return metrics
    return None

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    # DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler = train_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        shuffle=(train_sampler is None),
        collate_fn=multimodal_collate_fn
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler = val_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        shuffle=(val_sampler is None),
        collate_fn=multimodal_collate_fn
    )
    print("Dataloaders initialised.")

    # --- model ---
    print("Initializing model...")
    model = MultimodalLongTailClassifier(num_classes=args.num_classes)
    model = model.to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    print("Model initialised.")

    # --- optimizer & loss functions ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- training ---
    if rank == 0:
        writer = SummaryWriter()
    else:
        writer = None

    num_epochs = args.epochs
    global_step = 0
    patience = args.patience
    no_improvements = 0
    delta = args.delta
    best_recall = 0.0

    for epoch in range(1, num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        # val_sampler.set_epoch(epoch)

        torch.cuda.synchronize()
        train_loss, train_acc, global_step = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            writer,
            epoch,
            global_step,
            log_interval=10)

        if rank == 0:
            # log epoch-level metrics
            print(f"[Train] Epoch {epoch}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)


        # validation phase
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            writer,
            epoch,
        )

        # Sync all processes before going to the next epoch
        dist.barrier()

        if rank == 0:
            print(f"[Validation] Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | mAP: {val_metrics['mAP']:.4f}")

            # --- early stopping ---
            # save best model based on recall
            if val_metrics['recall'] > best_recall + delta:
                best_recall = val_metrics['recall']
                best_model_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.module.state_dict(), best_model_path)
                print(f"Best model saved with recall: {best_recall:.4f} at Epoch: {epoch}")
                no_improvements = 0
            else:
                no_improvements += 1
                print(f"No improvements for {no_improvements} consecutive epoch. Current best recall: {best_recall:.4f}")
                if no_improvements >= patience:
                    print(f"Early stopping at Epoch: {epoch}")
                    early_stop_flag = 1
                    break
                else:
                    early_stop_flag = 0
        else:
            early_stop_flag = 0

        early_stop_tensor = torch.tensor(early_stop_flag, dtype=torch.int, device=device)
        dist.broadcast(early_stop_tensor, src=0)
        dist.barrier()

        if early_stop_tensor.item() == 1:
            if rank == 0:
                print(f"Early stopping triggered after {epoch} epochs without improvement.")
            break
    # save final model
    if rank == 0:
        final_model_path = os.path.join(args.save_dir, "final_model.pth")
        torch.save(model.module.state_dict(), final_model_path)
        writer.close()
        print("Final model saved.")

    dist.destroy_process_group()


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

    print("DEBUG:", os.getenv('SLURM_NNODES'), os.getenv('SLURM_NTASKS'),
          os.getenv('SLURM_PROCID'), os.getenv('SLURM_LOCALID'))

    print(f"Found {world_size} GPUs in this machine.")
    if world_size < 1:
        raise RuntimeError("No GPU device found, DistributedDataParallel needs GPUs.")

    main(args)

