import argparse
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
from PytorchWildlife.models import classification as pw_classification
from datetime import timedelta
from collections import Counter
from dataset import NACTIAnnotationDataset

world_size = int(os.getenv('SLURM_NTASKS'))
rank = int(os.getenv('SLURM_PROCID'))
local_rank = int(os.getenv('SLURM_LOCALID'))

def pil_collect_fn(batch):
    """
    Custom collate function that returns a list of PIL.Image and target_dict.

    Args:
        batch (list): A list of (PIL.Image, target_dict) tuples.

    Returns:
        (list, list): List of PIL.Image and list of target_dict.
    """
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))

def train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        epoch,
        global_step_start=0,
        transform=None,
        writer=None,
        log_interval=10
):

    model.train()
    training_loss = torch.zeros(len(loader), dtype=torch.float32, device=device)
    running_samples = torch.tensor(0, dtype=torch.int32, device=device)
    correct = torch.tensor(0, dtype=torch.int32, device=device)
    total = torch.tensor(0, dtype=torch.int32, device=device)

    global_step = global_step_start

    for batch_idx, (images, targets) in enumerate(loader):
        all_crops = []
        all_labels = []

        for i in range(len(images)):
            pil_img = images[i]
            target_dict = targets[i]

            boxes = target_dict["boxes"]    # shape: [N, 4] => [x, y, w, h]
            labels = target_dict["labels"]  # shape: [N]

            # if boxes.size(0) == 0:
            #     continue

            for j in range(boxes.size(0)):
                x1, y1, w, h = boxes[j]
                x2 = x1 + w
                y2 = y1 + h

                x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                # if x1_ < 0 or y1_ < 0 or x2_ <= x1_ or y2_ <= y1_:
                #     continue

                cropped_pil = pil_img.crop((x1_, y1_, x2_, y2_))
                if transform:
                    cropped_tensor = transform(cropped_pil).to(device)
                else:
                    cropped_tensor = transforms.ToTensor()(cropped_pil).to(device)

                all_crops.append(cropped_tensor)
                all_labels.append(labels[j].item())

        # if len(all_crops) == 0:
        #     continue

        batch_crops = torch.stack(all_crops, dim=0)
        batch_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(batch_crops)
        loss = criterion(outputs, batch_labels)

        batch_loss = loss.item()
        batch_num = batch_labels.size(0)
        loss.backward()
        optimizer.step()

        # running_loss += batch_loss
        training_loss[batch_idx] = batch_loss
        running_samples += batch_num

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        # log per batch on rank 0
        if writer is not None and (batch_idx + 1) % log_interval == 0:
            batch_acc = (predicted == batch_labels).float().mean().item()
            writer.add_scalar('Train/Batch_Loss', batch_loss, global_step)
            writer.add_scalar('Train/Batch_Accuracy', batch_acc, global_step)

        global_step += 1

    torch.cuda.synchronize()
    dist.reduce(training_loss, 0, op=dist.ReduceOp.SUM)
    dist.reduce(running_samples, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        running_loss = training_loss.sum().item()
        epoch_loss = running_loss / max(running_samples, 1)
        epoch_acc = correct / max(total, 1)

        return epoch_loss, epoch_acc, global_step

    return None, None, global_step


def validate(model, loader, criterion, device, epoch, transform=None, writer=None):
    torch.cuda.synchronize()
    model.eval()
    validation_loss = torch.zeros(len(loader), dtype=torch.float32, device=device)
    val_running_total = torch.tensor(0, dtype=torch.int32, device=device)
    val_correct = torch.tensor(0, dtype=torch.int32, device=device)
    val_total = torch.tensor(0, dtype=torch.int32, device=device)

    local_preds = []
    local_labels = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            # if images is None or targets is None:
            #     continue

            all_crops = []
            all_labels = []

            for i in range(len(images)):
                # if images[i] is None or targets[i] is None:
                #     continue

                pil_img = images[i]
                target_dict = targets[i]
                boxes = target_dict["boxes"]
                labels = target_dict["labels"]

                # if boxes.size(0) == 0:
                #     continue

                for j in range(boxes.size(0)):
                    x1, y1, w, h = boxes[j]
                    x2 = x1 + w
                    y2 = y1 + h

                    x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])
                    # if x1_ < 0 or y1_ < 0 or x2_ <= x1_ or y2_ <= y1_:
                    #     continue

                    cropped_pil = pil_img.crop((x1_, y1_, x2_, y2_))
                    if transform:
                        cropped_tensor = transform(cropped_pil).to(device)
                    else:
                        cropped_tensor = transforms.ToTensor()(cropped_pil).to(device)

                    all_crops.append(cropped_tensor)
                    all_labels.append(labels[j].item())

            # if len(all_crops) == 0:
            #     continue

            batch_crops = torch.stack(all_crops, dim=0)
            batch_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

            outputs = model(batch_crops)
            loss = criterion(outputs, batch_labels)

            batch_loss = loss.item()
            batch_num = batch_labels.size(0)

            # val_running_loss += batch_loss
            validation_loss[batch_idx] = batch_loss
            val_running_total += batch_num

            _, predicted = torch.max(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            val_total += batch_labels.size(0)

            local_preds.append(predicted.cpu().numpy())
            local_labels.append(batch_labels.cpu().numpy())

    torch.cuda.synchronize()
    dist.reduce(validation_loss, 0, op=dist.ReduceOp.SUM)
    dist.reduce(val_running_total, 0, op=dist.ReduceOp.SUM)
    dist.reduce(val_correct, 0, op=dist.ReduceOp.SUM)
    dist.reduce(val_total, 0, op=dist.ReduceOp.SUM)

    local_preds = np.concatenate(local_preds, axis=0) if len(local_preds) > 0 else np.array([])
    local_labels = np.concatenate(local_labels, axis=0) if len(local_labels) > 0 else np.array([])

    gather_list_preds = [None for _ in range(world_size)]
    gather_list_labels = [None for _ in range(world_size)]

    dist.all_gather_object(gather_list_preds, local_preds)
    dist.all_gather_object(gather_list_labels, local_labels)

    # local_preds_t = torch.as_tensor(local_preds, dtype=torch.long, device=device)
    # local_labels_t = torch.as_tensor(local_labels, dtype=torch.long, device=device)
    # pred_len = torch.tensor([local_preds_t.numel()], dtype=torch.long, device=device)
    # label_len = torch.tensor([local_labels_t.numel()], dtype=torch.long, device=device)
    # pred_len_list = [torch.zeros_like(pred_len) for _ in range(world_size)]
    # label_len_list = [torch.zeros_like(label_len) for _ in range(world_size)]
    # dist.all_gather(pred_len_list, pred_len)
    # dist.all_gather(label_len_list, label_len)
    # max_pred_len = max(x.item() for x in pred_len_list)
    # max_label_len = max(x.item() for x in label_len_list)
    # pad_local_preds = torch.full((max_pred_len,), -1, dtype=torch.long, device=device)
    # pad_local_preds[: pred_len.item()] = local_preds_t

    # pad_local_labels = torch.full((max_label_len,), -1, dtype=torch.long, device=device)
    # pad_local_labels[: label_len.item()] = local_labels_t
    # gathered_preds_t = [torch.zeros_like(pad_local_preds) for _ in range(world_size)]
    # gathered_labels_t = [torch.zeros_like(pad_local_labels) for _ in range(world_size)]

    # dist.all_gather(gathered_preds_t, pad_local_preds)
    # dist.all_gather(gathered_labels_t, pad_local_labels)
    # # 5) 去掉 padding，把所有 rank 的真实数据拼起来
    # global_preds = []
    # global_labels = []
    # for r in range(world_size):
    #     real_pred_len = pred_len_list[r].item()
    #     real_label_len = label_len_list[r].item()
    #     # 这里假设 preds 和 labels 的长度一致，否则需要更健壮的处理
    #     global_preds.append(gathered_preds_t[r][:real_pred_len].cpu())
    #     global_labels.append(gathered_labels_t[r][:real_label_len].cpu())

    # global_preds = torch.cat(global_preds, dim=0).numpy()
    # global_labels = torch.cat(global_labels, dim=0).numpy()

    if rank == 0:
        val_running_loss = validation_loss.sum().item()
        val_loss = val_running_loss / max(val_running_total, 1)
        val_acc = val_correct / max(val_total, 1)

        global_preds = np.concatenate([arr for arr in gather_list_preds if arr is not None])
        global_labels = np.concatenate([arr for arr in gather_list_labels if arr is not None])

        val_precision = precision_score(global_labels, global_preds,  average='weighted' , zero_division=0)
        val_recall = recall_score(global_labels, global_preds,  average='weighted' , zero_division=0)
        val_f1 = f1_score(global_labels, global_preds,  average='weighted' , zero_division=0)

        # # per-class Accuracy and Recall
        # unique_labels = np.unique(global_labels)
        # # 1) Per-class Accuracy
        # per_class_accuracy = []
        # for label in unique_labels:
        #     total_class_samples = np.sum(global_labels == label)
        #     correct_class_samples = np.sum((global_labels == label) & (global_preds == label))
        #     if total_class_samples > 0:
        #         acc = correct_class_samples / total_class_samples
        #     else:
        #         acc = 0.0
        #     per_class_accuracy.append(acc)

        # # 2) Per-class Recall
        # per_class_recall = recall_score(
        #     global_labels,
        #     global_preds,
        #     labels=unique_labels,
        #     average=None,
        #     zero_division=0
        # )

        metrics = {
            'loss': val_loss,
            'acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
        }

        if writer is not None:
            # Log epoch-level stats
            writer.add_scalar('Val/Epoch_Loss', val_loss, epoch)
            writer.add_scalar('Val/Epoch_Accuracy', val_acc, epoch)
            writer.add_scalar('Val/Epoch_Precision', val_precision, epoch)
            writer.add_scalar('Val/Epoch_Recall', val_recall, epoch)
            writer.add_scalar('Val/Epoch_F1', val_f1, epoch)

            # # per-class metrics
            # for i, cls_label in enumerate(unique_labels):
            #     writer.add_scalar(f'Val/PerClassAccuracy/class_{cls_label}', per_class_accuracy[i], epoch)
            #     writer.add_scalar(f'Val/PerClassRecall/class_{cls_label}', per_class_recall[i], epoch)

        return metrics
    return None


def main_worker(args):
    print(rank)

    print(f"Rank {rank} about to init_process_group()...")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800)
    )
    print(f"Rank {rank} after init_process_group()!")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- dataset ---
    print("initialising dataset")

    dataset = NACTIAnnotationDataset(
        image_dir=r"/user/work/bw19062/Individual_Project/dataset/image",
        json_path=r"/user/work/bw19062/Individual_Project/dataset/metadata/detection/detection_filtered.json",
        csv_path=r"/user/work/bw19062/Individual_Project/dataset/metadata/nacti_metadata_balanced.csv",
    )

    total_len = len(dataset)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size  = total_len - train_size - val_size
    g = torch.Generator()
    g.manual_seed(0)
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=g)

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
        shuffle=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        shuffle=(train_sampler is None),
        collate_fn=pil_collect_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=torch.cuda.is_available(),
        shuffle=(val_sampler is None),
        collate_fn=pil_collect_fn,
    )

    print("dataloader initialised")

    # ---- model ----
    model = pw_classification.AI4GAmazonRainforest(device=device)
    num_features = model.net.classifier.in_features
    model.net.classifier = torch.nn.Linear(num_features, 49)
    model.to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # print(model)
    # weighted cross entropy loss
    all_labels = []
    for i in range(len(train_dataset)):
        _, targets = train_dataset[i]
        all_labels.extend(targets["labels"].tolist())
    cls_counts = Counter(all_labels)
    weights = []
    for i in range(49):
        weights.append(1.0 / (cls_counts[i] + 1e-6))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * 49
    weight = torch.tensor(weights, device=device)


    criterion = torch.nn.CrossEntropyLoss(weight)

    # Create a SummaryWriter only on rank 0, so only the main process logs.
    if rank == 0:
        writer = SummaryWriter(log_dir="./runs_ddp/")
    else:
        writer = None

    best_recall = 0
    global_step = 0
    non_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        torch.cuda.synchronize()

        # training phase
        train_epoch_loss, train_epoch_acc, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            global_step_start=global_step,
            transform=transform,
            writer=writer,   # Pass the writer to the function
            log_interval=10  # Log every 10 iterations
        )

        if rank == 0:
            # Log epoch-level stats
            writer.add_scalar('Train/Epoch_Loss', train_epoch_loss, epoch)
            writer.add_scalar('Train/Epoch_Accuracy', train_epoch_acc, epoch)

            print(f"[Train] Epoch {epoch}/{args.epochs} | "
                  f"Loss: {train_epoch_loss:.4f} | Acc: {train_epoch_acc:.4f}")

        # validation phase
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            transform=transform,
            writer=writer
        )

        if rank == 0:
            print(f"[Validation] Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['acc']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f}"
                  )
            # --- early stopping ---
            current_recall = val_metrics['recall']

            if current_recall > best_recall + args.delta:
                best_recall = current_recall
                non_improvement = 0
                torch.save(model.module.state_dict(), "./model/ddp/best_model_ddp.pth")
                print(f"Best model saved with recall: {best_recall:.4f}, at epoch: {epoch}")

    if rank == 0:
        torch.save(model.state_dict(), "./model/ddp/final_model_ddp.pth")
        print("Final model saved.")
        writer.close()  # Close the writer

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help="Number of total epochs to run")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per process")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--delta', type=float, default=1e-4, help="Minimal change for early stopping")
    args = parser.parse_args()

    print("DEBUG:", os.getenv('SLURM_NNODES'), os.getenv('SLURM_NTASKS'),
          os.getenv('SLURM_PROCID'), os.getenv('SLURM_LOCALID'))


    print(f"Found {world_size} GPUs in this machine.")
    if world_size < 1:
        raise RuntimeError("No GPU device found, DistributedDataParallel needs GPUs.")

    # mp.spawn(
    #     main_worker,
    #     nprocs=world_size,
    #     args=(world_size, args),
    #     join=True
    # )
    main_worker(args)

if __name__ == "__main__":
    main()
