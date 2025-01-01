def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device,
                    writer,
                    epoch,
                    global_step_start=0,
                    log_interval=5):
    """
    Train the model for one epoch in a batch_size>1 setting.
    For each batch, we iterate through all images in the batch,
    and for each image, we iterate through all bounding boxes,
    do a forward pass on the cropped region, accumulate the loss
    across the entire batch, and then do a single backward + optimizer step.

    Args:
        model (nn.Module): The classification model.
        loader (DataLoader): DataLoader (batch_size>=1) for training.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): GPU or CPU device to use.
        writer (SummaryWriter): For TensorBoard logging (optional).
        epoch (int): Current epoch index (for logging).
        global_step_start (int): Global step counter at the beginning of this epoch.
        log_interval (int): Interval (in batches) for logging to TensorBoard.

    Returns:
        (float, float, int): (epoch_loss, epoch_acc, new_global_step)
            - epoch_loss: Average loss across the entire epoch.
            - epoch_acc: Accuracy across all bounding boxes in the epoch.
            - new_global_step: The updated global step after this epoch.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    global_step = global_step_start
    w_thres, h_thres = 8, 8

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if images is None or targets is None:
            continue

        optimizer.zero_grad()

        total_loss_this_batch = 0.0
        batch_correct = 0
        batch_boxes_count = 0

        for i in range(len(images)):
            if images[i] is None or targets[i] is None:
                continue

            img_tensor = images[i].to(device)       # shape: [C, H, W]
            target_dict = targets[i]
            boxes = target_dict["boxes"].to(device)   # shape: [N, 4]
            labels = target_dict["labels"].to(device) # shape: [N]

            if boxes.size(0) == 0:
                continue

            for j in range(boxes.size(0)):
                x1, y1, w, h = boxes[j]
                x2 = x1 + w
                y2 = y1 + h

                x1_, y1_, x2_, y2_ = map(int, [x1, y1, x2, y2])

                if x1_ < 0 or y1_ < 0 or x2_ > img_tensor.shape[2] or y2_ > img_tensor.shape[1]:
                    continue

                if (x2_ - x1_) < w_thres or (y2_ - y1_) < h_thres:
                    continue

                # crop + forward
                cropped_tensor = img_tensor[:, y1_:y2_, x1_:x2_].unsqueeze(0)
                single_label = labels[j].unsqueeze(0)

                outputs = model(cropped_tensor)
                loss = criterion(outputs, single_label)

                total_loss_this_batch += loss

                _, predicted = torch.max(outputs, 1)
                batch_correct += (predicted == single_label).sum().item()
                batch_boxes_count += 1

        if batch_boxes_count > 0:
            avg_loss = total_loss_this_batch / batch_boxes_count
            avg_loss.backward()
            optimizer.step()

            running_loss += avg_loss.item()
            correct += batch_correct
            total += batch_boxes_count

        if writer is not None and ((batch_idx + 1) % log_interval == 0):
            if batch_boxes_count > 0:
                batch_acc = batch_correct / batch_boxes_count
                writer.add_scalar('Train/Loss_batch', avg_loss.item(), global_step)
                writer.add_scalar('Train/Accuracy_batch', batch_acc, global_step)

        global_step += 1

    epoch_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc, global_step
