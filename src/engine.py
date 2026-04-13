import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image

def draw_translucent_seg_maps(image, seg_map, label_colors, alpha=0.6):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    overlay = np.zeros_like(image, dtype=np.uint8)
    for class_index, color in enumerate(label_colors):
        mask = seg_map == class_index
        overlay[mask] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def save_segmentation_visual(orig_image, pred_mask, label_colors, epoch, save_dir='vis_outputs'):
    os.makedirs(save_dir, exist_ok=True)
    vis_img = draw_translucent_seg_maps(orig_image, pred_mask, label_colors)
    save_path = os.path.join(save_dir, f"val_epoch_{epoch}.png")
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved visual to {save_path}")

def save_mask(mask, save_path):
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(save_path)
    print(f"[INFO] Saved mask to {save_path}")

def train(model, train_dataloader, device, optimizer, classes_to_train, processor, metric, label_colors_list, epoch):
    print('Training')
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
    counter = 0
    num_classes = len(classes_to_train)
    batch_los = []
    for i, data in enumerate(prog_bar):
        counter += 1
        pixel_values = data['pixel_values'].to(device)
        pixel_mask = data['pixel_mask'].to(device)
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=[m.to(device) for m in data['mask_labels']],
            class_labels=[c.to(device) for c in data['class_labels']],
            pixel_mask=pixel_mask
        )
        loss = outputs.loss
        batch_los.append(loss)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
        pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        metric.add_batch(references=data['orig_mask'], predictions=pred_maps)
    train_loss = train_running_loss / counter
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)['mean_iou']
    return train_loss, iou, batch_los

def validate(model, valid_dataloader, device, classes_to_train, label_colors_list, epoch, save_dir, processor, metric):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    num_classes = len(classes_to_train)
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
        for i, data in enumerate(prog_bar):
            counter += 1
            pixel_values = data['pixel_values'].to(device)
            pixel_mask = data['pixel_mask'].to(device)
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=[m.to(device) for m in data['mask_labels']],
                class_labels=[c.to(device) for c in data['class_labels']],
                pixel_mask=pixel_mask
            )
            target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
            pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            loss = outputs.loss
            valid_running_loss += loss.item()
            metric.add_batch(references=data['orig_mask'], predictions=pred_maps)
    valid_loss = valid_running_loss / counter
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)['mean_iou']
    return valid_loss, iou
