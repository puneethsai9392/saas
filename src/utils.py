import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt

from config import VIS_LABEL_MAP as viz_map

plt.style.use('ggplot')

def set_class_values(all_classes, classes_to_train):
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value + 1
    label_mask = label_mask.astype(int)
    return label_mask

def denormalize(x, mean=None, std=None):
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    res = torch.clamp(t, 0, 1)
    res = res.squeeze(0).permute(1, 2, 0).numpy()
    return res

class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, out_dir, name='model'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

class SaveBestModelIOU:
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou

    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

def save_model(model, out_dir, name='model'):
    model.save_pretrained(os.path.join(out_dir, name))

def save_plots(train_loss, valid_loss, train_miou, valid_miou, out_dir):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='tab:blue', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='tab:red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    plt.figure(figsize=(10, 7))
    plt.plot(train_miou, color='tab:blue', linestyle='-', label='train mIoU')
    plt.plot(valid_miou, color='tab:red', linestyle='-', label='validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))
