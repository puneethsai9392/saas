import albumentations as A
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def load_yolo_instance_segmentation(label_path, img_w=2592, img_h=1944):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    polygons = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        class_id = int(values[0])
        coords = values[1:]
        polygon = [(int(x * img_w), int(y * img_h)) for x, y in zip(coords[0::2], coords[1::2])]
        polygons.append((class_id, polygon))
    return polygons

def save_yolo_instance_labels(label_path, polygons, img_w=2592, img_h=1944):
    lines = []
    for class_id, polygon in polygons:
        coords = [str(x / img_w) + " " + str(y / img_h) for x, y in polygon]
        line = f"{class_id} " + " ".join(coords) + "\n"
        lines.append(line)
    with open(label_path, 'w') as f:
        f.writelines(lines)

def augment_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augments=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.5),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    for img_name in os.listdir(image_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        if not os.path.exists(label_path):
            continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        polygons = load_yolo_instance_segmentation(label_path, img_w, img_h)
        for i in range(num_augments):
            aug_polygons = []
            all_keypoints = [point for _, polygon in polygons for point in polygon]
            aug_result = augmentations(image=image, keypoints=all_keypoints)
            aug_img = aug_result["image"]
            aug_keypoints = aug_result["keypoints"]
            idx = 0
            for class_id, polygon in polygons:
                new_polygon = aug_keypoints[idx:idx + len(polygon)]
                aug_polygons.append((class_id, new_polygon))
                idx += len(polygon)
            aug_img_name = f"aug{i+1}_{img_name}"
            aug_img_path = os.path.join(output_image_dir, aug_img_name)
            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            aug_label_name = f"aug{i+1}_{img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')}"
            aug_label_path = os.path.join(output_label_dir, aug_label_name)
            save_yolo_instance_labels(aug_label_path, aug_polygons, img_w, img_h)
