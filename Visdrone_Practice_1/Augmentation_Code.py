import os
import cv2
import albumentations as A

# Paths
sequences_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\train\images"
annotations_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\train\annotations"

aug_images_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset_aug\train\images"
aug_labels_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset_aug\train\labels"

os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Loop through all sequence folders
for seq_name in os.listdir(sequences_dir):
    seq_folder = os.path.join(sequences_dir, seq_name)
    ann_file = os.path.join(annotations_dir, seq_name + ".txt")

    if not os.path.exists(ann_file):
        print(f"Annotation file not found for sequence {seq_name}, skipping.")
        continue

    # Read entire annotation file for the sequence
    with open(ann_file, "r") as f:
        lines = f.readlines()

    # Create a mapping: frame_id -> list of human bounding boxes
    frame_bboxes = {}
    for line in lines:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        class_id = int(parts[7])
        if class_id == 0:  # human
            x, y, w, h = map(float, parts[2:6])
            # Convert to YOLO format (normalized)
            # Later we can normalize with image shape
            if frame_id not in frame_bboxes:
                frame_bboxes[frame_id] = []
            frame_bboxes[frame_id].append([x, y, w, h, class_id])

    # Process each image in the sequence
    for img_file in os.listdir(seq_folder):
        img_path = os.path.join(seq_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w, _ = image.shape

        # Determine frame_id from image name (VisDrone format)
        frame_id = int(os.path.splitext(img_file)[0].split('_')[-1])
        if frame_id not in frame_bboxes:
            continue  # No humans in this frame

        # Prepare YOLO-format bounding boxes normalized
        bboxes = []
        class_labels = []
        for bbox in frame_bboxes[frame_id]:
            x_pixel, y_pixel, bw_pixel, bh_pixel, cls = bbox
            x_center = (x_pixel + bw_pixel / 2) / w
            y_center = (y_pixel + bh_pixel / 2) / h
            bw = bw_pixel / w
            bh = bh_pixel / h
            bboxes.append([x_center, y_center, bw, bh])
            class_labels.append(cls)

        # Apply augmentation
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']

        # Save augmented image
        out_img_path = os.path.join(aug_images_dir, seq_name + "_" + img_file)
        cv2.imwrite(out_img_path, aug_image)

        # Save augmented labels
        out_label_path = os.path.join(aug_labels_dir, seq_name + "_" + os.path.splitext(img_file)[0] + ".txt")
        with open(out_label_path, "w") as f:
            for bbox, cls in zip(aug_bboxes, aug_class_labels):
                f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

print("Data augmentation complete for VisDrone-style dataset!")
