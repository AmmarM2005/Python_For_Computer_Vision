import os
import shutil

# ---------------- CONFIG ----------------
dataset_root = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\VisDrone2019-VID-val\VisDrone2019-VID-val"
sequences_dir = os.path.join(dataset_root, "sequences")
annotations_dir = os.path.join(dataset_root, "annotations")

yolo_root = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\YOLO_dataset"

train_ratio = 0.8  # 80% train, 20% val

# ---------------- CREATE YOLO FOLDERS ----------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(yolo_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, split, "labels"), exist_ok=True)

# ---------------- COLLECT DATA ----------------
all_images = []
all_labels = []

for seq_folder in sorted(os.listdir(sequences_dir)):
    seq_path = os.path.join(sequences_dir, seq_folder)
    if not os.path.isdir(seq_path):
        continue

    ann_file = os.path.join(annotations_dir, f"{seq_folder}.txt")
    if not os.path.isfile(ann_file):
        continue

    # Read annotation lines
    with open(ann_file, "r") as f:
        ann_lines = f.readlines()

    # Recursively get all images in sequence folder
    images = []
    for root, _, files in os.walk(seq_path):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                images.append(os.path.join(root, f))
    images.sort()

    # Process each image
    for idx, img_path in enumerate(images):
        if idx >= len(ann_lines):
            break
        line = ann_lines[idx].strip()
        if line == "":
            continue

        parts = line.split(",")
        if len(parts) < 8:
            continue

        x1, y1, w, h, _, category, *_ = map(float, parts[:8])
        if int(category) != 1:  # Keep only humans
            continue

        # YOLO normalized format
        img_w, img_h = 1920, 1080  # VisDrone default
        x_center = (x1 + w / 2) / img_w
        y_center = (y1 + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        label_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

        all_images.append(img_path)
        all_labels.append(label_line)

# ---------------- SPLIT DATA ----------------
split_idx = int(len(all_images) * train_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]
train_labels = all_labels[:split_idx]
val_labels = all_labels[split_idx:]

# ---------------- COPY IMAGES AND LABELS ----------------
for split_name, imgs, lbls in zip(["train", "val"], [train_images, val_images], [train_labels, val_labels]):
    for i, img_path in enumerate(imgs):
        img_name = os.path.basename(img_path)
        dst_img = os.path.join(yolo_root, split_name, "images", img_name)
        shutil.copy(img_path, dst_img)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        dst_label = os.path.join(yolo_root, split_name, "labels", label_name)
        with open(dst_label, "w") as f:
            f.write(lbls[i])

print("YOLO dataset preparation complete!")
