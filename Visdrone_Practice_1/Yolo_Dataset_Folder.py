import os
import shutil

base_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\YOLO_dataset"

# Create folder structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Copy train images & labels (augmented)
train_img_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset_aug\train\images"
train_lbl_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset_aug\train\labels"

for f in os.listdir(train_img_src):
    shutil.copy(os.path.join(train_img_src, f), os.path.join(base_dir, 'train', 'images', f))

for f in os.listdir(train_lbl_src):
    shutil.copy(os.path.join(train_lbl_src, f), os.path.join(base_dir, 'train', 'labels', f))

# Copy validation images & labels
val_img_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\val\images"
val_lbl_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\val\labels"

for f in os.listdir(val_img_src):
    shutil.copy(os.path.join(val_img_src, f), os.path.join(base_dir, 'val', 'images', f))

for f in os.listdir(val_lbl_src):
    shutil.copy(os.path.join(val_lbl_src, f), os.path.join(base_dir, 'val', 'labels', f))

# Copy test images & labels
test_img_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\test\images"
test_lbl_src = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset\test\labels"

for f in os.listdir(test_img_src):
    shutil.copy(os.path.join(test_img_src, f), os.path.join(base_dir, 'test', 'images', f))

for f in os.listdir(test_lbl_src):
    shutil.copy(os.path.join(test_lbl_src, f), os.path.join(base_dir, 'test', 'labels', f))

print("YOLOv8 dataset ready!")
