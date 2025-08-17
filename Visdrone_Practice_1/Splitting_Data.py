import os
import shutil
import random

# Paths
human_images_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\images"
human_labels_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\annotations"

output_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\dataset"

# Create folder structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Get all images
all_images = os.listdir(human_images_dir)
random.shuffle(all_images)

# Split indices
total = len(all_images)
train_end = int(0.7 * total)
val_end = int(0.9 * total)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]


# Function to copy images and labels
def copy_files(image_list, split):
    for img_file in image_list:
        # Copy image
        src_img = os.path.join(human_images_dir, img_file)
        dst_img = os.path.join(output_dir, split, 'images', img_file)
        shutil.copy(src_img, dst_img)

        # Copy corresponding label
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(human_labels_dir, label_file)
        dst_label = os.path.join(output_dir, split, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)


# Copy files
copy_files(train_images, 'train')
copy_files(val_images, 'val')
copy_files(test_images, 'test')

print("Dataset split complete! Train/val/test folders are ready for YOLOv8.")
