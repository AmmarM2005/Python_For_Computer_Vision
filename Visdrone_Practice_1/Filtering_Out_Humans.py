import os
import shutil

# Paths (update according to your dataset)
annotations_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\VisDrone2019-VID-val\VisDrone2019-VID-val\annotations"
sequences_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\VisDrone2019-VID-val\VisDrone2019-VID-val\sequences"

# Output folders
human_images_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\images"
human_labels_dir = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\Humans\annotations"

# Create output folders
os.makedirs(human_images_dir, exist_ok=True)
os.makedirs(human_labels_dir, exist_ok=True)

# Loop through all annotation files
for ann_file in os.listdir(annotations_dir):
    ann_path = os.path.join(annotations_dir, ann_file)

    with open(ann_path, "r") as f:
        lines = f.readlines()

    # Keep only humans (class_id = 0)
    human_lines = [line for line in lines if ',0,' in line]  # class_id = 0

    if human_lines:
        # Determine corresponding sequence folder
        seq_name = ann_file.split(".")[0]  # annotation file name matches sequence folder
        seq_folder = os.path.join(sequences_dir, seq_name)

        if os.path.exists(seq_folder):
            for img_file in os.listdir(seq_folder):
                # Copy image to human_images_dir
                src_img_path = os.path.join(seq_folder, img_file)
                dst_img_path = os.path.join(human_images_dir, f"{seq_name}_{img_file}")
                shutil.copy(src_img_path, dst_img_path)

        # Save filtered annotation
        out_ann_path = os.path.join(human_labels_dir, ann_file)
        with open(out_ann_path, "w") as f:
            f.writelines(human_lines)

print("Done! Human images and annotations are saved in 'Humans/' folder.")
