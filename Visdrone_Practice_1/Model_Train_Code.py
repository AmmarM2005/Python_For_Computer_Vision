# train_yolov8.py
from ultralytics import YOLO

# Path to your dataset YAML file
dataset_yaml = r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\YOLO_dataset\dataset.yaml"

# Load a YOLOv8n (nano) model — lightweight and good for real-time
model = YOLO("yolov8n.pt")  # pretrained model

# Train the model
model.train(
    data=dataset_yaml,  # dataset YAML
    epochs=50,          # number of epochs
    imgsz=640,          # image size
    batch=16,           # batch size
    device="cpu",       # CPU training (you can change to GPU if available)
    name="yolov8_human_drone",  # folder name to save results
    project=r"C:\Users\Ammar\Python_Programming\Pycharm_Projects\Survivor_Detection_Intermediate\runs"
)
