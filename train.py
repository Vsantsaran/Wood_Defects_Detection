import os
import torch
from ultralytics import YOLO

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'  # Set the visible CUDA devices
print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# print(ultralytics.checks())
data = 'data.yaml'  # Path to the dataset configuration file 
if not os.path.exists(data):
    raise FileNotFoundError(f"Dataset configuration file '{data}' does not exist.")

model = YOLO('YOLO_MODELS/yolov8l.pt')
# model = nn.DataParallel(model).to(device)
model.train(data=data, epochs=80, batch=32, imgsz=640, save=True, plots=True, workers=72, cos_lr=True, cache='disk', device='0,1,2,3,4,5,6,7')

