from ultralytics import YOLO
import os

# Load your trained model
model = YOLO('runs/detect/train6/weights/best.pt')

# Set input and output directories
input_dir = 'datasets/ds4000/test/images/'  # Folder with images
output_dir = 'runs/detect/'

# Run inference
results = model(source=input_dir, save=True, save_txt=True, project=output_dir, exist_ok=True)

# Optionally, access and process results
for i, r in enumerate(results):
    print(f"[Image {i}] Detected {len(r.boxes)} objects")
    for box in r.boxes:
        print(" Class:", int(box.cls.item()), "Confidence:", float(box.conf.item()), "Box:", box.xyxy.tolist())
