from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

ds_path = 'd:/fullstack/cv/backend/ds/Fruits&nonFruits/data.yaml'

results = model.train(data = ds_path, epochs = 60, imgsz = 640)