from ultralytics import YOLO

model = YOLO('runs/detect/train9/weights/best.pt')

#results = model.train(data = 'd:/fullstack/cv/backend/ds/Fruits&nonFruits/data.yaml', epochs = 500, imgsz = 640)

results = model(["D:/fullstack/CV/backend/predict/photo_5406897710499488781_y.jpg", "D:/fullstack/CV/backend/predict/photo_5406897710499488782_y.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen