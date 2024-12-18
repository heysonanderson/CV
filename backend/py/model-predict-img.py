from ultralytics import YOLO

photo_path = ["D:/fullstack/CV/backend/predict/photo_5406897710499488781_y.jpg",
              "D:/fullstack/CV/backend/predict/photo_5406897710499488782_y.jpg",
              "D:/fullstack/CV/backend/predict/2024-12-17_20-59-50.png"]
model = YOLO('runs/detect/train3/weights/last.pt')
results = model(photo_path)  # return a list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen