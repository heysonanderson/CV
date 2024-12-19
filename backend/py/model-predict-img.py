from ultralytics import YOLO

photo_path = ["../predict/2024-12-17_20-59-50.png",
              "../predict/photo_5409149510313174660_y.jpg",
              "../predict/photo_5409149510313174661_y.jpg",
              "../predict/photo_5409149510313174663_y.jpg",
              "../predict/photo_5409149510313174664_y.jpg",
              "../predict/photo_5409149510313174665_y.jpg",
              "../predict/photo_5409149510313174666_y.jpg",
              "../predict/photo_5409149510313174667_y.jpg",
              "../predict/photo_5409149510313174668_y.jpg",
              "../predict/photo_5409149510313174669_y.jpg",
              "../predict/photo_5409149510313174670_y.jpg",
              "../predict/photo_5409149510313174671_y.jpg",
              "../predict/photo_5409149510313174672_y.jpg",
              "../predict/photo_5409149510313174673_y.jpg",
              "../predict/photo_5409149510313174674_y.jpg"
              ]
model = YOLO('runs/detect/train/weights/best.pt')
results = model(photo_path)  # return a list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen