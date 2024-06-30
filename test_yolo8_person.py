from ultralytics import YOLO
import random
import cv2
import numpy as np
from glob import glob
# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
# We want only the person class
classes_ids = [0]

conf = 0.5

# Predict with the model
image_paths = glob('assets/images/*')
output_dir = 'assets/persons/'

for image_path in image_paths:
    img = cv2.imread(image_path)  # predict on an image

    results = model.predict(img, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    print(f'{results=}')
    for result in results:
        if result.masks is None:
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) not in classes_ids:
                continue
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(img, points, colors[color_number])

    cv2.imwrite(output_dir + image_path.split('/')[-1], img)