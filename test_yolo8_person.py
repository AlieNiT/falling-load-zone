from ultralytics import YOLO
import random
import cv2
import numpy as np
from glob import glob
# Load a model
model = YOLO("checkpoints/yolov8n-seg.pt")  # load an official model
# We want only the person class
classes_ids = [0]
color = (0, 255, 0)

conf = 0.5

# Predict with the model
image_paths = glob('assets/images/*')
output_dir = 'assets/persons/'

for image_path in image_paths:
    img = cv2.imread(image_path)  # predict on an image

    results = model.predict(img, conf=conf)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f'{results=}')
    for result in results:
        if result.masks is None:
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) not in classes_ids:
                continue
            points = np.int32([mask])
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(img, points, color)

    cv2.imwrite(f'{output_dir}/' + image_path.split('/')[-1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))