import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import copy
from ultralytics import YOLO
import random
import os
from test_florence2 import run_example, draw_florence_polygons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the image
image_path = 'assets/images/Default_A_small_crane_holding_a_suspended_load_above_the_groun_3.jpg'
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Ground Mask
image_name = os.path.basename(image_path).split('.')[0]
ground_image = cv2.imread(f'assets/ground/{image_name}.png')
ground_mask = ground_image[:, :, 0] > 0
show_image[ground_mask] = (255, 255, 255)


# DepthAnythingV2
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vits' # or 'vitb', 'vitl'

model_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
model.to(device)

depth = model.infer_image(image) # HxW raw depth map

# YOLO
model = YOLO("checkpoints/yolov8n-seg.pt")  # load an official model
model.to(device)
classes_ids = [0]
conf = 0.5
results = model.predict(image, conf=conf)
person_contours = []
person_color = (0, 255, 0)
person_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
for result in results:
    if result.masks is None:
        continue
    for mask, box in zip(result.masks.xy, result.boxes):
        if int(box.cls[0]) not in classes_ids:
            continue
        points = np.int32([mask])
        cv2.fillPoly(person_mask, points, 255)
        person_contours.append(points)

person_mask = cv2.erode(person_mask, np.ones((5, 5), np.uint8), iterations=1)
person_mask = person_mask.astype(bool)
show_image[person_mask] = person_color

# Florence 2
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
text_input = 'A suspended load'
results = run_example(model, processor, task_prompt, rgb_image, text_input=text_input)
results = results['<REFERRING_EXPRESSION_SEGMENTATION>']
load_mask = draw_florence_polygons(show_image, results, fill_color=(255, 0, 0))

# now we have ground mask, person mask, and load mask

