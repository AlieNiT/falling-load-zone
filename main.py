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
import math
from utils import *
from glob import glob

FIELD_OF_VIEW = 87

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
image_paths = glob('assets/images/*')
mask_paths = set(glob('assets/ground/*'))

# Load the image
for image_path in image_paths:
    image_name = os.path.basename(image_path).split('.')[0]
    if f'assets/ground/{image_name}.png' not in mask_paths:
        continue
    
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
    depth = 1 / depth
    recognizable_pixels = np.expand_dims(depth < 1, axis=-1)
    f = depth.shape[1] / (2 * math.tan(math.radians(FIELD_OF_VIEW / 2)))
    XYZ = Z_to_XYZ(depth, f=f, w=depth.shape[1], h=depth.shape[0])

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

    person_mask = cv2.erode(person_mask, np.ones((3, 3), np.uint8), iterations=1)
    person_mask = person_mask.astype(bool)
    show_image[person_mask] = person_color


    plane = fit_ground(XYZ)
    print(plane)

    person_XYZ = XYZ[person_mask]
    projected_person_XYZ = project_points_to_plane(person_XYZ, plane)
    projected_person_xy = XYZ_to_xy(projected_person_XYZ, f=f, w=depth.shape[1], h=depth.shape[0])
    projected_person_color = (136, 8, 8)
    projected_person_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for xy in projected_person_xy:
        x, y = xy
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            projected_person_mask[y, x] = 255

    projected_person_mask = cv2.erode(projected_person_mask, np.ones((3, 3), np.uint8), iterations=1)
    projected_person_mask = projected_person_mask.astype(bool)



    # Florence 2
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
    text_input = 'A suspended load'
    results = run_example(model, processor, task_prompt, rgb_image, text_input=text_input)
    results = results['<REFERRING_EXPRESSION_SEGMENTATION>']
    load_mask = draw_florence_polygons(show_image, results, fill_color=(255, 0, 0))

    load_XYZ = XYZ[load_mask]
    projected_load_XYZ = project_points_to_plane(load_XYZ, plane)
    projected_load_xy = XYZ_to_xy(projected_load_XYZ, f=f, w=depth.shape[1], h=depth.shape[0])
    projected_load_color = (255, 0, 0)
    projected_load_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for xy in projected_load_xy:
        x, y = xy
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            projected_load_mask[y, x] = 255


    projected_load_mask = cv2.erode(projected_load_mask, np.ones((3, 3), np.uint8), iterations=1)
    projected_load_mask = projected_load_mask.astype(bool)
    projected_load_xy = np.array(np.where(projected_load_mask)).T[:, ::-1]
    hull = cv2.convexHull(projected_load_xy)
    cv2.polylines(show_image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
    # convex hull around projected load and not the contours
    # outline the polygon on show_image

    show_image[projected_person_mask] = projected_person_color

    # load ground truth for person and load
    ground_person_mask = cv2.imread(f'assets/label_masks/{image_name}_people.png')
    ground_person_mask = cv2.cvtColor(ground_person_mask, cv2.COLOR_BGR2GRAY)
    ground_person_mask = ground_person_mask.astype(bool)

    ground_load_mask = cv2.imread(f'assets/label_masks/{image_name}_load.png')
    ground_load_mask = cv2.cvtColor(ground_load_mask, cv2.COLOR_BGR2GRAY)
    ground_load_mask = ground_load_mask.astype(bool)

    # calculate IoU on projected masks
    person_iou = IoU(projected_person_mask, ground_person_mask)
    load_iou = IoU(projected_load_mask, ground_load_mask)

    print(f'Image: {image_name}\n\tPerson IoU: {person_iou}\n\tLoad IoU: {load_iou}')


    # cv2.imshow('image', cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f'assets/outputs/{image_name}.png', cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR))
