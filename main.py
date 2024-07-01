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

def distance_to_plane(XYZ, plane):
    v1, v2, c = plane
    normal_vector = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal_vector)
    normal_vector_normalized = normal_vector / normal_norm
    c = c.reshape(1, 1, 3)
    normal_vector_normalized = normal_vector_normalized.reshape(1, 1, 3)
    diff = XYZ - c
    dot_product = np.sum(diff * normal_vector_normalized, axis=2)
    distances = np.abs(dot_product)
    return distances

def fit_ground_single(XYZ, p):
    px, py = p
    candidates = XYZ[py-20:py+20, px-50:px+50].reshape((-1, 3))
    for i in range(5):
        print(candidates.shape)
        if len(candidates) > 5000:
            candidates = candidates[np.random.choice(len(candidates), 5000, replace=False)]
        plane = fit_plane_to_points(candidates)
        print(plane)
        dist = distance_to_plane(XYZ, plane)
        dist_filter = dist < 0.1 / 2**i
        candidates = XYZ[dist_filter]
    return plane


def fit_ground(XYZ):
    height = XYZ.shape[0]
    width = XYZ.shape[1]
    p1x = int(width * 1/6)
    p2x = int(width * 4/6)
    p3x = int(width * 5/6)
    p1y = int(height-20)
    p2y = int(height-20)
    p3y = int(height-20)
    
    ps = [(p1x, p1y), (p2x, p2y), (p3x, p3y)]
    planes = []
    xs = []

    #return fit_ground_single(XYZ, ps[0])

    for p in ps:
        plane = fit_ground_single(XYZ, p)
        planes.append(plane)
        v1, v2, c = plane
        x = np.cross(v1, v2)
        x = x / np.linalg.norm(x)
        xs.append(x)

    # majority vote
    x01 = abs(np.dot(xs[0], xs[1]))
    x02 = abs(np.dot(xs[0], xs[2]))
    x12 = abs(np.dot(xs[1], xs[2]))
    xxx = min(x01, x02, x12)
    if xxx == x01:
        return planes[0]
    if xxx == x02:
        return planes[2]
    if xxx == x12:
        return planes[1]


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

    #cv2.imshow('image', cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'assets/outputs/{image_name}.png', cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR))
