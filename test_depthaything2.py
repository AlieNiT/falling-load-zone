import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vits' # or 'vitb', 'vitl'

traced_model_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(traced_model_path, map_location='cpu'))
model.eval()
model.to(device)

image_paths = glob('assets/images/*')
output_dir = 'assets/depths'
for image_path in image_paths:
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img) # HxW raw depth map
    depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8).squeeze() # For visualization purposes

    # save the image
    cv2.imwrite(f'{output_dir}/' + image_path.split('/')[-1], depth)