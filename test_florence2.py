from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from glob import glob
import cv2
import copy



def run_example(model, processor, task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.shape[1], image.shape[0])
    )

    return parsed_answer

import random
import numpy as np
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_florence_polygons(image, prediction, fill_color=(255, 0, 0)):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate over polygons and labels
    for polygons in prediction['polygons']:
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = _polygon.reshape(-1, 2)
            cv2.fillPoly(mask, [np.int32(_polygon)], 255)
    cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = mask.astype(bool)
    image[mask] = fill_color
    return mask


if __name__ == '__main__':
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
    text_input = 'A suspended load'
    image_paths = glob('assets/images/*')
    output_dir = 'assets/loads'
    for image_path in image_paths:
        print(image_path)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = run_example(model, processor, task_prompt, image, text_input=text_input)
        draw_florence_polygons(image, results['<REFERRING_EXPRESSION_SEGMENTATION>'])
        cv2.imwrite(f'{output_dir}/' + image_path.split('/')[-1], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))