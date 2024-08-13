import numpy as np
import os
import os.path as pth
import cv2
from PIL import Image
import torch
import io
from skimage.measure import label, regionprops, find_contours
import schp

import csv
import gradio as gr
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from modules import devices, lowvram, script_callbacks, shared
from modules.api import api
from typing import Optional, Set
from pydantic import BaseModel


# det_model = None
# seg_model = None
# seg_model_2 = None
# lndmrk_model = None
dataset_type = None
seg_model = None

model_dir = os.path.join('models', 'clotheseg')

model_list = ['SCHP (lip)', 'SCHP (atr)', 'SCHP (pascal)']

part_label_list = []
part_label_list += schp.model.dataset_settings['atr']['label']
part_label_list += schp.model.dataset_settings['lip']['label']
part_label_list += schp.model.dataset_settings['pascal']['label']
part_label_list = sorted(list(set(part_label_list)))


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def image_to_mask(image, model, included_parts, dilation_percentage=0, type_='pil'):
    global dataset_type
    global seg_model

    if model == 'SCHP (lip)':
        if dataset_type != 'lip':
            dataset_type = 'lip'
            seg_model = schp.SCHP(
                dataset_type=dataset_type, model_dir=model_dir
            )

    elif model == 'SCHP (atr)':
        if dataset_type != 'atr':
            dataset_type = 'atr'
            seg_model = schp.SCHP(
                dataset_type=dataset_type, model_dir=model_dir
            )

    elif model == 'SCHP (pascal)':
        if dataset_type != 'pascal':
            dataset_type = 'pascal'
            seg_model = schp.SCHP(
                dataset_type=dataset_type, model_dir=model_dir
            )

    original_input_image = image.copy()

    images = [
        image,
    ]
    images = np.stack(images)

    if seg_model:
        human_parsing_results = seg_model.parse(
            images=images, 
        )
        human_parsing_results = human_parsing_results[0]


        include_mask_list = []
        label = schp.model.dataset_settings[dataset_type]['label']
        for included_part in included_parts:
            if included_part in label:
                part_i = label.index(included_part)
                if part_i >= 0:
                    include_mask = (human_parsing_results==part_i)
                    include_mask_list.append(include_mask)
    else:
        include_mask_list = []
    
    include_mask = np.zeros(original_input_image.shape[:2], dtype=np.uint8)
    for each_mask in include_mask_list:
        include_mask = np.bitwise_or(include_mask, each_mask.astype(np.uint8))
    
    if dilation_percentage > 0:
        bboxes = mask_to_bbox(include_mask)
        if bboxes:
            bbox = sorted(bboxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]),reverse=True)[0] # largest_bbox
            fileter_size_w = int((bbox[2]-bbox[0]) * dilation_percentage/100)
            fileter_size_h = int((bbox[3]-bbox[1]) * dilation_percentage/100)
            if fileter_size_w > 1 and fileter_size_h > 1:
                kernel = np.ones((fileter_size_h, fileter_size_w), np.uint8)
                include_mask = cv2.dilate(include_mask, kernel, iterations=1)

    merged_mask = include_mask

    merged_mask = merged_mask[..., np.newaxis]
    merged_mask = merged_mask.astype(np.uint8)
    merged_mask *= 255
    merged_mask = np.tile(
        merged_mask, 
        reps=3
    )

    masked_image = None
    merged_mask_temp = (merged_mask == 255)
    masked_image = original_input_image.copy()
    masked_image[~merged_mask_temp] = 0 

    if type_=='pil' and merged_mask is not None:
        merged_mask = Image.fromarray(merged_mask)
        masked_image = Image.fromarray(masked_image)

    return [masked_image, merged_mask]


def unload_model(type_):
    return True


def add_tab():
    device = devices.get_optimal_device()
    vram_total = torch.cuda.get_device_properties(device).total_memory
    if vram_total <= 12*1024*1024*1024:
        low_vram = True

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Single"):
            single_tab()

    return [(ui, "Clotheseg", "Clothes Segmentation")]


def single_tab():
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='numpy', label="Image")
        with gr.Column():
            # mask = gr.Image(type='numpy', label="Mask")
            results = gr.Gallery(label="Results").style(grid=2)
            # label_results = gr.Textbox(label="label results", lines=3)
    with gr.Row():
        model = gr.Dropdown(label="Segmentation model", choices=model_list, value="None")
        return_mask = gr.Checkbox(label="Return mask", value=False)
        included_parts = gr.CheckboxGroup(part_label_list, label="Included parts")
        # excluded_parts = gr.CheckboxGroup(part_label_list, label='Excluded parts')
        dilation_percentage = gr.Slider(0, 100, value=0, label="dilation size (%)")
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Model unload")
    button.click(image_to_mask, inputs=[image, model, included_parts, dilation_percentage], outputs=results)
    unload_button.click(unload_model)


def base64_to_RGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 

def encode_np_to_base64(img):
    pil = Image.fromarray(img)
    return api.encode_pil_to_base64(pil)

def RGB_to_base64(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return encode_np_to_base64(img) 

def mount_api(_: gr.Blocks, app: FastAPI):
    @app.get(
        "/clothesg/models",
        summary="Get model type strs",
        description="Currnetly we support 'SCHP (lip)', 'SCHP (atr)', 'SCHP (pascal)'",
    )
    async def get_models():
        return ['SCHP (lip)', 'SCHP (atr)', 'SCHP (pascal)']

    @app.get(
        "/clothesg/labels",
        summary="Get segmentation part strs",
        description="Valid part strings for /clothesg/img2mask",
    )
    async def get_labels(model: str):
        if model == 'SCHP (lip)':
            labels = schp.model.dataset_settings['lip']['label']
        elif model == 'SCHP (atr)':
            labels = schp.model.dataset_settings['atr']['label']
        elif model == 'SCHP (pascal)':
            labels = schp.model.dataset_settings['pascal']['label']
        else:
            labels = []

        return labels

    class Img2MaskItem(BaseModel):
        img: str
        model: str
        include_parts: list[str]
        # exclude_parts: Optional[list[str]] = []
        dilate_percent: Optional[int] = 0
        # want_one_big_face: bool = False

    @app.post(
        "/clothesg/img2mask",
        summary="Get segmentation mask",
        description="Get segmentation mask from clothes image",
    )
    async def img2mask(item:Img2MaskItem):
        """
        - **img**: A portrait image to generate binary mask.
        - **mdoel**: Segmentation model you want to use.
        - **include_parts**: Parts you need to include.
        - **dilate_percent (Optional)**: Dilation percentage you need to applied.
        """
        img = base64_to_RGB(item.img)

        masked_image, merged_mask = image_to_mask(
            img, 
            model=item.model, 
            included_parts=item.included_parts, 
            dilation_percentage=item.dilate_percent, 
            type_='numpy',
        )

        result_dict= {
            'masked_image': RGB_to_base64(masked_image), 
            'mask': RGB_to_base64(merged_mask)
        }

        return result_dict


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_app_started(mount_api)
