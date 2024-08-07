import numpy as np
import os
import os.path as pth
# import facer
import cv2
from PIL import Image
import torch
import io

# from src.face_landmark_detector import face_aligner

import csv
import gradio as gr
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from modules import devices, lowvram, script_callbacks, shared
from modules.api import api
# from typing import Optional, Set
# from pydantic import BaseModel

import schp


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


def image_to_mask(image, model, included_parts, face_dilation_percentage=0, type_='pil'):
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
    
    include_mask = np.zeros_like(original_input_image, dtype=np.uint8)
    for each_mask in include_mask_list:
        include_mask = np.bitwise_or(include_mask, each_mask.astype(np.uint8))
        
    merged_mask = include_mask

    merged_mask = merged_mask.astype(np.uint8)
    merged_mask *= 255
    print('???')
    print(merged_mask.shape)
    print('???')
    # merged_mask = np.tile(
    #     merged_mask, 
    #     reps=3
    # )

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


# script_callbacks.on_app_started(mount_facer_api)
script_callbacks.on_ui_tabs(add_tab)
