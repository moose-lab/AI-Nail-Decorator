import os
from typing import Tuple, Optional

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm

from services.florence_vlm import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from services.segment_anything_model import load_sam_image_model, run_sam_inference
# from services.flux_inpainting import generate_manicure
from services.sdxl_inpainting import generate_manicure

DEVICE = torch.device("cuda")


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
# SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)

def generate_output(image, prompt, detections):
    output_image = image.copy()
    mask_image = np.zeros((image.height, image.width), dtype=np.uint8)
    for mask in detections.mask:
        mask_image[mask] = 255
    mask_image = Image.fromarray(mask_image)

    # 新增：创建目标的透明图
    obj_image = Image.new("RGBA", output_image.size)
    obj_image.paste(output_image, (0, 0), mask_image)  # 使用遮罩图作为透明度

    # 创建：生成结果图 - 修复生成器调用
    output = list(generate_manicure(output_image, mask_image, prompt))[-1]

    return output, obj_image

def process_image(
    image_input, text_prompt = "", texts = ["all human fingernails"]
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None

    detections_list = []
    # firstly florence-2 model to detect the object box border via ovd model
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )

        # secondly, extract the object mask via sam model
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
    return generate_output(image_input, text_prompt, detections)


with gr.Blocks() as demo:
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_processing_image_input_component = gr.Image(
                    type='pil', label='Upload image')
                text_prompt_component = gr.Textbox(
                    label="Text prompt", placeholder="Enter comma separated text prompts"
                )
                image_processing_submit_button_component = gr.Button(
                    value='Submit', variant='primary')
            with gr.Column():
                image_processing_image_output_component = gr.Image(
                    type='pil', label='Image output')
                image_processing_mask_output_component = gr.Image(
                    type='pil', label='Mask output')
                # image_processing_text_output_component = gr.Textbox(
                #     label='Caption output', visible=False)

    image_processing_submit_button_component.click(
        fn=process_image,
        inputs=[
            image_processing_image_input_component,
            text_prompt_component
        ],
        outputs=[
            image_processing_image_output_component,
            image_processing_mask_output_component,
            # image_processing_text_output_component
        ]
    )

demo.launch(debug=False,
    server_name="0.0.0.0",
    server_port=7778,)