import os

import cv2
import gradio as gr

from services.florence_vlm import (
    load_florence_model, 
    run_florence_inference,
    FLORENCE_DETAILED_CAPTION_TASK,
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
)
from services.segment_anything_model import load_sam_image_model, run_sam_inference


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