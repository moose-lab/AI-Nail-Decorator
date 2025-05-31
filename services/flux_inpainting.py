import torch

from PIL import Image

from diffusers.utils import check_min_version
from .flux.controlnet_flux import FluxControlNetModel
from .flux.transformer_flux import FluxTransformer2DModel
from .flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# Input the hf_token to download the model
import huggingface_hub
huggingface_hub.login("")

check_min_version("0.30.2")

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

size = (1024, 1024)

def generate_manicure(input_image, input_mask, text_prompt=""):
    raw_image = Image.open(input_image).convert("RGB").resize(size)
    mask = Image.open(input_mask).convert("RGB").resize(size)
    generator = torch.Generator(device="cuda").manual_seed(24)

    # Inpaint
    result = pipe(
        prompt=text_prompt,
        height=size[1],
        width=size[0],
        control_image=raw_image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    return result.resize(input_image.size[:2])

