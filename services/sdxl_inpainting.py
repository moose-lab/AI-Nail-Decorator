import torch
from PIL import Image

from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict

from .sdxl.controlnet_union import ControlNetModel_Union
from .sdxl.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from huggingface_hub import hf_hub_download
config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")

# 确保所有组件都使用正确的数据类型
if pipe.text_encoder is not None:
    pipe.text_encoder.to(dtype=torch.float16)
if pipe.text_encoder_2 is not None:
    pipe.text_encoder_2.to(dtype=torch.float16)
pipe.unet.to(dtype=torch.float16)
pipe.vae.to(dtype=torch.float16)
pipe.controlnet.to(dtype=torch.float16)

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

def generate_manicure(input_image, input_mask, text_prompt="", paste_back=False):
    # 强制使用半精度
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(text_prompt, torch.device("cuda"), True)

        # 支持PIL Image对象或文件路径
        if isinstance(input_image, str):
            raw_image = Image.open(input_image)
        else:
            raw_image = input_image
        
        if isinstance(input_mask, str):
            mask = Image.open(input_mask)
        else:
            mask = input_mask

        # 处理不同格式的mask图像
        if mask.mode == 'RGBA':
            alpha_channel = mask.split()[3]
            binary_mask = alpha_channel.point(lambda p: p > 0 and 255)
        elif mask.mode == 'L':
            # 灰度图像，直接使用
            binary_mask = mask.point(lambda p: p > 0 and 255)
        else:
            # 其他格式，转换为灰度然后使用
            binary_mask = mask.convert('L').point(lambda p: p > 0 and 255)
        
        cnet_image = raw_image.copy()
        cnet_image.paste(0, (0, 0), binary_mask)

        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
        ):
            yield cnet_image

        if paste_back:
            image = image.convert("RGBA")
            cnet_image.paste(image, (0, 0), binary_mask)
        else:
            cnet_image = image

        yield cnet_image