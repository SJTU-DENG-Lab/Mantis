import os
import sys
import torch
from PIL import Image
sys.path.append(os.getcwd())
from models.mantis import Mantis


model_path = "Yysrc/SSV2-Pretrained"
target_image_size = 512
vae_downsample_f = 32
input_size = target_image_size / vae_downsample_f

mantis = Mantis.from_pretrained(
    model_path,
    device="cuda",
)
mantis = mantis.to(torch.float32)
device = next(mantis.parameters()).device

input_images = [
    Image.open("image.png").convert("RGB").resize((512, 512))
]
instruction = "your_instruction"
timestep_gap = 1
caption = f"Instruction: {instruction}. Generate the updated image observation after {timestep_gap} timesteps."

samples = mantis.sample_images(
    caption=caption,
    input_images=input_images,
    negative_prompt="",
    num_inference_steps=30,
    num_images_per_prompt=1,
    gap=timestep_gap,
)

samples[0].save(f"sample.png")
