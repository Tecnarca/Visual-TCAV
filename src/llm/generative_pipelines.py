import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import base64
import openai
import time
import io
from PIL import Image

import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not set in environment")

"""
import torch
import tensorflow as tf

# PyTorch CUDA availability
print("PyTorch:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device count : {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name  : {torch.cuda.get_device_name(torch.cuda.current_device())}")

# TensorFlow CUDA availability
print("\nTensorFlow:")
print(f"  CUDA available: {tf.config.list_physical_devices('GPU') != []}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
"""

class GPTImageWrapper:
    def generate(self, prompt, n):
        response = openai.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=n,
            size="1024x1024",
            output_format="png",
            quality="low",
        )
        return [Image.open(io.BytesIO(base64.b64decode(img.b64_json))) for img in response.data]

    def edit(self, image_path, prompt):
        with open(image_path, "rb") as f:
            response = openai.images.edit(
                model="gpt-image-1",
                image=f,
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="low",
            )
        return Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))

class ImageGenerator:
    MODELS = {
        "stable_diffusion": (StableDiffusion3Pipeline, "stabilityai/stable-diffusion-3.5-medium", 40, 4.5),
        "flux": (FluxPipeline, "black-forest-labs/FLUX.1-schnell", 4, 1.0),
        "gpt-image-1": (GPTImageWrapper, None, None, None)
    }

    def __init__(self, model, steps, scale, negative_prompt):
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}")

        pipeline_cls, model_id, default_steps, default_scale = self.MODELS[model]
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps or default_steps
        self.scale = scale or default_scale
        self.negative_prompt = negative_prompt

        if model == "gpt-image-1":
            self.pipeline = pipeline_cls()
        else:
            self.pipeline = pipeline_cls.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)

    def generate_images(self, prompt, num_images, output_dir):
        chunk_size = 5 if self.model == "gpt-image-1" else 30
        output_dir.mkdir(parents=True, exist_ok=True) if output_dir else None

        for start in range(0, num_images, chunk_size):
            batch = min(chunk_size, num_images - start)
            images = (self.pipeline.generate(prompt, batch) if self.model == "gpt-image-1"
                      else self.pipeline(prompt, batch, self.steps, self.scale, self.negative_prompt).images)

            for i, image in enumerate(images, start=start):
                if output_dir:
                    image.save(output_dir / f"{self.model}_{i+1}.png")
                    print(f"Saved: {output_dir / f'{self.model}_{i+1}.png'}")
                else:
                    plt.imshow(image)
                    plt.axis('off')
                    plt.show()

            if self.model == "gpt-image-1":
                time.sleep(60)

    def edit_images(self, image_path, prompt, output_dir):
        if self.model != "gpt-image-1":
            raise ValueError("Edit functionality is only available for GPT-image-1.")

        image_paths = [image_path] if image_path.is_file() else list(image_path.glob("*.png")) + list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg"))

        output_dir.mkdir(parents=True, exist_ok=True) if output_dir else None

        for i, img_path in enumerate(image_paths):
            edited_image = self.pipeline.edit(img_path, prompt)
            if output_dir:
                output_path = output_dir / f"edited_{img_path.name}"
                edited_image.save(output_path)
                print(f"Saved edited image: {output_path}")
            else:
                plt.imshow(edited_image)
                plt.axis('off')
                plt.show()

            if (i + 1) % 5 == 0 and i < len(image_paths) - 1:
                print("Rate limit reached, waiting 60 seconds...")
                time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--model", choices=["flux", "stable_diffusion", "gpt-image-1"], default="flux")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--negative_prompt")
    parser.add_argument("--edit_image_path", type=Path)

    args = parser.parse_args()

    gen = ImageGenerator(args.model, args.steps, args.scale, args.negative_prompt)

    if args.edit_image_path:
        gen.edit_images(args.edit_image_path, args.prompt, args.output_dir)
    else:
        gen.generate_images(args.prompt, args.num_images, args.output_dir)
