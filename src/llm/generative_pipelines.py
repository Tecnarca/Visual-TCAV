import argparse
import base64
import io
import math
import os
import re
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import openai
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from huggingface_hub import login


# ---------- Auth ----------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not set in environment")
login(token=hf_token)


# ---------- Small utilities ----------
WAIT_PATTERNS = [
    r'(?i)(?:wait|retry(?:\s+after)?|try again in|try again after)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b',
    r'(?i)in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b',
    r'(?i)(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b',
]

def hhmm() -> str:
    return datetime.now().strftime("%H:%M")

def log(msg: str) -> None:
    print(f"[{hhmm()}] {msg}")

def parse_retry_after_seconds(exc: Exception, default_seconds: float = 5.0) -> int:
    msg = str(exc)
    for pat in WAIT_PATTERNS:
        m = re.search(pat, msg)
        if m:
            try:
                return math.ceil(float(m.group(1)))
            except ValueError:
                pass
    return math.ceil(default_seconds)

def is_moderation_block(exc: Exception) -> bool:
    s = str(exc).lower()
    return "moderation_blocked" in s or "safety system" in s

def error_label(exc: Exception) -> str:
    if isinstance(exc, openai.RateLimitError):
        return "Rate Limit"
    if isinstance(exc, openai.BadRequestError):
        return "Bad Request (moderation)" if is_moderation_block(exc) else "Bad Request"
    return exc.__class__.__name__

def save_or_show(images, output_dir: Path | None, prefix: str):
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(output_dir.glob(f"{prefix}_*.png"))
        next_idx = max([int(p.stem.split("_")[-1]) for p in existing], default=0) + 1
        for i, im in enumerate(images, start=next_idx):
            path = output_dir / f"{prefix}_{i}.png"
            im.save(path)
            log(f"Saved: {path}")
    else:
        for im in images:
            plt.imshow(im)
            plt.axis("off")
            plt.show()


# ---------- GPT image wrapper with adaptive backoff ----------
class GPTImageClient:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def generate(self, prompt: str, n: int):
        attempts_generic = 0
        moderation_retry_used = False
        while True:
            try:
                resp = openai.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    n=n,
                    size="1024x1024",
                    output_format="png",
                    quality="low",
                )
                print(f"[{hhmm()}] Revised prompt: {resp.data[0].revised_prompt}")
                return [Image.open(io.BytesIO(base64.b64decode(d.b64_json))) for d in resp.data]
            except (openai.BadRequestError, openai.RateLimitError) as e:
                # Moderation handling (BadRequest only)
                if isinstance(e, openai.BadRequestError) and is_moderation_block(e):
                    if moderation_retry_used:
                        log(f"[{hhmm()}] Bad Request (moderation) again → skipping")
                        return None
                    log(f"[{hhmm()}] Bad Request (moderation) → retrying once…")
                    moderation_retry_used = True
                    continue

                # Adaptive backoff for pacing/limit errors
                attempts_generic += 1
                if attempts_generic > self.max_retries:
                    raise
                wait_s = parse_retry_after_seconds(e)
                label = error_label(e)
                log(f"[{hhmm()}] {label} → wait {wait_s}s then retry (attempt {attempts_generic}/{self.max_retries})")
                time.sleep(wait_s)

    def edit(self, image_path: Path, prompt: str):
        """
        - On moderation block: retry once, then return None (caller will skip).
        - On other BadRequestError / RateLimitError: adaptive backoff up to max_retries.
        """
        attempts_generic = 0
        moderation_retry_used = False

        while True:
            try:
                with open(image_path, "rb") as f:
                    resp = openai.images.edit(
                        model="gpt-image-1",
                        image=f,
                        prompt=prompt,
                        n=1,
                        size="1024x1024",
                        quality="low",
                    )
                return Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json)))

            except (openai.BadRequestError, openai.RateLimitError) as e:
                # Moderation handling (BadRequest only)
                if isinstance(e, openai.BadRequestError) and is_moderation_block(e):
                    if moderation_retry_used:
                        log(f"[{image_path.name}] Bad Request (moderation) again → skipping")
                        return None
                    log(f"[{image_path.name}] Bad Request (moderation) → retrying once…")
                    moderation_retry_used = True
                    continue

                # Adaptive backoff for pacing/limit errors
                attempts_generic += 1
                if attempts_generic > self.max_retries:
                    raise
                wait_s = parse_retry_after_seconds(e)
                label = error_label(e)
                log(f"[{image_path.name}] {label} → wait {wait_s}s then retry (attempt {attempts_generic}/{self.max_retries})")
                time.sleep(wait_s)


class ImageGenerator:
    MODELS = {
        "sd35": (StableDiffusion3Pipeline, "stabilityai/stable-diffusion-3.5-medium", 40, 4.5),
        "flux":  (FluxPipeline,           "black-forest-labs/FLUX.1-schnell",       4,  1.0),
        "gpti1": (GPTImageClient,         None,                                      None, None),
    }

    def __init__(self, model: str, steps: int | None, scale: float | None, negative_prompt: str | None):
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}")
        pipe_cls, model_id, default_steps, default_scale = self.MODELS[model]

        self.model = model
        self.steps = steps or default_steps
        self.scale = scale or default_scale
        self.negative_prompt = negative_prompt

        if model == "gpti1":
            self.pipeline = pipe_cls()
        else:
            self.pipeline = pipe_cls.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
            # memory helpers
            try:
                self.pipeline.enable_sequential_cpu_offload()
                self.pipeline.enable_attention_slicing()   # <— extra memory saver
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.enable_tiling()
            except Exception:
                pass
            self.pipeline.to(torch.float16)

    def generate_images(self, prompt: str, num_images: int, output_dir: Path | None, batch_size: int = 2):
        if self.model == "gpti1":
            remaining = num_images
            all_imgs = []
            while remaining > 0:
                b = min(5, remaining)
                gen = self.pipeline.generate(prompt, b)
                if gen is None:  # moderation blocked twice → skip
                    continue
                all_imgs.extend(gen)
                remaining -= b
                print(f"[{hhmm()}] Generated {b}/{len(all_imgs)}")
            save_or_show(all_imgs, output_dir, prefix="gpti1")
            return

        # --- SD/FLUX batched generation to avoid OOM ---
        remaining = num_images
        while remaining > 0:
            b = min(batch_size, remaining)
            images = self.pipeline(
                prompt=prompt,
                num_images_per_prompt=b,
                num_inference_steps=self.steps,
                guidance_scale=self.scale,
                negative_prompt=self.negative_prompt,
                height=512,
                width=512,
                generator=torch.Generator("cpu").manual_seed(0),
            ).images
            save_or_show(images, output_dir, prefix=self.model)
            remaining -= b

    def edit_images(self, image_path: Path, prompt: str, output_dir: Path | None):
        if self.model != "gpti1":
            raise ValueError("Edit functionality is only available for GPT-image-1.")

        paths = [image_path] if image_path.is_file() else [
            *image_path.glob("*.png"), *image_path.glob("*.jpg"), *image_path.glob("*.jpeg")
        ]

        for p in paths:
            edited = self.pipeline.edit(p, prompt)
            if edited is None:  # moderation blocked twice → skip
                continue

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / p.name  # keep original filename
                edited.save(out_path)
                log(f"Saved edited image: {out_path}")
            else:
                plt.imshow(edited)
                plt.axis("off")
                plt.show()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--model", choices=["flux", "sd35", "gpti1"], default="flux")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--negative_prompt")
    parser.add_argument("--edit_image_path", type=Path)
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for SD/FLUX to control VRAM")
    args = parser.parse_args()

    gen = ImageGenerator(args.model, args.steps, args.scale, args.negative_prompt)

    if args.edit_image_path:
        gen.edit_images(args.edit_image_path, args.prompt, args.output_dir)
    else:
        gen.generate_images(args.prompt, args.num_images, args.output_dir, batch_size=args.batch_size)
