import argparse
import base64
import time
from pathlib import Path

import openai

from src.loggers import LOGGER

prompt = "Make the waffled parts be flat"


def is_image_file(file: Path) -> bool:
    return file.suffix.lower() in {".png", ".jpg", ".jpeg"}


def edit_image(image_path: Path, prompt: str) -> bytes:
    with image_path.open("rb") as f:
        response = openai.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="low",
        )
    result_b64 = response.data[0].b64_json
    return base64.b64decode(result_b64)


def main(input_dir: Path):
    concept_name = input_dir.name + "_imputation"
    image_data = []
    image_files = [f for f in input_dir.glob("*") if is_image_file(f)]

    for i, file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {file.name}")
        try:
            edited_image = edit_image(file, prompt)
            image_data.append(edited_image)
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

        if i < len(image_files) - 1:
            print("Waiting 12 seconds to respect rate limits...")
            time.sleep(12)  # ~5 requests per minute

    LOGGER.save_images(image_data, concept_name=concept_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove concepts from images.")
    parser.add_argument(
        "input_dir", type=Path, help="Path to the input directory of concept images"
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise ValueError(
            f"Input path {args.input_dir.resolve()} is not a valid directory."
        )

    main(args.input_dir)
