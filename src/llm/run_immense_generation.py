import subprocess
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "concept_prompts.csv")

models = ["flux", "stable_diffusion", "gpt-image-1"]
num_images_per_prompt = 200
base_output_dir = Path("generated_images")

for model in models:
    for idx, row in df.iterrows():
        concept_name = row['Concept']
        positive_prompt = row['Positive']
        negative_prompt = row['Negative']

        output_dir = base_output_dir / model / f"{concept_name.replace(' ', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(Path(__file__).parent / "generative_pipelines.py"),
            "--prompt", positive_prompt,
            "--negative_prompt", negative_prompt,
            "--num_images", str(num_images_per_prompt),
            "--model", model,
            "--output_dir", str(output_dir)
        ]
        print(f"Generating {num_images_per_prompt} images for '{concept_name}' using {model}...")
        subprocess.run(cmd, check=True)

print("Image generation completed successfully.")

