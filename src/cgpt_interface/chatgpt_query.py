import base64
import time

import openai
from src.loggers import LOGGER

# Prompts
positive_prompt = """
        An abstract image showcasing the texture of a waffle pattern, repeating across the surface with soft lighting and realistic shadows.
        No other objects or context — just the waffle texture filling the image. Make the picture varied, with different angles, textures and colors
    """

if __name__ == "__main__":
    n = 5
    concept_name = "waffled_cgpt"
    iters = 10
    calls_per_minute = 1

    for i in range(iters):

        print(f"Generating batch {i}")

        response = openai.images.generate(
            model="gpt-image-1",
            prompt=positive_prompt,
            n=n,
            size="1024x1024",
            output_format="png",
            quality="low",
        )

        image_data = [base64.b64decode(response.data[i].b64_json) for i in range(n)]
        LOGGER.save_images(image_data, concept_name=concept_name)
        print("Batch saved, going to sleep")
        time.sleep(60 / calls_per_minute)

    print("All images generated and saved.")
