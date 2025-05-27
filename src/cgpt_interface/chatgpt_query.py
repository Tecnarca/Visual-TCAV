import base64
import time

import openai
from src.loggers import LOGGER

# Prompts
positive_prompt = """
        An image depicting chequered texture seen from different angles that can belong to different materials or objects.
    """

if __name__ == "__main__":
    concept_name = "chequered_cgpt"
    iters = 1

    for i in range(iters):

        print(f"Generating batch {i}")

        response = openai.images.generate(
            model="gpt-image-1",
            prompt=positive_prompt,
            n=5,
            size="1024x1024",
            output_format="png",
            quality="low",
        )

        image_data = [base64.b64decode(response.data[i].b64_json) for i in range(n)]
        LOGGER.save_images(image_data, concept_name=concept_name)
        print("Batch saved, going to sleep")
        time.sleep(max(6, int(60 / iters))+1)

    print("All images generated and saved.")
