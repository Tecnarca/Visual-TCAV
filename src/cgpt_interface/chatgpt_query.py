import base64
import time

import openai
from src.loggers import save_images

# Prompts
positive_prompt = """
    A realistic image of a dotted, spotted or stained texture.
    Can belong to anything.
    Dots can have different shapes and be imperfect.
    I want the dots to be seen from different angles, to have different styles and colors.
    """

if __name__ == "__main__":
    for i in range(10):
        n = 5
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
        save_images(image_data, concept_name=f"dotted_{i}")
        print("Batch saved, going to sleep")
        time.sleep(60)

    print("All images generated and saved.")
