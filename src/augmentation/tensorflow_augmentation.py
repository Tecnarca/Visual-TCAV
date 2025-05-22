from random import uniform

import tensorflow as tf
import argparse
from pathlib import Path
from scipy.ndimage import rotate as scipy_rotate
from tensorflow.keras.utils import img_to_array, array_to_img, load_img


def advanced_augment(image):
    # Random flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Convert to numpy for scipy rotation
    image_np = image.numpy()

    # Random rotation
    angle = uniform(-45, 45)  # degrees
    image_np = scipy_rotate(image_np, angle=angle, axes=(0, 1), reshape=False, mode='nearest')

    # Back to tensor
    image = tf.convert_to_tensor(image_np, dtype=tf.float32)

    # Color transformations
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.1)

    # Random channel shuffle
    if tf.random.uniform([]) > 0.5:
        image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.random.shuffle(image)
        image = tf.transpose(image, perm=[1, 2, 0])

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def augment_and_save(image_path, output_dir, num_augmented):
    img = load_img(image_path)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    for i in range(num_augmented):
        aug_img = advanced_augment(img_tensor)
        aug_img_uint8 = tf.image.convert_image_dtype(aug_img, dtype=tf.uint8)
        pil_img = array_to_img(aug_img_uint8)
        filename = f"{Path(image_path).stem}_aug{i}.jpg"
        pil_img.save(output_dir / filename)


def main(input_folder, num_augmented):
    input_folder = Path(input_folder)
    output_folder = input_folder.parent / f"{input_folder.name}_augmented"
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_folder.glob("*.[jp][pn]*g"))
    if not image_paths:
        print("No images found.")
        return

    print(f"Augmenting {len(image_paths)} images with {num_augmented} variations each...")

    for img_path in image_paths:
        augment_and_save(img_path, output_folder, num_augmented)

    print(f"Done. Augmented images saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorFlow-based image augmenter.")
    parser.add_argument("folder", help="Path to input image folder")
    parser.add_argument("--num", type=int, default=10, help="Number of augmented images per original")
    args = parser.parse_args()

    main(args.folder, args.num)
