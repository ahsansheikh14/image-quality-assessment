"""
Synthetic Data Generator for Image Quality Assessment.

Takes clean images from Caltech101 and applies 9 types of distortions
to create labeled training data. Each distorted image is saved with a
label indicating its quality issue type.

Distortion Types:
    - blur: Gaussian blur simulating out-of-focus camera
    - darkness: Reduced brightness simulating poor lighting
    - overexposure: Increased brightness simulating washed-out photos
    - low_resolution: Pixelated images simulating low-quality camera
    - glare: White patches simulating light reflections
    - noise: Random pixel noise simulating sensor noise
    - motion_artifacts: Directional streak blur simulating camera shake
    - occlusion: Random blocks covering parts of the image
    - poor_framing: Off-center cropping simulating bad composition

Output:
    - data/distorted/ folder with all distorted + clean images
    - data/distorted/labels.csv mapping each filename to its issue type
"""

import os
import random
import csv
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import torchvision
from scipy.ndimage import convolve


def apply_blur(image):
    """Applies Gaussian blur to simulate out-of-focus camera."""
    radius = random.uniform(3, 7)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_darkness(image):
    """Reduces image brightness to simulate poor lighting conditions."""
    factor = random.uniform(0.15, 0.35)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_overexposure(image):
    """Increases image brightness to simulate overexposed photos."""
    factor = random.uniform(2.5, 4.0)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_low_resolution(image):
    """Creates a pixelated image by downscaling then upscaling with nearest-neighbor interpolation."""
    original_size = image.size
    tiny_size = random.randint(24, 48)
    small = image.resize((tiny_size, tiny_size))
    return small.resize(original_size, Image.NEAREST)


def apply_glare(image):
    """Adds white circular patches simulating light reflection hotspots."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    num_spots = random.randint(2, 4)
    for _ in range(num_spots):
        spot_size = random.randint(30, 80)
        x = random.randint(0, max(1, width - spot_size))
        y = random.randint(0, max(1, height - spot_size))
        draw.ellipse([x, y, x + spot_size, y + spot_size], fill=(255, 255, 255))

    return img_copy


def apply_noise(image):
    """Adds random Gaussian noise to simulate camera sensor grain."""
    img_array = np.array(image).astype(np.float32)
    noise_level = random.uniform(25, 50)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_motion_artifacts(image):
    """Applies directional 1D convolution kernel simulating camera shake."""
    img_array = np.array(image).astype(np.float32)
    kernel_size = random.randint(15, 25)

    kernel = np.zeros((kernel_size, kernel_size))
    direction = random.choice(["horizontal", "vertical", "diagonal"])

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)

    kernel = kernel / kernel.sum()

    blurred_channels = []
    for c in range(3):
        blurred_channel = convolve(img_array[:, :, c], kernel)
        blurred_channels.append(blurred_channel)

    result = np.stack(blurred_channels, axis=2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def apply_occlusion(image):
    """Draws opaque geometric patches to simulate view obstruction."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    num_blocks = random.randint(2, 4)
    for _ in range(num_blocks):
        block_w = random.randint(int(width * 0.15), int(width * 0.30))
        block_h = random.randint(int(height * 0.15), int(height * 0.30))
        x = random.randint(0, max(1, width - block_w))
        y = random.randint(0, max(1, height - block_h))

        color = (
            random.randint(0, 50),
            random.randint(0, 50),
            random.randint(0, 50),
        )
        draw.rectangle([x, y, x + block_w, y + block_h], fill=color)

    return img_copy


def apply_poor_framing(image):
    """Crops an off-center region and resizes back to simulate poor subject framing."""
    width, height = image.size
    crop_ratio = random.uniform(0.4, 0.6)
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)

    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    if corner == "top_left":
        box = (0, 0, crop_w, crop_h)
    elif corner == "top_right":
        box = (width - crop_w, 0, width, crop_h)
    elif corner == "bottom_left":
        box = (0, height - crop_h, crop_w, height)
    else:
        box = (width - crop_w, height - crop_h, width, height)

    cropped = image.crop(box)
    return cropped.resize((width, height))


DISTORTIONS = {
    "blur": apply_blur,
    "darkness": apply_darkness,
    "overexposure": apply_overexposure,
    "low_resolution": apply_low_resolution,
    "glare": apply_glare,
    "noise": apply_noise,
    "motion_artifacts": apply_motion_artifacts,
    "occlusion": apply_occlusion,
    "poor_framing": apply_poor_framing,
}

LABELS = list(DISTORTIONS.keys()) + ["clean"]


def generate_dataset(num_source_images=500, output_dir="data/distorted", clean_dir="data/clean"):
    """
    Main dataset generator routine.
    """
    print("Loading Caltech101 dataset...")
    search_engine_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "image-search-engine", "data"
    )
    
    if os.path.exists(os.path.join(search_engine_data, "caltech101")):
        dataset = torchvision.datasets.Caltech101(root=search_engine_data, download=False)
    else:
        dataset = torchvision.datasets.Caltech101(root="./data", download=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "labels.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label"])

    total_generated = 0
    num_source_images = min(num_source_images, len(dataset))
    num_versions = len(DISTORTIONS) + 1

    print(f"Generating synthetic distortions for {num_source_images} source images...")

    for i in range(num_source_images):
        img, _ = dataset[i]
        img = img.convert("RGB")

        clean_filename = f"clean_{i:04d}.jpg"
        img.save(os.path.join(output_dir, clean_filename))
        writer.writerow([clean_filename, "clean"])
        total_generated += 1

        img.save(os.path.join(clean_dir, f"source_{i:04d}.jpg"))

        for distortion_name, distortion_fn in DISTORTIONS.items():
            distorted_img = distortion_fn(img)
            distorted_filename = f"{distortion_name}_{i:04d}.jpg"
            distorted_img.save(os.path.join(output_dir, distorted_filename))
            writer.writerow([distorted_filename, distortion_name])
            total_generated += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_source_images} source images ({total_generated} total images created)...")

    csv_file.close()
    print(f"Dataset generation complete. Total images: {total_generated}")
    print(f"Labels CSV path: {csv_path}")


if __name__ == "__main__":
    generate_dataset(num_source_images=500)
