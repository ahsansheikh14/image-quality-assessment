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


# ============================================================
# DISTORTION FUNCTIONS
# Each function takes a PIL Image and returns a damaged version
# ============================================================

def apply_blur(image):
    """
    Applies Gaussian blur to simulate an out-of-focus camera.
    
    How it works:
    - GaussianBlur replaces each pixel with a weighted average of its neighbors.
    - radius controls the blur strength (higher = more blurry).
    - We randomize the radius (3 to 7) so the model sees varying blur levels.
    """
    radius = random.uniform(3, 7)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_darkness(image):
    """
    Reduces image brightness to simulate poor lighting conditions.
    
    How it works:
    - ImageEnhance.Brightness creates a brightness modifier for the image.
    - .enhance(factor) multiplies every pixel's brightness by `factor`.
    - factor=1.0 means no change, factor=0.0 means completely black.
    - We use 0.15 to 0.35 to create visibly dark but not pitch-black images.
    """
    factor = random.uniform(0.15, 0.35)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_overexposure(image):
    """
    Increases image brightness to simulate overexposed / washed-out photos.
    
    How it works:
    - factor > 1.0 brightens the image.
    - At 2.5-4.0x, pixel values get pushed toward 255 (white), losing detail.
    - This simulates taking a photo with too much light or wrong camera settings.
    """
    factor = random.uniform(2.5, 4.0)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_low_resolution(image):
    """
    Creates a pixelated / blocky image by shrinking then re-enlarging.
    
    How it works:
    - Step 1: Shrink image to a tiny size (24x24 to 48x48 pixels).
      This permanently destroys fine visual details.
    - Step 2: Stretch it back to original size using NEAREST interpolation.
      NEAREST keeps the blocky pixelated look (no smoothing).
    - The result looks like a low-quality camera or heavily compressed image.
    """
    original_size = image.size
    tiny_size = random.randint(24, 48)
    small = image.resize((tiny_size, tiny_size))
    return small.resize(original_size, Image.NEAREST)


def apply_glare(image):
    """
    Adds white circular patches to simulate light glare / reflections.
    
    How it works:
    - Creates a copy of the image and draws white semi-transparent ellipses.
    - Random position and size to simulate natural glare spots.
    - 2-4 glare spots are added per image for realism.
    """
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
    """
    Adds random Gaussian noise to simulate camera sensor noise.
    
    How it works:
    - Convert PIL image to a NumPy array of pixel values (0-255).
    - Generate random noise from a normal distribution (mean=0, std=25-50).
    - Add noise to every pixel: noisy_pixel = original_pixel + random_noise.
    - np.clip ensures values stay in valid 0-255 range.
    - Convert back to PIL Image.
    """
    img_array = np.array(image).astype(np.float32)
    noise_level = random.uniform(25, 50)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_motion_artifacts(image):
    """
    Applies directional motion blur to simulate camera shake or subject movement.
    
    How it works:
    - A motion blur kernel is a 1D line of equal weights in one direction.
    - Instead of blurring equally in all directions (Gaussian), we blur
      ONLY along a single axis (horizontal, vertical, or diagonal).
    - This creates the characteristic "streaked" look of a moving camera.
    
    Implementation:
    - Create a NumPy kernel (a small matrix of numbers) shaped as a line.
    - The kernel size (15-25 pixels) controls streak length.
    - We randomly pick horizontal or vertical direction.
    - PIL's ImageFilter.Kernel applies this custom convolution to the image.
    """
    img_array = np.array(image).astype(np.float32)
    kernel_size = random.randint(15, 25)

    # Create a 1D motion blur kernel
    # Horizontal motion: a single row of 1s
    # Vertical motion: a single column of 1s
    kernel = np.zeros((kernel_size, kernel_size))
    direction = random.choice(["horizontal", "vertical", "diagonal"])

    if direction == "horizontal":
        # Fill the middle row with equal weights → horizontal streak
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "vertical":
        # Fill the middle column with equal weights → vertical streak
        kernel[:, kernel_size // 2] = 1.0
    else:
        # Fill the diagonal with equal weights → diagonal streak
        np.fill_diagonal(kernel, 1.0)

    # Normalize kernel so pixel values don't explode (weights must sum to 1.0)
    kernel = kernel / kernel.sum()

    # Apply convolution: slide the kernel across every pixel in the image
    # Each output pixel = weighted average of its neighbors along the streak direction
    from scipy.ndimage import convolve
    blurred_channels = []
    for c in range(3):  # Process each RGB channel independently
        blurred_channel = convolve(img_array[:, :, c], kernel)
        blurred_channels.append(blurred_channel)

    result = np.stack(blurred_channels, axis=2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def apply_occlusion(image):
    """
    Draws random opaque rectangles over the image to simulate partial blocking.
    
    What is occlusion?
    - When something physically blocks part of the camera's view.
    - Examples: a hand in front of the lens, a sticker on the windshield,
      a post blocking part of a street view, a watermark.
    
    How we simulate it:
    - Draw 2-4 random black or dark-colored rectangles on the image.
    - Each rectangle covers roughly 10-20% of the image area.
    - The model learns that large opaque patches = occlusion.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    num_blocks = random.randint(2, 4)
    for _ in range(num_blocks):
        # Each block is 15-30% of image width/height
        block_w = random.randint(int(width * 0.15), int(width * 0.30))
        block_h = random.randint(int(height * 0.15), int(height * 0.30))
        x = random.randint(0, max(1, width - block_w))
        y = random.randint(0, max(1, height - block_h))

        # Random dark color (not pure black to add variety)
        color = (
            random.randint(0, 50),
            random.randint(0, 50),
            random.randint(0, 50),
        )
        draw.rectangle([x, y, x + block_w, y + block_h], fill=color)

    return img_copy


def apply_poor_framing(image):
    """
    Crops a random corner/edge of the image to simulate bad composition.
    
    What is poor framing?
    - The subject is not properly centered in the photo.
    - Part of the subject is cut off, or there's too much empty space.
    - Like taking a selfie where half your face is out of frame.
    
    How we simulate it:
    - Crop a random 40-60% region from a corner or edge of the image.
    - Resize the cropped region back to original dimensions.
    - Result: the image looks zoomed into an off-center area,
      with the main subject partially or fully missing.
    """
    width, height = image.size

    # Crop size: 40-60% of original (cutting out a lot of content)
    crop_ratio = random.uniform(0.4, 0.6)
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)

    # Random corner/edge position for the crop
    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])

    if corner == "top_left":
        box = (0, 0, crop_w, crop_h)
    elif corner == "top_right":
        box = (width - crop_w, 0, width, crop_h)
    elif corner == "bottom_left":
        box = (0, height - crop_h, crop_w, height)
    else:
        box = (width - crop_w, height - crop_h, width, height)

    # Crop the corner region and stretch it back to original size
    cropped = image.crop(box)
    return cropped.resize((width, height))


# ============================================================
# MAIN DATA GENERATION PIPELINE
# ============================================================

# Map of distortion names to their functions
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

# Quality issue label names (including "clean" for good images)
LABELS = list(DISTORTIONS.keys()) + ["clean"]


def generate_dataset(num_source_images=500, output_dir="data/distorted", clean_dir="data/clean"):
    """
    Main function that generates the full synthetic dataset.
    
    Args:
        num_source_images: How many clean Caltech101 images to use as source.
        output_dir: Where to save distorted images.
        clean_dir: Where to save original clean images.
    
    Process:
        1. Load num_source_images from Caltech101.
        2. For each clean image:
           - Save a clean copy (label: "clean")
           - Apply each of the 9 distortions and save (label: distortion type)
        3. Write labels.csv mapping every filename to its quality issue.
    
    Total images generated = num_source_images * 10 (9 distortions + 1 clean)
    Example: 500 source images -> 5,000 labeled training images
    """
    # Download Caltech101 dataset (reuses existing download if present)
    print("Loading Caltech101 dataset...")
    caltech_data_root = os.path.join(os.path.dirname(output_dir), "..", "data")
    
    # Check if Caltech101 data exists in the image-search-engine project
    search_engine_data = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "image-search-engine", "data"
    )
    
    if os.path.exists(os.path.join(search_engine_data, "caltech101")):
        print(f"Found existing Caltech101 data at image-search-engine/data/")
        dataset = torchvision.datasets.Caltech101(root=search_engine_data, download=False)
    else:
        dataset = torchvision.datasets.Caltech101(root="./data", download=True)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # Prepare CSV label file
    csv_path = os.path.join(output_dir, "labels.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "label"])  # CSV header row

    total_generated = 0
    num_source_images = min(num_source_images, len(dataset))
    num_versions = len(DISTORTIONS) + 1  # 9 distortions + 1 clean

    print(f"Generating synthetic distortions for {num_source_images} source images...")
    print(f"Distortion types: {list(DISTORTIONS.keys())}")
    print(f"Total images to generate: {num_source_images * num_versions} ({len(DISTORTIONS)} distortions + 1 clean per source)")
    print()

    for i in range(num_source_images):
        img, _ = dataset[i]
        img = img.convert("RGB")

        # Save clean version
        clean_filename = f"clean_{i:04d}.jpg"
        img.save(os.path.join(output_dir, clean_filename))
        writer.writerow([clean_filename, "clean"])
        total_generated += 1

        # Also save to clean/ directory for reference
        img.save(os.path.join(clean_dir, f"source_{i:04d}.jpg"))

        # Apply each distortion and save
        for distortion_name, distortion_fn in DISTORTIONS.items():
            distorted_img = distortion_fn(img)
            distorted_filename = f"{distortion_name}_{i:04d}.jpg"
            distorted_img.save(os.path.join(output_dir, distorted_filename))
            writer.writerow([distorted_filename, distortion_name])
            total_generated += 1

        # Progress logging every 100 images
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_source_images} source images ({total_generated} total generated)")

    csv_file.close()
    print(f"\n Dataset generation complete!")
    print(f"   Total images: {total_generated}")
    print(f"   Labels file:  {csv_path}")
    print(f"   Labels:       {LABELS}")


if __name__ == "__main__":
    generate_dataset(num_source_images=500)
