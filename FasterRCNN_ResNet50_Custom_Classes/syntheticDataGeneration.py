import os
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
def load_images_from_folders(base_path):
    """Load images and organize them by class."""
    class_images = {}
    for class_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, class_folder)
        if os.path.isdir(folder_path):
            class_images[class_folder] = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    with Image.open(img_path) as img:
                        img = img.convert("RGBA")
                        class_images[class_folder].append((img, filename, class_folder))
    return class_images




def apply_transformations(img, max_size=(256, 256), scale_range=(0.7, 4.0)):
    """Apply random rotation, brightness adjustment, and resize to fixed dimensions."""
    
    # Rotate within a range of -45 to 45 degrees
    angle = random.randint(-45, 45)
    img = img.rotate(angle, expand=True)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    brightness_factor = random.uniform(0.7, 1.3)  # Random brightness factor between 0.7 and 1.3
    img = enhancer.enhance(brightness_factor)

    # Random scaling
    scale_factor = random.uniform(*scale_range)
    original_size = img.size
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Resize to fit output while maintaining aspect ratio
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    return img

def check_overlap(new_box, boxes):
    """Check if new_box overlaps with any in boxes."""
    for box in boxes:
        if (new_box[0] < box[2] and new_box[2] > box[0] and
            new_box[1] < box[3] and new_box[3] > box[1]):
            return True
    return False

def create_composite_image(class_images, output_size, max_images=20):
    """Create a composite image with one random non-overlapping image from each class."""
    background = Image.new('RGBA', output_size, (255, 255, 255, 255))
    noise = np.random.randint(0, 256, (output_size[1], output_size[0], 4), dtype=np.uint8)
    noise_image = Image.fromarray(noise, 'RGBA')
    background = Image.blend(background, noise_image, alpha=0.2)

    annotations = []
    occupied_areas = []
    image_count = 0

    all_images = [(img_data, class_name) for class_name, images in class_images.items() for img_data in images]
    random.shuffle(all_images)  # Shuffle to randomize the order of images

    for img_data, class_name in all_images:
        if image_count >= max_images:
            break  # Stop if we've reached the max number of images

        img, filename, label = img_data
        img = apply_transformations(img)  # Apply transformations including scaling

        placed = False
        attempts = 0
        while not placed and attempts < 50:
            max_x = background.width - img.width
            max_y = background.height - img.height
            if max_x > 0 and max_y > 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                new_box = [x, y, x + img.width, y + img.height]
                if not check_overlap(new_box, occupied_areas):
                    background.paste(img, (x, y), img)
                    occupied_areas.append(new_box)
                    annotations.append(f"{class_name} {x} {y} {x + img.width} {y + img.height}")
                    placed = True
            attempts += 1
            if not placed and attempts == 50:
                break  # Stop trying to place the current image
        image_count+=1

    return background.convert('RGB'), annotations

def create_dataset(base_folder, num_images):
    """Generate a dataset of composite images with balanced class representation."""
    class_images = load_images_from_folders(base_folder)
    output_folder = os.path.join(base_folder, '../output_dataset')
    os.makedirs(output_folder, exist_ok=True)

    images_per_class = {class_name: 0 for class_name in class_images}

    for i in range(num_images):
        composite_image, image_annotations = create_composite_image(class_images, (1024, 1024))
        composite_image.save(os.path.join(output_folder, f'composite_{i+1}.png'))
        with open(os.path.join(output_folder, f'composite_{i+1}.txt'), 'w') as file:
            file.write("\n".join(image_annotations))
        for annotation in image_annotations:
            class_label = annotation.split()[0]
            images_per_class[class_label] += 1

    print("Images per class used in the dataset:", images_per_class)

# Usage example:
create_dataset('archive', 400)  # Specify the path to your folders and the number of images