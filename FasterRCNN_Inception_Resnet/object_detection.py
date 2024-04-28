import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Set the environment variable for TensorFlow Hub cache
os.environ['TFHUB_CACHE_DIR'] = '/N/u/naddank/BigRed200/26/tfhub_cache'
os.makedirs('/N/u/naddank/BigRed200/26/tfhub_cache', exist_ok=True)

# Load the model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
print("Loading model...")
detector = hub.load(module_handle).signatures['default']
print("Model loaded successfully.")

# Function to load and preprocess an image
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img

# Function to save image along with the bounding boxes and labels
def save_image(image, results, output_dir, filename, threshold=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = image[0]  # Remove the batch dimension
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    labels = results['detection_class_entities'].numpy()  # Convert tensor to numpy
    scores = results['detection_scores'].numpy()          # Convert tensor to numpy
    boxes = results['detection_boxes'].numpy()            # Convert tensor to numpy

    # Display bounding boxes and labels
    for label, score, box in zip(labels, scores, boxes):
        if score >= threshold:
            y1, x1, y2, x2 = box
            y1 *= image.shape[0]
            y2 *= image.shape[0]
            x1 *= image.shape[1]
            x2 *= image.shape[1]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            label_str = label.decode("utf-8")  # Decode bytes to string
            ax.text(x1, y1, f'{label_str} {score:.2f}', color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Image saved to {output_path}")

# Directory containing images
image_folder = 'o2/7m01c8269-30000157758990_032_mezzCrop-high'  # Update this path to your folder
output_folder = 'o3'  # Update this to where you want to save the output images

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(image_folder, filename)
        image_tensor = load_img(image_path)
        
        # Perform object detection
        detector_output = detector(image_tensor)
        
        # Save the results
        save_image(image_tensor, detector_output, output_folder, filename, threshold=0.3)
