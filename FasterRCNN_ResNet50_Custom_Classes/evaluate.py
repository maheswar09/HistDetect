import os
import time
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data.dataset import random_split
from torch.utils.data import Subset
from collections import defaultdict
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class_to_idx = {'Bra': 0, 'Building': 1, 'Camera': 2, 'Cap': 3, 'CigBox': 4, 'Jewelry': 5, 'Radio': 6, 'Shoe': 7, 'Spectacle': 8}

class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform():
    transforms = [ToTensor()]
    return ComposeTransforms(transforms)

def get_model(num_classes):
    # Load a pre-trained model for fine-tuning
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=len(class_to_idx) + 1)  # Include background as a class
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Assuming model and test_loader have been initialized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_predictions(image, boxes, labels, scores, class_to_idx, threshold=0.5):
    """
    image: PIL image
    boxes: Tensor of dimensions [N, 4] where N is the number of boxes, and each box is (xmin, ymin, xmax, ymax)
    labels: Tensor of dimensions [N] where N is the number of labels
    scores: Tensor of dimensions [N] where N is the number of scores
    class_to_idx: Dictionary mapping class names to indices
    threshold: Score threshold for displaying bounding boxes
    """
    # Inverse mapping from indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Adding bounding boxes with labels and scores
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f"{idx_to_class[label.item()]}: {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

from torchvision.transforms import Compose, ToTensor, Normalize
def get_inference_transform():
    return Compose([
        ToTensor()
    ])


def save_predictions(image, boxes, labels, scores, class_to_idx, output_path, threshold=0.5):
    # Convert index to class
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Loop through each detected object
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            # Annotate class and score
            label_text = f"{idx_to_class[label.item()]}: {score:.2f}"
            ax.text(xmin, ymin, label_text, color='white', fontsize=8, verticalalignment='top', bbox=dict(facecolor='red', alpha=0.5))

    # Remove axis
    plt.axis('off')
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

import os

def save_annotations(boxes, labels, scores, class_to_idx, output_path_txt, threshold=0.5):
    # Convert index to class
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Prepare to write to file
    with open(output_path_txt, 'w') as file:
        # Loop through each detected object
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                # Format: xmin, ymin, xmax, ymax, class, score
                file.write(f"{box[0]}, {box[1]}, {box[2]}, {box[3]}, {idx_to_class[label.item()]}, {score:.2f}\n")

def evaluate_and_save(image_path, model, device, class_to_idx, output_dir):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image for inference
    transform = get_inference_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract data from the model output
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # Prepare output path
    base_filename = os.path.basename(image_path)
    base_filename = base_filename[:-4]  # Remove the .jpg extension
    output_path_img = os.path.join(output_dir, base_filename + '.png')
    output_path_txt = os.path.join(output_dir, base_filename + '_annotations.txt')

    # Call functions to save predictions and annotations
    save_predictions(image, boxes.cpu(), labels.cpu(), scores.cpu(), class_to_idx, output_path_img)
    save_annotations(boxes.cpu(), labels.cpu(), scores.cpu(), class_to_idx, output_path_txt)

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_path = 'test'
files = os.listdir(test_path)
for file in files:
    if file.endswith('.jpg') or file.endswith('.png'):
        evaluate_and_save(os.path.join(test_path,file), model, device, class_to_idx, 'output')