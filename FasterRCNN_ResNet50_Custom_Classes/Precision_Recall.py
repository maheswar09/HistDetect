import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import itertools


class_to_idx = {'Bra': 0, 'Building': 1, 'Camera': 2, 'Cap': 3, 'CigBox': 4, 'Jewelry': 5, 'Radio': 6, 'Shoe': 7, 'Spectacle': 8}


def load_ground_truth(file_path, class_to_idx):
    labels, boxes = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            labels.append(class_to_idx[parts[0]])
            # Box coordinates are expected to be [xmin, ymin, xmax, ymax]
            boxes.append(list(map(float, parts[1:])))
    return {'labels': torch.tensor(labels), 'boxes': torch.tensor(boxes)}

def predict(model, device, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    outputs = outputs[0]
    # Process outputs to remove unnecessary data if needed
    # Example: apply a threshold to remove low-confidence predictions
    threshold = 0.5
    keep = outputs['scores'] > threshold
    final_outputs = {
        'boxes': outputs['boxes'][keep],
        'labels': outputs['labels'][keep],
        'scores': outputs['scores'][keep]
    }

    return final_outputs

def calculate_iou(box_a, box_b):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """ Match predictions to ground truth using an IoU threshold."""
    matches = []
    used = set()

    for i, (p_box, p_label, p_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        match_found = False
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in used:
                continue
            if p_label == gt_label and calculate_iou(p_box, gt_box) >= iou_threshold:
                matches.append((p_score, 1))
                used.add(j)
                match_found = True
                break
        if not match_found:
            matches.append((p_score, 0))
    return matches

def evaluate_model(test_folder, model, device, class_to_idx, num_classes):
    model.eval()
    scores = []
    true_positives = []
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for filename in os.listdir(test_folder):

        if filename.endswith('.png'):
            image_path = os.path.join(test_folder, filename)
            annotation_path = image_path.replace('.png', '.txt')

            # Load ground truth
            gt_data = load_ground_truth(annotation_path, class_to_idx=class_to_idx)

            # Predict
            outputs = predict(model, device, image_path)
            matched = match_predictions(gt_data['boxes'], gt_data['labels'], outputs['boxes'], outputs['labels'], outputs['scores'])

            # Accumulate scores and binary labels for PR curve
            for score, is_true_positive in matched:
                scores.append(score)
                true_positives.append(is_true_positive)  # 1 for TP, 0 for FP
            
            for gt_label in gt_data['labels']:
                if gt_label in matched:
                    pred_label = outputs['labels'][matched[gt_label]]
                    conf_matrix[gt_label, pred_label] += 1
                else:
                    conf_matrix[gt_label, num_classes-1] += 1  # Assume the last class index is for false negatives (or background)
            print(matched)
            for pred_label in outputs['labels']:
                if pred_label not in matched.values():
                    conf_matrix[num_classes-1, pred_label] += 1 

    precision, recall, _ = precision_recall_curve(true_positives, scores)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, list(class_to_idx.keys()) + ['False Pos/Neg'], rotation=45)
    plt.yticks(tick_marks, list(class_to_idx.keys()) + ['False Pos/Neg'])

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the confusion matrix plot as a PNG file
    plt.savefig('confusion_matrix.png')
    plt.close()
    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')  # Save the plot as a PNG file
    plt.close()

def get_model(num_classes):
    # Load a pre-trained model for fine-tuning
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=len(class_to_idx) + 1)  # Include background as a class
model.load_state_dict(torch.load('TrainedModels/model_first10_19Apr.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

evaluate_model('test',model,device, class_to_idx=class_to_idx, num_classes=len(class_to_idx) + 1)
