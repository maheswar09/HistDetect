import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import cv2
from collections import Counter
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont

        
class_to_idx = {'Bra': 0, 'Building': 1, 'Camera': 2, 'Cap': 3, 'CigBox': 4, 'Jewelry': 5, 'Radio': 6, 'Shoe': 7, 'Spectacle': 8}
idx_to_class = {0 : 'Bra', 1: 'Building', 2:'Camera', 3: 'Cap', 4: 'CigBox', 5: 'Jewelry', 6: 'Radio', 7: 'Shoe', 8: 'Spectacle'}
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Adjust 'num_classes' to match the number of classes your model was trained on.
model = load_model(r'C:\Users\ravin\OneDrive\Desktop\CV_Final_Project\model.pth', num_classes=len(class_to_idx) + 1)

def get_unique_labels(frame, model, device):
    """
    Runs the model on a frame and returns unique labels detected.
    """
    # Convert frame to tensor and add batch dimension
    frame = F.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(frame)
    
    # Extract labels and convert to list of class names
    labels = prediction[0]['labels'].tolist()
    unique_labels = set(labels)  # Use a set to get unique labels
    return unique_labels

def get_video_fps(video_path):
    """
    Returns the frames per second (fps) of the video.

    Parameters:
    video_path (str): Path to the video file.
    """
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Get fps
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()  # Release the video after getting the information
    return fps

def extract_frames(video_path, output_folder, n):
    """
    Extracts every Nth frame from a video and saves them as images.
    
    Parameters:
    video_path (str): Path to the video file.
    output_folder (str): Folder where extracted images will be saved.
    n (int): Extract every Nth frame.
    """
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        # Read the next frame
        success, frame = video.read()
        if not success:
            break  # No more frames or error
        
        # Check if this frame is the Nth frame
        if frame_count % n == 0:
            # Save the frame
            output_path = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted_count += 1
        
        frame_count += 1
    
    # Release the video after processing
    video.release()
    #print(f"Done! Extracted {extracted_count} frames.")


def process_video_folder(input_folder, output_base_folder, frame_extraction_rate_func, model, device):

    # Ensure the output base folder exists
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    for file in os.listdir(input_folder):
        
        video_path = os.path.join(input_folder, file)
        
        if os.path.isfile(video_path) and video_path.endswith(('.mp4', '.avi', '.mov')):

            output_folder = os.path.join(output_base_folder, os.path.splitext(os.path.basename(video_path))[0])
            n = frame_extraction_rate_func(video_path)
            extract_frames(video_path, output_folder, n)

            # Create a Counter object to hold the frequency of each detected class
            class_counter = Counter()

            # Get the paths to all extracted frames
            frame_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
            
            for frame_path in frame_files:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                unique_labels = get_unique_labels(frame, model, device)
                class_counter.update(unique_labels)  # Update counter with detected classes
            
            # Select the top 10 most common classes
            top_classes = class_counter.most_common(10)
            
            # Save the results to a file
            results_path = os.path.join(output_folder, 'top_detected_classes.txt')
            with open(results_path, 'w') as file:
                for class_id, count in top_classes:
                    file.write(f'{idx_to_class[class_id]}: {count}\n')

            print(f'Processed video {video_path}, top classes saved to {results_path}')


# Set the device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example usage
input_folder = r"C:\Users\ravin\Downloads\Project\VideoData" #change input folder containing directory of videos
output_base_folder = r"C:\Users\ravin\Downloads\Project\framesForVideos" #change output folder

# Call the updated function
process_video_folder(input_folder, output_base_folder, get_video_fps, model, device)

