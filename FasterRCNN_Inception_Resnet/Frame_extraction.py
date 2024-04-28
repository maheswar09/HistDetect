import cv2
import os
from PIL import Image, ImageDraw, ImageFont


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

# # Example usage
# video_path = r"C:\Users\ravin\Downloads\Project\VideoData\k643b705c-30000157781315_009_mezzCrop.mp4"
# output_folder = r"C:\Users\ravin\Downloads\Project\framesForVideos\k643b705c-30000157781315_009_mezzCrop"
# n = get_video_fps(video_path)  # Change this based on your video's frame rate and required detail

# extract_frames(video_path, output_folder, n)


def process_video_folder(input_folder, output_base_folder, frame_extraction_rate_func):
    """
    Processes each video in the input folder, extracting frames according to the specified frame rate.
    
    Parameters:
    input_folder (str): Folder containing the video files.
    output_base_folder (str): Base output folder where extracted frames for each video will be saved.
    frame_extraction_rate_func (function): Function to determine the frame extraction rate (n).
    """
    # Ensure the output base folder exists
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    # Iterate over all files in the input folder
    for file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, file)
        if os.path.isfile(video_path) and video_path.endswith(('.mp4', '.avi', '.mov')):
            # Create a dedicated folder for this video's frames
            output_folder = os.path.join(output_base_folder, os.path.splitext(os.path.basename(video_path))[0])
            n = frame_extraction_rate_func(video_path)
            extract_frames(video_path, output_folder, n)



# Example usage
input_folder = r"o2"
output_base_folder = r"o2"
process_video_folder(input_folder, output_base_folder, get_video_fps)