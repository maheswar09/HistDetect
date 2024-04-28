import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TFHUB_CACHE_DIR'] = '/N/u/naddank/BigRed200/26/tfhub_cache'
os.makedirs('/N/u/naddank/BigRed200/26/tfhub_cache', exist_ok=True)

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def load_img(frame):
    img = tf.convert_to_tensor(frame)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img

def extract_and_detect_frames(video_path, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = get_video_fps(video_path)
    frame_count = 0
    detection_results = {}

    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % fps == 0:  # Process one frame per second
            image_tensor = load_img(frame)  # Convert the frame to a tensor directly
            detector_output = model(image_tensor)
            labels = detector_output['detection_class_entities'].numpy()
            scores = detector_output['detection_scores'].numpy()

            for label, score in zip(labels, scores):
                if score >= 0.3:
                    label_str = label.decode('utf-8')
                    if label_str in detection_results:
                        detection_results[label_str] += 1
                    else:
                        detection_results[label_str] = 1
        frame_count += 1

    video.release()
    return detection_results

def main(video_path, output_base_folder):
    os.makedirs(output_base_folder, exist_ok=True)
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    model = hub.load(module_handle).signatures['default']
    
    video_folder = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_folder, video_name)
    detection_results = extract_and_detect_frames(video_path, output_folder, model)

    output_file = os.path.join(output_folder, f"{video_name}_detection_results.txt")
    with open(output_file, 'w') as f:
        for label, count in detection_results.items():
            f.write(f"{label}: {count}\n")
    print(f"Detections saved to {output_file}")

# Example usage
video_path = 'vedio_files/xp68m415k-30000157759097_017_mezzCrop-high.mp4'  # Update this path to your video file
output_base_folder = 'output_texts'  # Update this to your output folder path
main(video_path, output_base_folder)
