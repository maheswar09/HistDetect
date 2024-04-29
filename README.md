# HistDetect

## Project Description
This project utilizes TensorFlow and TensorFlow Hub to perform object detection on video files. It processes video files to detect objects using the Faster R-CNN model hosted on TensorFlow Hub and outputs detection results in text files.

## Prerequisites
- Python 3.6+
- pip (Python package installer)
- Virtual environment (recommended)

## Environment Setup

Clone the repository and navigate to your project directory:
```bash
git clone https://github.com/maheswar09/HistDetect.git
cd HistDetect
```

## Creating a Virtual Environment
Creating a virtual environment is recommended to manage dependencies.

**For Unix/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**For Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
## Installing Dependencies
Install all required packages using pip:
```bash
pip install -r requirements.txt
```
### Set Up TensorFlow Hub Cache
Update the TFHUB cache directory in the Python files before running them:

```python
os.environ['TFHUB_CACHE_DIR'] = '/N/u/naddank/BigRed200/26/tfhub_cache'  # Update the path for the cache file
os.makedirs('/N/u/naddank/BigRed200/26/tfhub_cache', exist_ok=True)  # Ensure the cache directory exists
```
### Main.py

**Description:**
This script takes video input and generates a text file listing detected objects along with their respective frequencies.

**Modifications Required:**

- Update the `video_path` to point to your video file:
  ```python
  video_path = 'vedio_files/xp68m415k-30000157759097_017_mezzCrop-high.mp4'  # Update this path to your video file
  output_base_folder = 'output_texts'  # Update this to your output folder path
  ```
### Running the Application

To run the application after configuring your paths in `Main.py`, execute the following command from your terminal:

```bash
python main.py
```

### Object_detection.py

**Description:**
This script takes a folder of frames extracted from a video and generates images with bounding boxes highlighting detected objects. It is important to run `Frame_extraction.py` before this script to ensure the frames are available for processing.

**Pre-requisites:**
Ensure that the frames have been extracted and are stored in a designated folder.

**Modifications Required:**
- Ensure the path to the folder containing the frames is correctly set in the script.

**Running the Script:**
Update the paths accordingly in the script to point to the folder containing the frames and the output directory where the processed images will be saved.

To run `Object_detection.py`, execute the following command from your terminal:
```bash
python object_detection.py
```
### Results from Main.py

After running `main.py`, the following objects were detected and their counts were recorded:

```yaml
Human face: 25
Clothing: 31
Person: 11
Man: 28
Door: 1
Window: 5
Tree: 2
Building: 2
Land vehicle: 7
Vehicle: 3
Footwear: 3
Train: 4
House: 1
Glasses: 9
Human nose: 2
Poster: 6
Picture frame: 1
Flower: 1
```
### Results from Object_detection.py

After running `object_detection.py`, the following images with detected objects and bounding boxes were generated:

![Image 1](FasterRCNN_Inception_Resnet/output_images/frame_0003.jpg)
![Image 2](FasterRCNN_Inception_Resnet/output_images/frame_0014.jpg)

These images show the detected objects with bounding boxes, illustrating the model's ability to identify and localize objects within frames extracted from the video.

