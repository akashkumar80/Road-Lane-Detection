# Road Lane Detection System

## Overview

The Road Lane Detection System is a computer vision application that detects and highlights lane markings on roads in real-time. Using a combination of image processing techniques and deep learning, this project aims to enhance lane detection accuracy across various driving conditions.

## Features

- **Lane Detection**: Detects lane markings in images, videos, and live camera feeds.
- **Real-Time Processing**: Processes video streams in real-time.
- **Versatile Input**: Supports single images, video files, and live camera input.
- **Output Options**: Generates output files with detected lanes highlighted.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/road-lane-detection.git
   cd road-lane-detection
   ```
2. Install Required Libraries: Make sure you have Python installed, then install the required libraries using pip:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
## Usage
To run the lane detection system, you can process images, videos, or live camera feeds using the command line. Below are the command formats:

**Detect lanes in a single image**
```bash
python lane_detection.py --image input_image.jpg --output output_image.jpg
```
**Detect lanes in a video file:**
```bash
python lane_detection.py --video input_video.mp4 --output output_video.avi
```
**Use live camera input:**
```bash
python lane_detection.py --camera 0 --output output_live.avi
```

## Command Line Argument
<ul>
  <li>-i or --image: Specify the path to the input image file.</li>
  <li>-v or --video: Specify the path to the input video file.</li>
  <li>-o or --output: Specify the output file name or location (default is output).</li>
  <li>-c or --camera: Specify the camera parameter (default is 0 for the primary camera).</li>
</ul>

## Technology Used
<ul>
  <li>Python</li>
  <li>OenCV</li>
  <li>Numpy</li>
  <li>Matplotlib</li>
</ul>

## Code Structure
<ul>
  <li>**lane_detection.py:** Main script that implements lane detection using various functions.</li>
  <li>**region_of_interest():** Defines the area of interest for lane detection.</li>
  <li>**make_line_points():** Calculates line endpoints based on slope and intercept.</li>
  <li>**draw_line():** Draws detected lane lines on the image.</li>
  <li>**Detect_lane_frame():** Processes a single frame/image for lane detection.</li>
  <li>**Detect_Lane():** Handles video input for lane detection.</li>
  <li>**Detect_image():** Detects lanes in a single image file.</li>
  <li>**Detect_video():** Detects lanes in a video file.</li>
  <li>**Detect_live():** Detects lanes using live camera input.</li>
</ul>
