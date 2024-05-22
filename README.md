# YOLOv3 Object Detection and Evaluation

This project demonstrates the use of YOLOv3 for object detection on an image and evaluates the performance using precision, recall, and average precision metrics. The project includes a helper function to calculate Intersection over Union (IoU) for evaluation purposes.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#discussion)
- [References](#references)

## Introduction

YOLOv3 (You Only Look Once, Version 3) is a state-of-the-art, real-time object detection system. This project loads a YOLOv3 model, performs object detection on a given image, and evaluates the detection performance by comparing the results against ground truth bounding boxes.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Pre-trained YOLOv3 weights and configuration files
- COCO class names file

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/yolov3-object-detection.git
    cd yolov3-object-detection
    ```

2. Install the required Python packages:

    ```sh
    pip install opencv-python numpy
    ```

3. Download YOLOv3 weights and configuration files from [YOLO website](https://pjreddie.com/darknet/yolo/). Place `yolov3.weights` and `yolov3.cfg` in the project directory.

4. Download the COCO class names file and place `coco.names` in the project directory.

## Usage

1. Place the image you want to test in the project directory. Update the image path in the script if necessary.

2. Update the `ground_truth_boxes` list in the script with the actual ground truth bounding box coordinates for evaluation.

3. Run the script:

    ```sh
    python yolov3_object_detection.py
    ```

4. The script will display the image with detected bounding boxes and print the evaluation results.

## Results

After running the script, the results section will display the precision, recall, and false positives/negatives. The detected bounding boxes will be shown on the image with labels and confidence scores.

**Output Image:**

<img width="577" alt="Screenshot 2024-05-22 at 7 13 55 PM" src="https://github.com/kulfi-12/Object-Detection-using-DNN/assets/128511001/8c661afd-a204-4bdd-af8f-968a5547a269">


## Discussion

This section provides an analysis of the detection performance. It discusses the number of false positives and false negatives, potential reasons for misdetections, and suggestions for improving the model's performance.

**Example Discussion Points:**
- Why certain objects might be missed (false negatives).
- Potential causes of false positives.
- Adjustments to confidence thresholds and IoU thresholds.

## References

- YOLOv3 Paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- OpenCV Documentation: [OpenCV](https://opencv.org/)
- COCO Dataset: [COCO](https://cocodataset.org/)

Feel free to modify the ground truth bounding boxes and detection thresholds according to your specific use case and dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
