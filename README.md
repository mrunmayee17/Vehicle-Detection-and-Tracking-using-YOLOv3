# Autonomous_Vehicle_Detection and Classification using OpenCV and YOLO

This project demonstrates vehicle counting and classification using OpenCV and the YOLOv3(You Only Look Once) object detection model. It can analyze both real-time video streams to count and classify vehicles. 

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- YOLOv3 model configuration and weights files(YOLOv3-320 cfg and weights) (Download from [YOLO website](https://pjreddie.com/darknet/yolo/))
- 'coco.names' file containing the class names

<img width="272" alt="Screenshot 2023-04-27 at 12 56 56 AM" src="https://github.com/mrunmayee17/Autonomous_Vehicle_Detection/assets/48186569/cc193479-cfd3-4ef4-83b0-4e5cebb265be">
<img width="274" alt="Screenshot 2023-09-15 at 6 47 36 AM" src="https://github.com/mrunmayee17/Autonomous_Vehicle_Detection-and-Classification-using-OpenCV-and-YOLO/assets/48186569/606d6cd6-819d-459d-b5d0-f9338d1007c5">


### Real-time Vehicle Counting and Classification

1. Place the video file in the video_file folder or update the video file path in the code.
2. Ensure you have the required model files and 'coco.names' file.
3. Run the `realTime()` function in the provided Python script.
4. Press 'q' to exit the real-time analysis.

## Project Structure

- `main`: The main Python script for vehicle counting and classification.
- Utility:
    - tracker.py: The Euclidean Distance Tracker module for object tracking.
    - counting_vehicles.py: Function for counting vehicles
    - detected_objects.py: Function for finding the detected objects from the network output
    - find_center.py: finding center of bounding boxes
    
- model_config
  - `yolov3-320.cfg` and `yolov3-320.weights`: YOLOv3 model configuration and weights files.
- video_file
  - `video.mp4`: Sample video file for real-time analysis.
- coco_class_index
  - `coco.names`: File containing COCO dataset class names.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
