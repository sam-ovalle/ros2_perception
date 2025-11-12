# ROS2 Perception - The Construct Sim

## Tooling: OpenCV, PCL, YOLO

## Unit 5: Human Robot Interaction
Exploring the use of perception to interface a robot with humans in its environment

Throughout these exercises, Haar cascade XML files are used as the OpenCV provided pre-trained classifiers 

![Alt text](readme/unit5-opencv-haarcascades.png "Image of opensource classifier models from OpenCV")

In this first exercise, a ROS node detects the people's faces and eyes

![Alt text](readme/unit5-face-detection.gif "Gif of detecting a person's face")
![Alt text](readme/unit5-eye-detection.gif "Gif of detecting a person's face and eyes")

In this second exercise, a ROS node detects and recognizes a persons face as opposed to a mona lisa painting face

![Alt text](readme/unit5-face-recognition.gif "Gif of detecting a person's face versus the Mona Lisa painting's face")

In this final exercise, a ROS node identifies and follows a standing person, rotating the robot's heading to face them

![Alt text](readme/unit5-human-tracking.gif "Gif of tracking a human with Deepmind Robot")

## Unit 6: YOLO
Exploring the functionality of YOLO through some use cases

![Alt text](readme/unit6-YOLO.png "Capabilities of You Only Look Once (YOLO)")

Throughout this coursework, I strictly use the YOLOv8n model trained on the COCO dataset

![Alt text](readme/unit6-YOLOv8.png "Comparison table of YOLOv8 models including model used")

In the first exercise, a node running YOLOv8n detects and navigates to a banana and an apple it classified on a countertop

![Alt text](readme/unit6-object-detection.gif "Gif of YOLOv8n detecting and navigating to fruits")

In the second exercise, a node running YOLOv8n detects a sitting person and navigates the robot to them by centering them in frame (through heading of robot)

![Alt text](readme/unit6-navigating-to-person.gif "Gif of YOLOv8n detecting and navigating to a sitting woman")

In the third exercise, a node running YOLOv8n detects a standing person and defines a wireframe posture of their connected joints (a pose estimation)

![Alt text](readme/unit6-pose-estimation.png "Image of YOLOv8n estimating the pose of a standing woman")

In this final exercise, a node running YOLOv8n identifies classes of fruit and generates segmentation masks

![Alt text](readme/unit6-segmentation1.gif "Gif of YOLOv8n identifying fruit on a countertop")
![Alt text](readme/unit6-segmentation2.png "Gif of YOLOv8n generating the segmentation masks for detected fruit")

## Unit 7: Final Project
Putting it all together to navigate shelving and count classes of inventory

In this capstone, I navigate the robot to marked locations with stocked inventory to be counted

![Alt text](readme/unit7-final-project.png "Image of stocked shelves for final project")

The robot can tilt its head to use a YOLOv8n model for classification of objects and counting of different classes

![Alt text](readme/unit7-inventory-management.gif "Gif of inventory counting functionality")