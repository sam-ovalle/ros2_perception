# ROS2 Perception - The Construct Sim

## Tooling: OpenCV, PCL, YOLO

## Unit 6: YOLO
Exploring the functionality of YOLO through some use cases

In the first exercise, a node running YOLOv8 detects and navigates to a banana and an apple it classified on a countertop

![Alt text](readme/unit6-object-detection.gif "Gif of YOLOv8 detecting and navigating to fruits")

In the second exercise, a node running YOLOv8 detects a sitting person and navigates the robot to them by centering them in frame (through heading of robot)

![Alt text](readme/unit6-navigating-to-person.gif "Gif of YOLOv8 detecting and navigating to a sitting woman")

In the third exercise, a node running YOLOv8 detects a standing person and defines a wireframe posture of their connected joints (a pose estimation)

![Alt text](readme/unit6-pose-estimation.png "Image of YOLOv8 estimating the pose of a standing woman")

In this final exercise, a node running YOLOv8 identifies classes of fruit and generates segmentation masks

![Alt text](readme/unit6-segmentation1.gif "Gif of YOLOv8 identifying fruit on a countertop")
![Alt text](readme/unit6-segmentation2.png "Gif of YOLOv8 generating the segmentation masks for detected fruit")

## Unit 7: Final Project
Putting it all together to navigate shelving and count classes of inventory

In this capstone, I navigate the robot to marked locations with stocked inventory to be counted

![Alt text](readme/unit7-final-project.png "Image of stocked shelves for final project")

The robot can tilt its head to use a YOLOv8 model for classification of objects and counting of different classes

![Alt text](readme/unit7-inventory-management.gif "Gif of inventory counting functionality")