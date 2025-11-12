# ROS2 Perception - The Construct Sim

## Tooling: OpenCV, PCL, YOLO

## Unit 3: Visualization
Exploring RViz2 for visualization of sensor message types

In the first exercise, we visualize the 2D Lidar data as laser scan msgs

![Alt text](readme/unit2-laser-scan.gif "Gif of laser scan msg in RViz2")

In the second exercise, we visualize the camera data as Image msgs

![Alt text](readme/unit2-images.png "Image of image msg in RViz2")
![Alt text](readme/unit2-images2.gif "Gif of image msg in RViz2")

In the final exercise, we append markers to visualize 3D lidar data in RViz2

![Alt text](readme/unit2-3d-markers.png "Gif of markers in RViz2")

## Unit 3: Image Processing
Exploring the use of open-Computer-Vision (openCV) for robotics use cases

This section makes extensive use of the cv_bridge to convert cv::Mat matrices of images from OpenCV into sensor_msgs.msg.Image data for ROS2

![Alt text](readme/unit3-cv-bridge.png "Image of cv_bridge openCV architectural interface with ROS")

In the first exercise, the robot rotates heading until it finds an orange blob, and it navigates to it

![Alt text](readme/unit3-blob-tracker.gif "Gif of robot rotating to orange blob and navigating to door using it")

The HSV min-max ranges for the orange blob were identified using a ranging tool provided

![Alt text](readme/unit3-range-finder.gif "Gif of HSV range finder used to identify range for orange blob")

In the second exercise, a line follower is created by analyzing and manipulating yaw to track the centroid of the yellow blob (line) on the ground

![Alt text](readme/unit3-line-follower.gif "Gif of robot tracking and following yellow lines on ground")

In the third exercise, the line follower is improved by adding functionality to handle scenarios of multiple centroids by prioritizing an appropriate centroid based on continuing on the rotational path

![Alt text](readme/unit3-line-follower-optimized.gif "Gif of improved robot tracking and following yellow lines on ground")

In the final exercise, a ROS subscriber is appended to the line follower to read a color choice and navigate the robot to the corresponding door

![Alt text](readme/unit3-door-follower.gif "Gif of robot being provided a color marker of which door to stop at from the yellow track")

## Unit 4: Point Cloud Processing
Exploring the use of point cloud library (PCL) for robotics use cases

In the first exercise, a ROS node using PCL detects the surface of the bench and isolates the point cloud detections defining the surface

![Alt text](readme/unit4-surface-detection.gif "Gif of detection and isolation of bench surface in point cloud")

In the second exercise, a ROS node using PCL detects the bench and a cube over it, and isolates the point cloud detections for each as indexed objects with poses 

![Alt text](readme/unit4-object-detection.gif "Gif of isolating point cloud of objects detected (surface and box)")

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