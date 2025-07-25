#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import PoseKeypoint, PoseResult
import numpy as np
from typing import List

class PoseEstimationNode(Node):
    def __init__(self) -> None:
        super().__init__('pose_estimation_node')
        # Load pre-trained yolov8n-pose model for pose estimation 
        self.model = YOLO('/home/user/ros2_ws/src/advanced_perception/data/yolov8n-pose.pt') 

        self.subscription = self.create_subscription(
            Image,
            '/deepmind_robot1/deepmind_robot1_camera/image_raw',
            self.image_callback,
            10)

        self.pose_publisher = self.create_publisher(PoseResult, "/yolov8_pose_results", 1)
        self.image_publisher = self.create_publisher(Image, "/pose_estimation_result", 1)
        self.bridge = CvBridge()

    def image_callback(self, data: Image) -> None:
        """ Performs the pose estimation by extracting the detected keypoints 
            and publishes the annotated image for visualization purposes.
            Additonally, it calculates the angle between specified keypoints for further analysis """
        
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.model(img)

        pose_result = PoseResult()
        pose_result.header.stamp = self.get_clock().now().to_msg()
        pose_result.header.frame_id = "pose_estimation"

        
        if results and results[0].keypoints:
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]  
            confidence = results[0].keypoints.conf.cpu().numpy()[0]  
            for idx, kpt in enumerate(keypoints):
                keypoint = PoseKeypoint()
                keypoint.id = idx
                keypoint.x = float(kpt[0])
                keypoint.y = float(kpt[1])
                keypoint.confidence = float(confidence[idx]) 
                pose_result.keypoints.append(keypoint)
        
            # E.g., Left shoulder (5), Left elbow (7), and Left wrist (9)
            self.calculate_angle(pose_result.keypoints, 5, 7, 9)
            self.pose_publisher.publish(pose_result)

            annotated_frame = results[0].plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
            self.image_publisher.publish(img_msg)

    def calculate_angle(self, keypoints: List[PoseKeypoint], index1: int, index2: int, index3: int) -> None:
        """ Calculate the angle in degrees between three keypoints """
        # Make sure we have enough keypoints
        if len(keypoints) >= 3:
            pt1 = [keypoints[index1].x, keypoints[index1].y]
            pt2 = [keypoints[index2].x, keypoints[index2].y]
            pt3 = [keypoints[index3].x, keypoints[index3].y]

            # Calculate the vectors
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)
            
            # Calculate the angle in radians
            angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            
            # Convert to degrees
            angle_deg = np.degrees(angle_rad)
            
            self.get_logger().info(f"The angle between the keypoints {index1}, {index2} and {index3} is equal to {angle_deg:.2f} degrees.")

        else:
            self.get_logger().info(f"Not enough keypoints to calculate the angle ...")


def main(args=None) -> None:
    rclpy.init(args=args)
    pose_estimation = PoseEstimationNode()
    rclpy.spin(pose_estimation)
    pose_estimation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()