import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from geometry_msgs.msg import Twist

class HumanTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('human_tracker')
        self.subscription = self.create_subscription(
            Image,
            '/deepmind_robot1/deepmind_robot1_camera/image_raw',
            self.image_callback,
            10)
        self.human_publisher = self.create_publisher(
            Image,
            '/detected_human',
            10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # Initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def image_callback(self, msg: None) -> None:
        """ Process image received from camera to detect humans and track them by adjusting angular velocity of robot"""
        # Convert the ROS2 image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detect humans
        (rects, _) = self.hog.detectMultiScale(cv_image, winStride=(8, 8),
                                               padding=(8, 8), scale=1.05)

        # Convert rects to the format expected by imutils' non_max_suppression
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        # Apply non-maxima suppression from imutils
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        self.get_logger().info(f'Detected {len(pick)} humans')

        # Draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(cv_image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Convert the OpenCV image back to a ROS2 message and publish it
        image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.human_publisher.publish(image_msg)

        if len(pick) > 0:
            (xA, yA, xB, yB) = pick[0]
            # Calculate the position of the human's based on bounding box position
            human_center_x = (xA + xB) / 2

            # Calculate the center of the image
            image_center_x = cv_image.shape[1] / 2

            # Calculate the offset of the human from the center of the image
            offset_x = human_center_x - image_center_x

            # Determine the angular velocity needed to align the robot with detected human
            gain = 0.001 
            angular_velocity = -offset_x * gain
            self.get_logger().info(f'Computed angular velocity: {angular_velocity}')

            # Create a Twist message and populate it
            twist_msg = Twist()
            twist_msg.angular.z = angular_velocity

            # Publish the Twist message to adjust the robot's angular velocity
            self.cmd_vel_publisher.publish(twist_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    human_tracker_node = HumanTrackerNode()
    rclpy.spin(human_tracker_node)
    # Explicitly destroy the node
    human_tracker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()