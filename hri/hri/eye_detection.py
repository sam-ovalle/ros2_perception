import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ament_index_python.packages import get_package_share_directory

class EyeDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('eye_detector_node')
        self.subscription = self.create_subscription(
            Image,
            '/deepmind_robot1/deepmind_robot1_camera/image_raw',  
            self.image_callback,
            10)
        self.face_publisher = self.create_publisher(
            Image,
            '/detected_faces/faces',  
            10)
        self.eye_publisher = self.create_publisher(
            Image,
            '/detected_faces/eyes',  
            10)
        self.bridge = CvBridge()
        data_folder = os.path.join(get_package_share_directory('hri'), 'data')
        face_cascade_path = os.path.join(data_folder, 'haarcascade_frontalface_default.xml')
        eye_cascade_path = os.path.join(data_folder, 'haarcascade_eye.xml')
       
        if not os.path.isfile(face_cascade_path) or not os.path.isfile(eye_cascade_path):
            self.get_logger().error("Cascade xml file(s) not found")
            return

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

   
    def image_callback(self, msg: Image) -> None:
        """ Detects faces and eyes drawing blue rectangles around faces and green rectangles around eyes"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            # draw blue box surrounding detected face
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.face_publisher.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

        for (x, y, w, h) in faces:
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_color = cv_image[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                self.get_logger().info("Eyes detected")
            for (ex, ey, ew, eh) in eyes:
                # draw green boxes surrounding detected eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        self.eye_publisher.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))    

def main(args=None) -> None:
    rclpy.init(args=args)
    face_detector_node = EyeDetectorNode()
    rclpy.spin(face_detector_node)
    face_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()