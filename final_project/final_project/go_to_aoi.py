import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time
from typing import Tuple, List

class GoToAOI(Node):
    def __init__(self) -> None:
        super().__init__('go_to_aoi')
        self.bridge = CvBridge()
        self.aoi_choice_sub = self.create_subscription(
            String,
            '/aoi_choice',
            self.aoi_callback,
            10
        )
        self.img_sub = self.create_subscription(Image, '/deepmind_robot1/deepmind_robot1_camera/image_raw', self.camera_callback, 10)
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.aoi_choice = None
        self.aoi_detected = False
        self.last_detection_time = None


    def aoi_callback(self, msg: String) -> None:
        """ Callback function for the 'aoi_choice' subscription
            It updates the area of interest (AOI) choice based on the incoming message """
        self.aoi_choice = msg.data

    def camera_callback(self, msg: Image) -> None:
        """ Process each incoming image from the camera. It converts the ROS image
        to an OpenCV format, applies color thresholding to detect specific colors based on the
        current AOI choice, and calculates navigation errors based on detected contours """
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().info(f"Error converting image: {e}")
            return

        # Define lower and upper bounds for purple, yellow, green, blue and red colors in HSV
        purple_bounds = ([120, 50, 50], [160, 255, 255])
        yellow_bounds = ([20, 100, 50], [35, 255, 255])
        green_bounds = ([45, 100, 50], [85, 255, 255])
        blue_bounds = ([90, 150, 50], [125, 255, 255])
        red_bounds = ([170, 150, 50], [180, 255, 255])
        # Define height and width of image
        height, width, _ = cv_image.shape

        # Crop images
        crop_img, crop_aoi_img = self.crop_image(cv_image, height, width)
        # Convert to HSV
        hsv, hsv_detect_aoi = self.convert_to_hsv(crop_img, crop_aoi_img)         

        # Detects purple contours
        res, purple_contours = self.detect_contours(hsv, purple_bounds[0], purple_bounds[1], crop_img)

        if self.aoi_choice == 'yellow':
            # Detects yellow contours
            res2, yellow_contours = self.detect_contours(hsv_detect_aoi, yellow_bounds[0], yellow_bounds[1], crop_aoi_img)
            # Checks if yellow aoi has been detected
            self.aoi_detected = self.search_aoi('yellow', yellow_contours, res2)

        if self.aoi_choice == 'green':
            # Detects green contours
            res2, green_contours = self.detect_contours(hsv_detect_aoi, green_bounds[0], green_bounds[1], crop_aoi_img)
            # Checks if green aoi has been detected
            self.aoi_detected = self.search_aoi('green', green_contours, res2)

        if self.aoi_choice == 'blue':
            # Detects blue contours
            res2, blue_contours = self.detect_contours(hsv_detect_aoi, blue_bounds[0], blue_bounds[1], crop_aoi_img)
            # Checks if blue aoi has been detected
            self.aoi_detected = self.search_aoi('blue', blue_contours, res2)

        if self.aoi_choice == 'red':
            # Detects red contours
            res2, red_contours = self.detect_contours(hsv_detect_aoi, red_bounds[0], red_bounds[1], crop_aoi_img)
            # Checks if red aoi has been detected
            self.aoi_detected = self.search_aoi('red', red_contours, res2)

        # Calculate centroids for each purple contours
        centres = []
        for i in range(len(purple_contours)):
            moments = cv2.moments(purple_contours[i])
            try:
                # append centroid coordinates for each contour
                centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
                # draw green circle to visualize detected centroids
                cv2.circle(res, centres[-1], 10, (0, 255, 0), -1)
            except ZeroDivisionError:
                pass

        # checks if arrived at the aoi destination
        if len(centres) < 1 and self.aoi_detected:
            self.go_to_aoi(hsv_detect_aoi, crop_aoi_img, width)
            return  

        # Ensure centres is not empty before attempting to find the max and min centroids
        if centres:
            most_right_centroid = max(centres, key=lambda x: x[0])
            most_left_centroid = min(centres, key=lambda x: x[0])
        else:
            self.get_logger().error("No purple contours detected. Cannot find centroids.")
            return  # Exit the callback if no centroids are found

        # Checks if the aoi choice color choice is valid
        self.is_valid_aoi_choice()
            
        # Calculate centroids coordinates
        if (self.is_valid_aoi_choice() and self.aoi_detected):
            cx, cy = self.get_centroid_coordinates(most_left_centroid, height, width)

        else:
            cx, cy = self.get_centroid_coordinates(most_right_centroid, height, width)
    
        # draw red circle to visualize selected centroid
        cv2.circle(res, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # Display
        cv2.imshow("RES", res)
        cv2.waitKey(1)

        # Calculate horizontal error between the centroid and the center of the image
        error_x = cx - width / 2

        # Publish velocities
        self.pub_velocities(error_x)


    def crop_image(self, cv_image: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Crop image, we create two cropped image
            One for the line follower and one to access the AOI """
        rows_to_watch = 20

        crop_img_start_row = height * 3 // 4
        crop_img_end_row = min(height, crop_img_start_row + rows_to_watch)
        crop_img = cv_image[crop_img_start_row:crop_img_end_row, 0:width]  

        crop_aoi_img_start_row = height // 3
        crop_aoi_img_end_row = min(height, height * 2 // 3)
        crop_aoi_img_start_col = width // 8
        crop_aoi_img_end_col = min(width, width // 2)
        crop_aoi_img = cv_image[crop_aoi_img_start_row:crop_aoi_img_end_row, crop_aoi_img_start_col:crop_aoi_img_end_col]  

        return crop_img, crop_aoi_img

    def convert_to_hsv(self, crop_img: np.ndarray, crop_aoi_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Converts from BGR to HSV for the two cropped image """
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        hsv_detect_aoi = cv2.cvtColor(crop_aoi_img, cv2.COLOR_BGR2HSV)
        return hsv, hsv_detect_aoi

    def detect_contours(self, hsv: np.ndarray, lower_bound: Tuple[int, int, int], upper_bound: Tuple[int, int, int], image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ Detects for contours based on HSV range values """
        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
        res = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        return res, contours

    def is_recently_detected(self) -> bool:
        """ Determines if aoi was recently detected """
        if self.last_detection_time is not None:
            current_time = time.time()
            elapsed_time = current_time - self.last_detection_time
            if elapsed_time < 60:
                return True
        return False

    def search_aoi(self, aoi_color: str, contours: List[np.ndarray], res: np.ndarray) -> bool:
        """ Searches for a specific colored area of interest """
        if contours:
            self.get_logger().warning(f'{aoi_color.capitalize()} area of interest detected')
            self.aoi_detected = True
            # Set timer when the contour has been detected
            self.last_detection_time = time.time()
            cv2.imshow("RES2", res)
            cv2.waitKey(1)
        else:
            if self.is_recently_detected():
                self.aoi_detected = True
            else:
                self.get_logger().warning(f'Still searching for the {aoi_color} area of interest')
                self.aoi_detected = False

        return self.aoi_detected

    def go_to_aoi(self, hsv_detect_aoi: np.ndarray, cv_image: np.ndarray, width: int) -> None:
        """ Ensure that the robot positions itself accurately to one of the four key locations of interest"""
        # Find the blob of interest
        blob_bounds = None
        if self.aoi_choice == 'yellow':
            blob_bounds = ([20, 100, 50], [35, 255, 255])
        elif self.aoi_choice == 'green':
            blob_bounds = ([45, 100, 50], [85, 255, 255])
        elif self.aoi_choice == 'blue':
            blob_bounds = ([90, 150, 50], [125, 255, 255])
        elif self.aoi_choice == 'red':
            blob_bounds = ([170, 150, 50], [180, 255, 255])

        if blob_bounds:
            _, contours = self.detect_contours(hsv_detect_aoi, blob_bounds[0], blob_bounds[1], cv_image)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                try:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    # Draw the centroid on the image for visualization
                    cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at the centroid
                    self.pub_aoi_velocities(cx, width)
                    # Display the image for debugging
                    cv2.imshow("Centroid Visualization", cv_image)
                    cv2.waitKey(1)
                except ZeroDivisionError:
                    self.get_logger().error("Zero division error while calculating centroid.")
                    self.stop_robot()
            else:
                self.get_logger().info("Arrived at the area of interest. Stopping the robot and shutting down.")
                self.stop_robot()
                rclpy.shutdown()
        else:
            self.get_logger().error("Invalid aoi choice. Cannot determine blob bounds.")
            self.stop_robot()

    def pub_aoi_velocities(self, cx: int, width: int) -> None:
        """ Publish aoi velocities based on the centroid and AOI color choice """
        linear_vel = 0.2
        angular_vel = 0.06

        # Adjust velocities based on AOI color choice
        if self.aoi_choice == 'green':
            angular_vel = 0.07
        elif self.aoi_choice == 'blue':
            linear_vel = 0.25
            angular_vel = 0.05

        error_x = cx - width / 2
        twist_object = Twist()
        twist_object.linear.x = linear_vel  
        angular_velocity = -error_x / 100

        # Ensure angular velocity is within a certain range
        if angular_velocity < -0.3:
            angular_velocity = -angular_vel
        elif angular_velocity > 0.3:
            angular_velocity = angular_vel

        twist_object.angular.z = angular_velocity
        self.get_logger().info(f"Navigating to {self.aoi_choice} aoi: Adjusting angular velocity ---> {twist_object.angular.z:.2f}")
        self.pub_vel.publish(twist_object)

    def is_valid_aoi_choice(self) -> bool:
        """ Checks if the area of interest location choice is valid """
        valid_aoi_colors = ['yellow', 'green', 'blue', 'red']
        if self.aoi_choice in valid_aoi_colors:
            return True
        else:
            self.get_logger().warning("Please choose between 'yellow', 'green', 'blue' or 'red' for the area of interest location color.")
            return False

    def get_centroid_coordinates(self, centroid: Tuple[int, int], height: int, width: int) -> Tuple[int, int]:
        """ Retrieves centroid's x and y coordinates"""
        try:
            # retrieve selected centroid coordinates
            cx = centroid[0]
            cy = centroid[1]
        except:
            cy, cx = height/2, width/2

        return cx, cy
        
    def stop_robot(self) -> None:
        """ Stops the robot movement """
        twist_object = Twist()
        twist_object.linear.x = 0.0
        twist_object.angular.z = 0.0
        self.pub_vel.publish(twist_object)

    def pub_velocities(self, error: float) -> None:
        """ Publish command velocities """
        twist_object = Twist()
        twist_object.linear.x = 0.2
        angular_velocity = -error / 100

        # Ensure angular velocity is within a certain range
        if angular_velocity < -0.35:
            angular_velocity = -0.35
        elif angular_velocity >= 0.35:
            angular_velocity = 0.35

        twist_object.angular.z = angular_velocity
        self.get_logger().info(f"Adjusting the angular velocity ---> {twist_object.angular.z:.2f}" )
        self.pub_vel.publish(twist_object)

def main(args=None) -> None:
    rclpy.init(args=args)
    go_to_aoi = GoToAOI()
    try:
        rclpy.spin(go_to_aoi)
    except KeyboardInterrupt:
        pass
    go_to_aoi.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()