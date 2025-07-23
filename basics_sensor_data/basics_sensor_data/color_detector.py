import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ColorDetector(Node):

    def __init__(self) -> None:
        super().__init__('color_detector')
        self.subscription = self.create_subscription(
            Image,
            '/deepmind_robot1/deepmind_robot1_camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg: Image) -> None:
        """
        Callback function to process image data
        """

        # Retrieve image properties
        height = msg.height
        width = msg.width
        data = msg.data
        byte_depth = 3  # Each pixel consists of 3 bytes (RGB)
 
        # Calculate center pixel index
        center_index = ((height // 2) * width + (width // 2)) * byte_depth

        # Calculate color component indices based on center pixel
        if center_index % byte_depth == 0:
            red_component_index = center_index
            green_component_index = center_index + 1
            blue_component_index = center_index - 1
        if center_index % byte_depth == 1:
            green_component_index = center_index
            blue_component_index = center_index + 1
            red_component_index = center_index - 1
        if center_index % byte_depth == 2:
            blue_component_index = center_index 
            red_component_index = center_index + 1
            green_component_index = center_index - 1

        # Print RGB values of center pixel
        print(f'\nRGB({data[red_component_index]}, {data[green_component_index]}, {data[blue_component_index]}) ')
        
        # Detect colors based on RGB values
        self.detect_color(data[red_component_index], data[green_component_index], data[blue_component_index])

    def detect_color(self, red: int, green: int, blue: int) -> None:
        """
        Detects color based on RGB values
        """

        if (240 <= red <= 255) and (55 <= green <= 80) and (0 <= blue <= 10):
            print(f'ORANGE COLOR DETECTED ....')

        if (220 <= red <= 255) and (190 <= green <= 255) and (0 <= blue <= 10):
            print(f'YELLOW COLOR DETECTED ....')

        if (0 <= red <= 10) and (230 <= green <= 255) and (0 <= blue <= 10):
            print(f'GREEN COLOR DETECTED ....')

        if (10 <= red <= 40) and (90 <= green <= 125) and (130 <= blue <= 185):
            print(f'BLUE COLOR DETECTED ....')


def main(args=None) -> None:
    rclpy.init(args=args)
    color_detector = ColorDetector()
    rclpy.spin(color_detector)
    color_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()