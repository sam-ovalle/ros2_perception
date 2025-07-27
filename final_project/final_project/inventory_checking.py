from typing import Dict
import rclpy
from rclpy.node import Node
from final_project.trajectory_sender import TrajectorySender
from std_srvs.srv import Trigger
from custom_msgs.msg import ObjectCount
from std_msgs.msg import String
import time


class InventoryChecking(Node):
    def __init__(self) -> None:
        super().__init__('inventory_checking')
        self.trajectory_sender = TrajectorySender(self)
        self.upper_count = {}
        self.lower_count = {}
        self.detected_objects = ""
        self.aoi_choice = ""

        self.object_count_sub = self.create_subscription(
            ObjectCount,
            '/detected_object_count',
            self.object_count_callback,
            10)

        self.aoi_choice_sub = self.create_subscription(
            String,
            '/aoi_choice',
            self.aoi_choice_callback,
            10
        )

        self.start_detection_client = self.create_client(Trigger, 'start_detection')
        self.stop_detection_client = self.create_client(Trigger, 'stop_detection')

        while not self.start_detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('start_detection service not available, waiting...')

        while not self.stop_detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('stop_detection service not available, waiting...')

    def aoi_choice_callback(self, msg: String) -> None:
        self.aoi_choice = msg.data

    def object_count_callback(self, msg: ObjectCount) -> None:
        if not self.detected_objects:
            return

        object_counts = {msg.classes[i]: msg.counts[i] for i in range(len(msg.classes))}
        self.get_logger().info(f'Detected objects: {object_counts}')

        if self.detected_objects == 'upper':
            self.upper_count = object_counts
        elif self.detected_objects == 'lower':
            self.lower_count = object_counts

        self.detected_objects = ""

    def call_service(self, client, service_name: str) -> bool:
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'{service_name} service call succeeded: {future.result().message}')
            return True
        else:
            self.get_logger().info(f'{service_name} service call failed')
            return False

    def perform_inventory(self) -> None:
        # Move head to look upwards
        self.get_logger().info('Moving head upwards for upper shelf inventory...')
        if not self.trajectory_sender.send_trajectory(
                ['deepmind_robot1_head_base_joint', 'deepmind_robot1_head_joint'],
                [0.0, -0.2], 1):
            return

        # Start detection for upper shelf
        self.detected_objects = 'upper'
        if not self.call_service(self.start_detection_client, 'start_detection'):
            return

        # Wait for detection to complete
        time.sleep(10)  # Adjust the sleep time as necessary

        # Stop detection
        if not self.call_service(self.stop_detection_client, 'stop_detection'):
            return

        # Move head to look downwards
        self.get_logger().info('Moving head downwards for lower shelf inventory...')
        if not self.trajectory_sender.send_trajectory(
                ['deepmind_robot1_head_base_joint', 'deepmind_robot1_head_joint'],
                [0.0, 0.7], 1):
            return

        # Start detection for lower shelf
        if not self.call_service(self.start_detection_client, 'start_detection'):
            return

        self.detected_objects = 'lower'
        # Wait for detection to complete
        time.sleep(10)  # Adjust the sleep time as necessary

        # Stop detection
        if not self.call_service(self.stop_detection_client, 'stop_detection'):
            return

        # Sum up the counts
        total_count = self.sum_counts(self.upper_count, self.lower_count)
        self.get_logger().info(f'Total inventory count: {total_count}')

        # Perform inventory checks
        self.get_logger().info(f'Lets check the inventory: ')
        self.check_inventory(total_count)

    def sum_counts(self, upper_count: Dict[str, int], lower_count: Dict[str, int]) -> Dict[str, int]:
        total_count = upper_count.copy()
        for obj, count in lower_count.items():
            if obj in total_count:
                total_count[obj] += count
            else:
                total_count[obj] = count
        return total_count

    def check_inventory(self, total_count: Dict[str, int]) -> None:
        expected_counts = {}
        allowed_items = []

        if self.aoi_choice in ['red', 'blue']:
            expected_counts = {'banana': 6, 'pizza': 4}
            allowed_items = ['banana', 'pizza']
        elif self.aoi_choice in ['green', 'yellow']:
            expected_counts = {'cell phone': 8, 'laptop': 4}
            allowed_items = ['cell phone', 'laptop']

        missing_items = {item: expected_counts[item] - total_count[item] for item in expected_counts if item not in total_count or total_count[item] < expected_counts[item]}
        misplaced_items = {item: count for item, count in total_count.items() if item not in allowed_items}

        if missing_items:
            self.get_logger().info(f'Missing items: {missing_items}')
        else:
            self.get_logger().info('No missing items.')

        if misplaced_items:
            self.get_logger().info(f'Misplaced items: {misplaced_items}')
        else:
            self.get_logger().info('No misplaced items.')

def main(args=None) -> None:
    rclpy.init(args=args)
    node = InventoryChecking()
    try:
        node.perform_inventory()
    except Exception as e:
        node.get_logger().error(f'Error during inventory: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()