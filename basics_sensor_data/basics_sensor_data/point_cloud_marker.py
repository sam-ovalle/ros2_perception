import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import struct
from typing import List

class PointCloudMarker(Node):

    def __init__(self) -> None:
        super().__init__('point_cloud_marker')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/deepmind_robot1_depth_sensor/points',
            self.point_cloud_callback,
            10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

    def point_cloud_callback(self, msg: PointCloud2) -> None:

        # Extract points from the point cloud message uniformly
        points = self.extract_points(msg)
        
        # Create markers for the extracted points
        marker_array = self.create_markers(msg, points)

        # Publish the marker array
        self.marker_publisher.publish(marker_array)

    def extract_points(self, msg: PointCloud2) -> List[Point]:
        """Extract points uniformly from the point cloud message"""
        points = []
        num_points = 50
        step_size = len(msg.data) // num_points  # Divide the data into 50 equal parts

        for i in range(0, len(msg.data), step_size):
            if len(points) >= num_points:
                break

            index = i
            x = self.read_float32(msg.data, index)
            y = self.read_float32(msg.data, index + 4)
            z = self.read_float32(msg.data, index + 8)
            p = Point()
            p.x, p.y, p.z = x, y, z
            points.append(p)

        return points

    def create_markers(self, msg: PointCloud2, points: List[Point]) -> MarkerArray:
        """Create markers for the extracted points"""
        marker_array = MarkerArray()
        for idx, point in enumerate(points):
            marker = Marker()
            marker.header = msg.header
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.pose.position = point
            marker.id = idx
            marker_array.markers.append(marker)

        return marker_array

    def read_float32(self, data: bytes, index: int) -> float:
        """Helper function to read float32 from byte array"""
        return struct.unpack_from('f', data, index)[0]

def main(args=None) -> None:
    rclpy.init(args=args)
    point_cloud_marker = PointCloudMarker()
    rclpy.spin(point_cloud_marker)
    point_cloud_marker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()