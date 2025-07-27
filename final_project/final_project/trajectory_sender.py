import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from typing import List

class TrajectorySender:
    def __init__(self, node):
        self.node = node
        self.action_client = ActionClient(self.node, FollowJointTrajectory, '/deepmind_bot_head_controller/follow_joint_trajectory')
        
        # Wait for the action server to be available
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error('Action server not available. Exiting...')
            raise Exception('Action server not available')

    def send_trajectory(self, joint_names: List[str], positions: List[float], time_from_start_sec: float) -> bool:
        # Create a FollowJointTrajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = joint_names

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = time_from_start_sec

        # Add the trajectory point to the goal
        goal_msg.trajectory.points.append(point)

        # Send the goal to the action server
        future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            self.node.get_logger().info('Goal successfully completed.')
            return True
        else:
            self.node.get_logger().error('Goal failed to complete')
            return False


