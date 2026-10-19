#!/usr/bin/env python3

import math
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy

from policy_deployment.msg import JointCommand, RobotState
from policy_runtime import JoyControls, Policy, load_config


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_deployment")
        default_config = get_package_share_directory("policy_deployment") + "/config/policy_python.json"
        config = load_config(self.declare_parameter("config_path", default_config).value)
        model = self.declare_parameter("model_path", "").value
        threads = self.declare_parameter("inference_threads", 1).value
        self.timeout = self.declare_parameter("joy_timeout", 0.5).value
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("joy_timeout must be finite and positive")
        self.policy = Policy(config, model or None, threads)
        self.controls = JoyControls(config["joystick"])
        self.last_joy = None
        self.last_warning = float("-inf")
        self.publisher = self.create_publisher(JointCommand, "~/command", QoSProfile(depth=1))
        self.create_subscription(RobotState, "~/state", self.on_state, QoSProfile(depth=1))
        self.create_subscription(Joy, "/joy", self.on_joy, QoSProfile(depth=1))
        self.create_timer(min(0.1, self.timeout / 2), self.check_joy)
        self.get_logger().info("TorchScript policy ready; A arms, L1+R1 enables commands, B stops")

    def warn(self, error):
        now = time.monotonic()
        if now - self.last_warning >= 2:
            self.get_logger().warning(str(error))
            self.last_warning = now

    def check_joy(self):
        if self.last_joy is not None and time.monotonic() - self.last_joy > self.timeout:
            self.controls.clear()
            self.last_joy = None
            self.warn("Joy messages timed out; commands disabled. Arm and enable again.")

    def on_joy(self, message):
        self.check_joy()
        try:
            self.controls.update(message.axes, message.buttons)
            self.last_joy = time.monotonic()
        except (ValueError, TypeError, IndexError) as error:
            self.controls.clear()
            self.last_joy = None
            self.warn(error)

    def on_state(self, message):
        self.check_joy()
        if not self.controls.enabled:
            return
        try:
            actions, targets = self.policy.step(
                message.angular_velocity_body, message.orientation_xyzw,
                list(message.joint_names), message.joint_positions, message.joint_velocities,
                self.controls.command, reset=message.reset_history or self.controls.reset_history,
            )
            output = JointCommand()
            output.header = message.header
            joints = self.policy.config["joints"]
            output.joint_names = [joint["name"] for joint in joints]
            output.actions = actions.tolist()
            output.target_positions = targets.tolist()
            output.stiffness = [float(joint["stiffness"]) for joint in joints]
            output.damping = [float(joint["damping"]) for joint in joints]
            self.publisher.publish(output)
            self.controls.reset_history = False
        except Exception as error:
            self.warn(f"Rejected state: {error}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
