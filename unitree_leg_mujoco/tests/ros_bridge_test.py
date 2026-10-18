"""Exercise the compiled bridge with real ROS messages and services, without a joystick."""
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from policy_deployment.msg import JointCommand, RobotState
from std_srvs.srv import SetBool, Trigger


def main():
    os.environ["ROS_DOMAIN_ID"] = str(100 + os.getpid() % 100)
    rclpy.init()
    node = Node("mujoco_bridge_test")
    config_path = get_package_share_directory("policy_deployment") + "/config/policy.json"
    with open(config_path, encoding="utf-8") as stream:
        joints = json.load(stream)["joints"]
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(JointCommand, "/policy_deployment/command", qos)
    states = []
    enabled = False

    def on_state(state):
        states.append(state)
        if enabled:
            command = JointCommand()
            command.header = state.header
            # Deliberately reverse the policy order over the transport.
            command.joint_names = [joint["name"] for joint in reversed(joints)]
            command.target_positions = [joint["default_position"] for joint in reversed(joints)]
            command.stiffness = [joint["stiffness"] for joint in reversed(joints)]
            command.damping = [joint["damping"] for joint in reversed(joints)]
            publisher.publish(command)

    subscription = node.create_subscription(RobotState, "/policy_deployment/state", on_state, qos)
    elastic = node.create_client(SetBool, "/unitree_leg_mujoco/elastic")
    reset = node.create_client(Trigger, "/unitree_leg_mujoco/reset")

    def spin_until(predicate, timeout=6):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            assert process.poll() is None, f"Bridge exited with code {process.returncode}"
            rclpy.spin_once(node, timeout_sec=0.02)
        assert predicate(), "Timed out waiting for bridge response"

    def call(client, request):
        assert client.wait_for_service(timeout_sec=5), "Bridge service unavailable"
        future = client.call_async(request)
        spin_until(future.done)
        return future.result()

    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([
            sys.argv[1], "--ros-args", "-p", "headless:=true",
            "-p", f"config_path:={config_path}",
            "-p", "elastic_hold:=0.1", "-p", "elastic_release:=0.2",
        ], stdout=log, stderr=log)

        def log_contains(text):
            log.seek(0)
            return text in log.read()

        try:
            spin_until(lambda: len(states) >= 5)
            state = states[-1]
            assert state.joint_names == [joint["name"] for joint in joints]
            assert len(state.joint_positions) == len(state.joint_velocities) == 12
            assert all(math.isfinite(value) for value in state.joint_positions)
            assert abs(sum(value * value for value in state.orientation_xyzw) - 1) < 1e-5
            assert not call(elastic, SetBool.Request(data=False)).success
            assert not log_contains("Elastic support deactivated")
            # Allow DDS endpoint discovery before expecting the stream.
            spin_until(lambda: publisher.get_subscription_count() > 0)
            enabled = True
            spin_until(lambda: log_contains("Elastic support deactivated"))
            enabled = False
            spin_until(lambda: log_contains("Policy commands timed out"))
            assert call(elastic, SetBool.Request(data=True)).success
            assert not call(elastic, SetBool.Request(data=False)).success
            # A delayed command must not restart a stopped controller.
            stale = JointCommand()
            stale.header = states[0].header
            stale.joint_names = [joint["name"] for joint in joints]
            publisher.publish(stale)
            spin_until(lambda: log_contains("Rejected command"))
            before = len(states)
            assert call(reset, Trigger.Request()).success
            spin_until(lambda: any(state.reset_history for state in states[before:]))
            assert process.poll() is None, "Bridge exited unexpectedly"
            print("ROS states, command echo, elastic release, watchdog, stale rejection and reset passed")
        except BaseException:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.returncode != 0:
                log.seek(0)
                print(log.read(), file=sys.stderr)
            node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
