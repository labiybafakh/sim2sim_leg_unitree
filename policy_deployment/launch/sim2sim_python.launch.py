"""Run the Python TorchScript policy, ROS Joy, and MuJoCo simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = ParameterValue(LaunchConfiguration("config_path"), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument(
            "model_path", default_value="",
            description="Python backend: TorchScript file (default: policy_directory/policy.pt).",
        ),
        DeclareLaunchArgument(
            "joy_device_id", default_value="0",
            description="Python backend: SDL joystick index for joy_node.",
        ),
        DeclareLaunchArgument(
            "joy_timeout", default_value="0.5",
            description="Python backend: disable commands after this many seconds without Joy.",
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("policy_deployment"), "config", "policy_python.json",
            ]),
            description="Policy JSON shared by the policy and simulation.",
        ),
        DeclareLaunchArgument(
            "scene_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("unitree_leg_mujoco"), "scene_29dof.xml",
            ]),
            description="MuJoCo scene XML.",
        ),
        DeclareLaunchArgument(
            "headless", default_value="false", choices=["true", "false"],
            description="Run without the MuJoCo viewer.",
        ),
        DeclareLaunchArgument(
            "inference_threads", default_value="1",
            description="Number of CPU inference threads.",
        ),
        DeclareLaunchArgument(
            "auto_release", default_value="true", choices=["true", "false"],
            description="Automatically release elastic support after fresh commands.",
        ),
        Node(
            package="joy", executable="joy_node", name="joy_node", namespace="/",
            output="screen",
            parameters=[{
                "device_id": ParameterValue(LaunchConfiguration("joy_device_id"), value_type=int),
                "deadzone": 0.0,  # Apply the policy JSON deadzone once in the runner.
                "autorepeat_rate": 20.0,
            }],
            on_exit=Shutdown(reason="Joy node exited"),
        ),
        Node(
            package="policy_deployment", executable="policy_node_py",
            name="policy_deployment", namespace="/", output="screen",
            parameters=[{
                "config_path": config,
                "model_path": ParameterValue(LaunchConfiguration("model_path"), value_type=str),
                "inference_threads": ParameterValue(
                    LaunchConfiguration("inference_threads"), value_type=int,
                ),
                "joy_timeout": ParameterValue(LaunchConfiguration("joy_timeout"), value_type=float),
            }],
            on_exit=Shutdown(reason="Python policy node exited"),
        ),
        Node(
            package="unitree_leg_mujoco",
            executable="simulation",
            namespace="/",
            output="screen",
            parameters=[{
                "config_path": config,
                "scene_path": ParameterValue(
                    LaunchConfiguration("scene_path"), value_type=str,
                ),
                "headless": ParameterValue(
                    LaunchConfiguration("headless"), value_type=bool,
                ),
                "auto_release": ParameterValue(
                    LaunchConfiguration("auto_release"), value_type=bool,
                ),
            }],
            on_exit=Shutdown(reason="MuJoCo simulation exited"),
        ),
    ])
