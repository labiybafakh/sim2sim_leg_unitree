"""Run the CPU policy and the Unitree leg MuJoCo simulation together."""

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
            "config_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("policy_deployment"), "config", "policy.json",
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
            package="policy_deployment",
            executable="policy_node",
            name="policy_deployment",
            namespace="/",
            output="screen",
            parameters=[{
                "config_path": config,
                "inference_threads": ParameterValue(
                    LaunchConfiguration("inference_threads"), value_type=int,
                ),
            }],
            on_exit=Shutdown(reason="Policy node exited"),
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
