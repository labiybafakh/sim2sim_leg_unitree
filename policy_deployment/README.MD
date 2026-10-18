# policy_deployment

ROS 2 C++ node for twelve-joint CPU ONNX inference. Defines the messages shared
with the simulator. See the [workspace guide](../README.md) for build and launch.
Dependency setup helpers live in the workspace-level [scripts/](../scripts/).

## Files

| Path | Purpose |
| --- | --- |
| `src/` and `include/` | Policy inference, state validation, and joystick handling. |
| `msg/` | Robot state and joint command definitions. |
| `config/policy.json` | Model location, joint settings, and joystick mapping. |
| `models/` | Exported `policy.onnx` and `policy.onnx.data`; ignored by Git. |
| `launch/sim2sim.launch.py` | Starts the policy and MuJoCo together. |

## Configuration

`policy_directory` is absolute or relative to the JSON's directory. The default
`../models` works in both the source and installed package.

The `joints` array defines policy order. Each entry sets the name, standing
position in radians, stiffness in Nm/rad, and damping in Nm s/rad. These must
match training. Targets are `default_position + action_scale * action`.
`policy_frequency_hz` defaults to 50 Hz.

The joystick defaults to `/dev/input/js0`. `axes` selects forward, lateral, and
yaw inputs; `scales` sets their speeds; `deadzone` removes small movements.
Button defaults are A=0, B=1, L1=4, R1=5. Change them to match your controller.
L1 and R1 are digital buttons and need no threshold. Commands are disabled at
startup or disconnect; after reconnecting, arm and enable again.

## ROS interface

| Topic | Message | Direction |
| --- | --- | --- |
| `/policy_deployment/state` | `policy_deployment/msg/RobotState` | Input |
| `/policy_deployment/command` | `policy_deployment/msg/JointCommand` | Output |

Both topics use reliable QoS, depth 1. Each valid state produces a command while
enabled. States must include all twelve unique joint names and matching position
and velocity arrays. Use rad, rad/s, body-frame angular velocity, and a
body-to-world quaternion in `[x, y, z, w]` order. Set `reset_history` on the first
state after reset. The output includes targets, actions, stiffness, and damping.

To run only the policy after sourcing the workspace:

```bash
ros2 run policy_deployment policy_node
```

ROS parameters are `config_path` (installed JSON by default) and
`inference_threads` (default `1`).

## Launch arguments

| Argument | Default |
| --- | --- |
| `config_path` | Installed `policy_deployment/config/policy.json` |
| `scene_path` | Installed `unitree_leg_mujoco/scene_29dof.xml` |
| `headless` | `false` |
| `inference_threads` | `1` |
| `auto_release` | `true` |

License: [BSD-3-Clause](LICENSE).
