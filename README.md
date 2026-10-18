# sim2sim_leg_unitree

A learning project for leg locomotion through sim2sim, inspired by
[Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab). Some assets are collected from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) and [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco). It is modified to only use leg structure as learning part.
The upper body is removed to reduce complexity: train a twelve-joint leg policy
in Isaac Lab, then run it in MuJoCo through ROS 2.

| Package | Purpose |
| --- | --- |
| [unitree_leg](unitree_leg/README.md) | Policy training and export. |
| [policy_deployment](policy_deployment/README.MD) | CPU ONNX inference, joystick input, ROS messages, and combined launch. |
| [unitree_leg_mujoco](unitree_leg_mujoco/README.md) | MuJoCo simulation, PD control, viewer, and elastic support. |

## Requirements
1. ROS 2 Jazzy
2. IsaacLab 3.0.0-EA
3. MuJoCo 3.3.7
4. ONNX Runtime 1.22.2


## Build

Clone the workspace with its pinned dependencies:

```bash
git clone --recurse-submodules <repository-url> sim2sim_leg_unitree
cd sim2sim_leg_unitree
git lfs pull
```

Git LFS is required for the training robot assets. For an existing clone, run
`git submodule update --init --recursive` to restore the recorded dependency
versions. The three project packages share the root repository; MuJoCo and
ONNX Runtime remain submodules. Commit project changes from the workspace root.

```bash
./scripts/setup_dependencies.sh
```

Copy your exported model into `policy_deployment/models/` and check joint and
joystick settings in `policy_deployment/config/policy.json`:

```bash
cp /path/to/exported/policy.onnx /path/to/exported/policy.onnx.data policy_deployment/models/
```

Build both ROS packages:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to unitree_leg_mujoco \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
```

## Run

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch policy_deployment sim2sim.launch.py
```

- Joystick: **A** arms, **L1+R1** enables commands, **B** stops commands.
- Viewer: drag to rotate, scroll to zoom, **Backspace** resets, **Escape** exits.
- Add `headless:=true` to run without a display or with a viewer-free build.
- Ctrl+C or exiting either node stops both.

## Credits and licenses

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): inspiration for the locomotion task and sim2sim workflow.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): training framework.
- [MuJoCo](https://github.com/google-deepmind/mujoco): simulation engine; [Apache-2.0](unitree_leg_mujoco/third_party/mujoco/LICENSE).
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): policy inference; [MIT](policy_deployment/third_party/onnxruntime/source/LICENSE).
- Local package licenses: [policy_deployment/LICENSE](policy_deployment/LICENSE) and [unitree_leg/LICENSE](unitree_leg/LICENSE).
