# unitree_leg_mujoco

ROS 2 C++ simulation of a floating pelvis and twelve leg joints. Physics runs at
200 Hz and publishes policy states at 50 Hz by default. See the
[workspace guide](../README.md) for build and combined launch instructions.
Run `scripts/setup_dependencies.sh` from the workspace root before colcon; this
package links the MuJoCo installation in `third_party/install/`.

## Standalone simulation

After sourcing ROS and the workspace:

```bash
ros2 run unitree_leg_mujoco simulation
```

The policy node must run separately to provide commands. Both nodes use the
installed policy JSON; for a custom file, pass
`--ros-args -p config_path:=/path/to/policy.json` to each.
For no display, use `--ros-args -p headless:=true`.
Rebuild after changing XML assets.

## Elastic support

Support starts attached. After fresh commands arrive, it holds for 2 seconds,
then fades over 2 seconds. A 0.25-second command timeout reattaches support and
restores standing targets. Reset if the robot has fallen.

Tune `elastic_hold`, `elastic_release`, and `command_timeout` through ROS
parameters when running the simulation alone. For manual release, use
`auto_release:=false` with the combined launch, then enable policy commands and
call:

```bash
ros2 service call /unitree_leg_mujoco/elastic std_srvs/srv/SetBool '{data: false}'
```

Use `'{data: true}'` to reattach support. To reset:

```bash
ros2 service call /unitree_leg_mujoco/reset std_srvs/srv/Trigger '{}'
```

## Tests

From the workspace root, after building with `-DBUILD_TESTING=ON`:

```bash
colcon test --packages-select unitree_leg_mujoco
colcon test-result --verbose
```

Engine license: [Apache-2.0](third_party/mujoco/LICENSE).
