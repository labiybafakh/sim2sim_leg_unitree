# Header copied from the input state. All arrays use the names below.
std_msgs/Header header
string[12] joint_names
float32[12] actions
# default_position + action_scale * action [rad], from the policy JSON config.
float32[12] target_positions
# Proportional [Nm/rad] and derivative [Nm s/rad] gains.
float32[12] stiffness
float32[12] damping
