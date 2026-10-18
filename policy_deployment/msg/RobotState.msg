# Send one fresh state per policy step at the configured rate (default 50 Hz).
std_msgs/Header header
bool reset_history
# Body-frame angular velocity [rad/s].
float32[3] angular_velocity_body
# Quaternion rotating body into world, in xyzw order (normalized by the node).
float32[4] orientation_xyzw
# Exactly twelve unique joint names; array order may differ from policy order.
string[] joint_names
# Joint positions [rad] and velocities [rad/s], in the named order.
float32[] joint_positions
float32[] joint_velocities
