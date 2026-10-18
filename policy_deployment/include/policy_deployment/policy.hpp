#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

const int kJoints = 12;
const int kFrame = 45;
const int kHistory = 5;
const int kObservation = 225;

// Entries in the JSON file define the order of these arrays.
struct PolicyConfig {
  std::string policy_directory;
  std::string joint_names[kJoints];
  float default_positions[kJoints];  // [rad]
  float stiffness[kJoints];  // [Nm/rad]
  float damping[kJoints];  // [Nm s/rad]
  float action_scale;
  double policy_frequency_hz;
  std::string joystick_device;
  int joystick_axes[3];  // Forward, lateral, yaw axis indices.
  float joystick_scales[3];  // Full-stick vx, vy [m/s], yaw rate [rad/s].
  float joystick_deadzone;
  int joystick_buttons[4];  // Start (A), stop (B), L1, R1 button indices.
};

struct RobotState {
  float angular_velocity_body[3];  // [rad/s]
  float orientation_xyzw[4];  // Body to world, unit quaternion.
  float positions[kJoints];  // [rad]
  float velocities[kJoints];  // [rad/s]
  float command[3];  // vx, vy [m/s], yaw rate [rad/s].
  RobotState();
};

PolicyConfig load_config(const std::string& path);
void reorder_joints(const std::vector<std::string>& names, const std::vector<float>& values,
                    const PolicyConfig& config, float result[kJoints]);
void action_targets(const float actions[kJoints], const PolicyConfig& config, float targets[kJoints]);

class Policy {
 public:
  Policy();
  void load(const PolicyConfig& config, int threads = 1);
  void step(const RobotState& state, bool reset_history, float actions[kJoints]);

 private:
  void infer(const float observation[kObservation], float actions[kJoints]);
  void reset();
  void build_observation(const RobotState& state, bool reset_history,
                         float observation[kObservation], float next_history[kHistory][kFrame]);

  PolicyConfig config_;
  Ort::Env environment_;
  Ort::Session session_;
  std::string input_name_;
  std::string output_name_;
  float history_[kHistory][kFrame];
  float previous_actions_[kJoints];
  bool history_initialized_;
};
