#pragma once

#include "unitree_leg_mujoco/elastic_support.hpp"
#include <mujoco/mujoco.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace unitree_leg_mujoco {
constexpr size_t kJoints = 12;
using JointValues = std::array<double, kJoints>;

struct Settings {
  double timestep = 0.005;
  double command_timeout = 0.25;
  ElasticSettings elastic;
};

class Simulation {
 public:
  Simulation(const std::string& scene, const std::string& config, Settings settings = {});
  void reset();
  void step();
  void command(const std::vector<std::string>& names, const JointValues& targets,
               const JointValues& stiffness, const JointValues& damping);
  void set_elastic(bool enabled);
  bool take_reset_history();
  const std::vector<std::string>& names() const { return names_; }
  JointValues positions() const;
  JointValues velocities() const;
  std::array<double, 4> orientation_xyzw() const;
  std::array<double, 3> angular_velocity_body() const;
  double frequency() const { return frequency_; }
  int decimation() const { return decimation_; }
  double elastic_weight() const { return support_.weight(); }
  bool commands_active() const { return last_command_ >= 0; }
  mjModel* model() { return model_.get(); }
  mjData* data() { return data_.get(); }

 private:
  int id(mjtObj type, const std::string& name) const;
  void standing_targets();
  void apply_joint_torques();
  Settings settings_;
  ElasticSupport support_;
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model_{nullptr, mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data_{nullptr, mj_deleteData};
  std::vector<std::string> names_;
  std::array<int, kJoints> qpos_{}, qvel_{}, actuators_{};
  JointValues standing_{}, standing_kp_{}, standing_kd_{};
  std::vector<double> targets_, kp_, kd_;
  int body_ = 0, decimation_ = 0;
  double frequency_ = 0, last_command_ = -1;
  bool reset_history_ = true;
};
}  // namespace unitree_leg_mujoco
