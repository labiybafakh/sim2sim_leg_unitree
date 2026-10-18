#include "unitree_leg_mujoco/simulation.hpp"
#include <json/json.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace unitree_leg_mujoco {
namespace {
double number(const Json::Value& value, const char* name) {
  if (!value.isNumeric() || !std::isfinite(value.asDouble()))
    throw std::runtime_error(std::string("Expected finite number: ") + name);
  return value.asDouble();
}
}

Simulation::Simulation(const std::string& scene, const std::string& config, Settings settings)
    : settings_(settings), support_(settings.elastic) {
  for (double value : {settings.timestep, settings.command_timeout}) {
    if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Timestep and command timeout must be positive");
  }
  Json::Value root;
  std::ifstream stream(config);
  if (!stream) throw std::runtime_error("Cannot open policy config: " + config);
  stream >> root;
  frequency_ = number(root["policy_frequency_hz"], "policy_frequency_hz");
  if (frequency_ <= 0) throw std::runtime_error("Policy frequency must be positive");
  double steps = 1.0 / frequency_ / settings.timestep;
  if (steps < 1 || steps > 100000 || std::abs(steps - std::round(steps)) > 1e-6)
    throw std::runtime_error("Policy period must be an integer multiple of timestep");
  decimation_ = static_cast<int>(std::round(steps));
  if (settings.command_timeout <= 1.0 / frequency_)
    throw std::runtime_error("Command timeout must exceed the policy period");
  char error[1024] = {};
  model_.reset(mj_loadXML(scene.c_str(), nullptr, error, sizeof(error)));
  if (!model_) throw std::runtime_error(std::string("Cannot load scene: ") + error);
  model_->opt.timestep = settings.timestep;
  model_->opt.integrator = mjINT_IMPLICITFAST;
  data_.reset(mj_makeData(model_.get()));
  if (!data_) throw std::runtime_error("Cannot allocate MuJoCo data");
  body_ = id(mjOBJ_BODY, "pelvis");
  const auto& joints = root["joints"];
  if (!joints.isArray() || joints.size() != kJoints)
    throw std::runtime_error("Policy config must contain twelve joints");
  std::set<std::string> unique;
  for (size_t i = 0; i < kJoints; ++i) {
    const auto& entry = joints[static_cast<int>(i)];
    std::string name = entry["name"].asString();
    if (name.empty() || !unique.insert(name).second) throw std::runtime_error("Duplicate or empty joint name");
    names_.push_back(name);
    int joint = id(mjOBJ_JOINT, name);
    if (model_->jnt_type[joint] != mjJNT_HINGE) throw std::runtime_error("Policy joints must be hinges");
    qpos_[i] = model_->jnt_qposadr[joint];
    qvel_[i] = model_->jnt_dofadr[joint];
    int actuator = -1;
    for (int a = 0; a < model_->nu; ++a) {
      if (model_->actuator_trntype[a] == mjTRN_JOINT && model_->actuator_trnid[2*a] == joint) {
        if (actuator >= 0) throw std::runtime_error("Multiple motors for " + name);
        actuator = a;
      }
    }
    if (actuator < 0) throw std::runtime_error("Missing motor for " + name);
    actuators_[i] = actuator;
    standing_[i] = number(entry["default_position"], "default_position");
    standing_kp_[i] = number(entry["stiffness"], "stiffness");
    standing_kd_[i] = number(entry["damping"], "damping");
    if (standing_kp_[i] < 0 || standing_kd_[i] < 0) throw std::runtime_error("Negative standing gains");
    // Match the training limits, which differ from the stock 29-DOF XML.
    double limit = name.find("ankle") != std::string::npos ? 25 :
        (name.find("knee") != std::string::npos || name.find("hip_roll") != std::string::npos ? 139 : 88);
    model_->actuator_ctrlrange[2*actuator] = model_->jnt_actfrcrange[2*joint] = -limit;
    model_->actuator_ctrlrange[2*actuator+1] = model_->jnt_actfrcrange[2*joint+1] = limit;
  }
  targets_.resize(model_->nu);
  kp_.resize(model_->nu);
  kd_.resize(model_->nu);
  reset();
}

int Simulation::id(mjtObj type, const std::string& name) const {
  int result = mj_name2id(model_.get(), type, name.c_str());
  if (result < 0) throw std::runtime_error("Model is missing " + name);
  return result;
}

void Simulation::standing_targets() {
  for (size_t i = 0; i < kJoints; ++i) {
    targets_[actuators_[i]] = standing_[i];
    kp_[actuators_[i]] = standing_kp_[i];
    kd_[actuators_[i]] = standing_kd_[i];
  }
}

void Simulation::reset() {
  mj_resetData(model_.get(), data_.get());
  for (size_t i = 0; i < kJoints; ++i) data_->qpos[qpos_[i]] = standing_[i];
  for (int a = 0; a < model_->nu; ++a) {
    int joint = model_->actuator_trnid[2*a];
    targets_[a] = data_->qpos[model_->jnt_qposadr[joint]];
    kp_[a] = 40;
    kd_[a] = 2;
  }
  standing_targets();
  mj_forward(model_.get(), data_.get());
  support_.reset(*data_, body_);
  last_command_ = -1;
  reset_history_ = true;
}

void Simulation::command(const std::vector<std::string>& names, const JointValues& targets,
                         const JointValues& stiffness, const JointValues& damping) {
  if (names.size() != kJoints || std::set<std::string>(names.begin(), names.end()).size() != kJoints)
    throw std::runtime_error("Commands must name twelve unique policy joints");
  std::array<size_t, kJoints> order{};
  for (size_t i = 0; i < kJoints; ++i) {
    auto found = std::find(names.begin(), names.end(), names_[i]);
    if (found == names.end()) throw std::runtime_error("Missing command joint: " + names_[i]);
    order[i] = std::distance(names.begin(), found);
    if (!std::isfinite(targets[i]) || !std::isfinite(stiffness[i]) || !std::isfinite(damping[i]) ||
        stiffness[i] < 0 || damping[i] < 0) throw std::runtime_error("Invalid command targets or gains");
  }
  // Commit only after validating the entire message.
  for (size_t i = 0; i < kJoints; ++i) {
    targets_[actuators_[i]] = targets[order[i]];
    kp_[actuators_[i]] = stiffness[order[i]];
    kd_[actuators_[i]] = damping[order[i]];
  }
  support_.on_command(data_->time);
  last_command_ = data_->time;
}

void Simulation::set_elastic(bool enabled) {
  if (enabled) {
    support_.attach(data_->time);
  } else {
    if (last_command_ < 0 || data_->time - last_command_ > settings_.command_timeout)
      throw std::runtime_error("Fresh policy commands are required before releasing support");
    support_.request_release(data_->time);
  }
}

void Simulation::step() {
  if (last_command_ >= 0 && data_->time - last_command_ > settings_.command_timeout) {
    last_command_ = -1;
    support_.on_timeout();
    standing_targets();
    reset_history_ = true;
  }
  apply_joint_torques();
  mju_zero(data_->xfrc_applied, 6*model_->nbody);
  support_.apply(*model_, *data_, body_);
  mj_step(model_.get(), data_.get());
  // Body transforms must describe the post-integration state.
  mj_forward(model_.get(), data_.get());
}

void Simulation::apply_joint_torques() {
  for (int a = 0; a < model_->nu; ++a) {
    int joint = model_->actuator_trnid[2*a];
    double torque = kp_[a] * (targets_[a] - data_->qpos[model_->jnt_qposadr[joint]])
                  - kd_[a] * data_->qvel[model_->jnt_dofadr[joint]];
    data_->ctrl[a] = std::clamp(torque, model_->actuator_ctrlrange[2*a], model_->actuator_ctrlrange[2*a+1]);
  }
}

bool Simulation::take_reset_history() {
  bool value = reset_history_;
  reset_history_ = false;
  return value;
}
JointValues Simulation::positions() const {
  JointValues result{};
  for (size_t i = 0; i < kJoints; ++i) result[i] = data_->qpos[qpos_[i]];
  return result;
}
JointValues Simulation::velocities() const {
  JointValues result{};
  for (size_t i = 0; i < kJoints; ++i) result[i] = data_->qvel[qvel_[i]];
  return result;
}
std::array<double, 4> Simulation::orientation_xyzw() const {
  const auto* quat = data_->xquat + 4*body_;
  return {quat[1], quat[2], quat[3], quat[0]};
}
std::array<double, 3> Simulation::angular_velocity_body() const {
  mjtNum velocity[6];
  // XBODY uses the link frame; BODY uses the rotated principal-inertia frame.
  mj_objectVelocity(model_.get(), data_.get(), mjOBJ_XBODY, body_, velocity, 1);
  return {velocity[0], velocity[1], velocity[2]};
}
}  // namespace unitree_leg_mujoco
