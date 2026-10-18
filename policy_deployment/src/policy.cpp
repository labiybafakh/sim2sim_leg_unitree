#include "policy_deployment/policy.hpp"
#include <json/json.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

static void check_finite(const float* values, int count) {
  for (int i = 0; i < count; i++) {
    if (!std::isfinite(values[i])) {
      throw std::invalid_argument("State, observations, and actions must be finite");
    }
  }
}

static float read_number(const Json::Value& object, const char* key) {
  const Json::Value& value = object[key];
  if (!value.isNumeric() || !std::isfinite(value.asDouble()) || !std::isfinite(value.asFloat())) {
    throw std::invalid_argument(std::string("Config requires a finite number: ") + key);
  }
  return value.asFloat();
}

RobotState::RobotState() {
  for (int i = 0; i < 3; i++) {
    angular_velocity_body[i] = 0;
    command[i] = 0;
    orientation_xyzw[i] = 0;
  }
  orientation_xyzw[3] = 1;
  for (int i = 0; i < kJoints; i++) {
    positions[i] = 0;
    velocities[i] = 0;
  }
}

PolicyConfig load_config(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::invalid_argument("Cannot open policy config: " + path);
  Json::CharReaderBuilder reader;
  reader["rejectDupKeys"] = true;
  reader["failIfExtra"] = true;
  Json::Value root;
  std::string errors;
  if (!Json::parseFromStream(reader, file, &root, &errors) || !root.isObject()) {
    throw std::invalid_argument("Invalid policy JSON: " + path + ": " + errors);
  }
  PolicyConfig config;
  if (!root["policy_directory"].isString() || root["policy_directory"].asString().empty()) {
    throw std::invalid_argument("policy_directory must be a nonempty directory path");
  }
  // Resolve relative model paths against the JSON, independent of the launch cwd.
  const std::filesystem::path model_directory(root["policy_directory"].asString());
  config.policy_directory = (std::filesystem::absolute(path).parent_path() /
                            model_directory).lexically_normal().string();
  config.action_scale = read_number(root, "action_scale");
  config.policy_frequency_hz = read_number(root, "policy_frequency_hz");
  if (config.action_scale <= 0 || config.policy_frequency_hz <= 0) {
    throw std::invalid_argument("action_scale and policy_frequency_hz must be positive");
  }
  const Json::Value& joystick = root["joystick"];
  if (!joystick.isObject() || !joystick["device"].isString() ||
      joystick["device"].asString().empty() || joystick["device"].asString()[0] != '/') {
    throw std::invalid_argument("joystick.device must be an absolute device path");
  }
  config.joystick_device = joystick["device"].asString();
  if (!joystick["axes"].isArray() || joystick["axes"].size() != 3 ||
      !joystick["scales"].isArray() || joystick["scales"].size() != 3) {
    throw std::invalid_argument("joystick.axes and joystick.scales must each contain 3 values");
  }
  for (int i = 0; i < 3; i++) {
    const Json::Value& axis = joystick["axes"][i];
    const Json::Value& scale = joystick["scales"][i];
    if (!axis.isInt() || axis.asInt() < 0 || axis.asInt() > 255) {
      throw std::invalid_argument("Joystick axis indices must be integers from 0 to 255");
    }
    if (!scale.isNumeric() || !std::isfinite(scale.asFloat())) {
      throw std::invalid_argument("Joystick scales must be finite numbers");
    }
    config.joystick_axes[i] = axis.asInt();
    config.joystick_scales[i] = scale.asFloat();
  }
  config.joystick_deadzone = read_number(joystick, "deadzone");
  if (config.joystick_deadzone < 0 || config.joystick_deadzone >= 1) {
    throw std::invalid_argument("Joystick deadzone must be at least 0 and less than 1");
  }
  const char* button_names[4] = {"start", "stop", "l1", "r1"};
  if (!joystick["buttons"].isObject()) {
    throw std::invalid_argument("Joystick buttons must define start, stop, l1, and r1");
  }
  for (int i = 0; i < 4; i++) {
    const Json::Value& button = joystick["buttons"][button_names[i]];
    if (!button.isInt() || button.asInt() < 0 || button.asInt() > 255) {
      throw std::invalid_argument(std::string("Invalid joystick button: ") + button_names[i]);
    }
    config.joystick_buttons[i] = button.asInt();
    for (int j = 0; j < i; j++) {
      if (config.joystick_buttons[i] == config.joystick_buttons[j]) {
        throw std::invalid_argument("Joystick button mappings must be different");
      }
    }
  }
  const Json::Value& joints = root["joints"];
  if (!joints.isArray() || joints.size() != kJoints) {
    throw std::invalid_argument("Config must contain exactly 12 joints in policy order");
  }
  for (int i = 0; i < kJoints; i++) {
    const Json::Value& joint = joints[i];
    if (!joint.isObject() || !joint["name"].isString() || joint["name"].asString().empty()) {
      throw std::invalid_argument("Every configured joint needs a name");
    }
    config.joint_names[i] = joint["name"].asString();
    for (int j = 0; j < i; j++) {
      if (config.joint_names[i] == config.joint_names[j]) {
        throw std::invalid_argument("Duplicate configured joint: " + config.joint_names[i]);
      }
    }
    config.default_positions[i] = read_number(joint, "default_position");
    config.stiffness[i] = read_number(joint, "stiffness");
    config.damping[i] = read_number(joint, "damping");
    if (config.stiffness[i] < 0 || config.damping[i] < 0) {
      throw std::invalid_argument("Stiffness and damping must be nonnegative");
    }
  }
  return config;
}

void reorder_joints(const std::vector<std::string>& names, const std::vector<float>& values,
                    const PolicyConfig& config, float result[kJoints]) {
  if (names.size() != kJoints || values.size() != kJoints) {
    throw std::invalid_argument("Expected exactly 12 named joints and 12 values");
  }
  for (int i = 0; i < kJoints; i++) {
    if (!std::isfinite(values[i])) throw std::invalid_argument("Joint values must be finite");
    for (int j = 0; j < i; j++) {
      if (names[i] == names[j]) throw std::invalid_argument("Duplicate joint: " + names[i]);
    }
  }
  for (int i = 0; i < kJoints; i++) {
    int found = -1;
    for (int j = 0; j < kJoints; j++) {
      if (names[j] == config.joint_names[i]) found = j;
    }
    if (found < 0) throw std::invalid_argument("Missing joint: " + config.joint_names[i]);
    result[i] = values[found];
  }
}

void action_targets(const float actions[kJoints], const PolicyConfig& config, float targets[kJoints]) {
  check_finite(actions, kJoints);
  for (int i = 0; i < kJoints; i++) {
    targets[i] = config.default_positions[i] + config.action_scale * actions[i];
  }
  check_finite(targets, kJoints);
}

Policy::Policy() : environment_(ORT_LOGGING_LEVEL_WARNING, "policy_deployment"), session_(nullptr) {
  reset();
}

void Policy::load(const PolicyConfig& config, int threads) {
  if (config.policy_directory.empty()) throw std::invalid_argument("policy_directory is required");
  std::string model_path = config.policy_directory + "/policy.onnx";
  if (threads < 1) throw std::invalid_argument("threads must be positive");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetInterOpNumThreads(1);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = Ort::Session(environment_, model_path.c_str(), options);
  if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
    throw std::invalid_argument("Policy must have exactly one input and one output");
  }
  Ort::TypeInfo input_type = session_.GetInputTypeInfo(0);
  Ort::TypeInfo output_type = session_.GetOutputTypeInfo(0);
  Ort::ConstTensorTypeAndShapeInfo input = input_type.GetTensorTypeAndShapeInfo();
  Ort::ConstTensorTypeAndShapeInfo output = output_type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape = input.GetShape();
  std::vector<int64_t> output_shape = output.GetShape();
  if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      input_shape.size() != 2 || input_shape[0] != 1 || input_shape[1] != kObservation ||
      output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] != kJoints) {
    throw std::invalid_argument("Expected float32 input [1,225] and output [1,12]");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_.GetInputNameAllocated(0, allocator).get();
  output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
  config_ = config;
  reset();
}

void Policy::infer(const float observation[kObservation], float actions[kJoints]) {
  if (!session_) throw std::runtime_error("Load the policy before inference");
  check_finite(observation, kObservation);
  float input_values[kObservation];
  for (int i = 0; i < kObservation; i++) input_values[i] = observation[i];
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t shape[2] = {1, kObservation};
  Ort::Value input = Ort::Value::CreateTensor(memory, input_values, sizeof(input_values), shape, 2,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  const char* input_name = input_name_.c_str();
  const char* output_name = output_name_.c_str();
  Ort::Value output(nullptr);
  session_.Run(Ort::RunOptions(nullptr), &input_name, &input, 1, &output_name, &output, 1);
  std::vector<int64_t> output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
  if (output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] != kJoints) {
    throw std::runtime_error("Unexpected inference output shape");
  }
  const float* values = static_cast<const float*>(output.GetTensorRawData());
  check_finite(values, kJoints);
  for (int i = 0; i < kJoints; i++) actions[i] = values[i];
}

void Policy::reset() {
  history_initialized_ = false;
  for (int i = 0; i < kJoints; i++) previous_actions_[i] = 0;
}

void Policy::build_observation(const RobotState& state, bool reset_history,
                               float observation[kObservation], float next_history[kHistory][kFrame]) {
  check_finite(state.angular_velocity_body, 3);
  check_finite(state.orientation_xyzw, 4);
  check_finite(state.positions, kJoints);
  check_finite(state.velocities, kJoints);
  check_finite(state.command, 3);
  double norm = 0;
  for (int i = 0; i < 4; i++) {
    norm += static_cast<double>(state.orientation_xyzw[i]) * state.orientation_xyzw[i];
  }
  if (norm < 1e-12) throw std::invalid_argument("Orientation quaternion has zero norm");
  double inverse_norm = 1.0 / std::sqrt(norm);
  double x = state.orientation_xyzw[0] * inverse_norm;
  double y = state.orientation_xyzw[1] * inverse_norm;
  double z = state.orientation_xyzw[2] * inverse_norm;
  double w = state.orientation_xyzw[3] * inverse_norm;
  float frame[kFrame];
  for (int i = 0; i < 3; i++) {
    frame[i] = 0.2F * state.angular_velocity_body[i];
    frame[6 + i] = state.command[i];
  }
  // Inverse-rotate world down into the body frame.
  frame[3] = static_cast<float>(2 * (w * y - x * z));
  frame[4] = static_cast<float>(-2 * (y * z + w * x));
  frame[5] = static_cast<float>(2 * (x * x + y * y) - 1);
  for (int i = 0; i < kJoints; i++) {
    frame[9 + i] = state.positions[i] - config_.default_positions[i];
    frame[21 + i] = 0.05F * state.velocities[i];
    frame[33 + i] = reset_history ? 0 : previous_actions_[i];
  }
  check_finite(frame, kFrame);
  for (int sample = 0; sample < kHistory; sample++) {
    for (int i = 0; i < kFrame; i++) {
      if (reset_history || !history_initialized_ || sample == kHistory - 1) {
        next_history[sample][i] = frame[i];
      } else {
        next_history[sample][i] = history_[sample + 1][i];
      }
    }
  }
  // Training concatenates each term's history, oldest sample first.
  const int offsets[6] = {0, 3, 6, 9, 21, 33};
  const int widths[6] = {3, 3, 3, 12, 12, 12};
  int output = 0;
  for (int term = 0; term < 6; term++) {
    for (int sample = 0; sample < kHistory; sample++) {
      for (int i = 0; i < widths[term]; i++) {
        observation[output] = next_history[sample][offsets[term] + i];
        output++;
      }
    }
  }
}

void Policy::step(const RobotState& state, bool reset_history, float actions[kJoints]) {
  if (!session_) throw std::runtime_error("Load the policy before inference");
  float observation[kObservation];
  float next_history[kHistory][kFrame];
  float result[kJoints];
  float targets[kJoints];
  build_observation(state, reset_history, observation, next_history);
  infer(observation, result);
  action_targets(result, config_, targets);
  // A rejected state or failed inference must not advance history.
  for (int sample = 0; sample < kHistory; sample++) {
    for (int i = 0; i < kFrame; i++) history_[sample][i] = next_history[sample][i];
  }
  for (int i = 0; i < kJoints; i++) {
    previous_actions_[i] = result[i];
    actions[i] = result[i];
  }
  history_initialized_ = true;
}
