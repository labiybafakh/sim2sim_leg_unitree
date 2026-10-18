#include "policy_deployment/policy.hpp"
#include "policy_deployment/msg/robot_state.hpp"
#include "policy_deployment/msg/joint_command.hpp"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <functional>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <linux/joystick.h>
#include <time.h>
#include <unistd.h>

class PolicyNode : public rclcpp::Node {
 public:
  PolicyNode() : Node("policy_deployment"), joystick_fd_(-1), next_joystick_retry_(0) {
    reset_controls();
    std::string config_path = declare_parameter<std::string>("config_path",
        ament_index_cpp::get_package_share_directory("policy_deployment") + "/config/policy.json");
    int threads = declare_parameter<int>("inference_threads", 1);
    config_ = load_config(config_path);
    policy_.load(config_, threads);

    rclcpp::QoS qos(1);
    qos.reliable();
    qos.durability_volatile();
    publisher_ = create_publisher<policy_deployment::msg::JointCommand>("~/command", qos);
    subscription_ = create_subscription<policy_deployment::msg::RobotState>(
        "~/state", qos, std::bind(&PolicyNode::on_state, this, std::placeholders::_1));
    joystick_timer_ = create_wall_timer(std::chrono::milliseconds(20),
                                       std::bind(&PolicyNode::read_joystick, this));
    RCLCPP_INFO(get_logger(), "CPU policy ready: ~/state -> ~/command; send states at %g Hz",
                config_.policy_frequency_hz);
    RCLCPP_INFO(get_logger(), "Controls: A arms, L1+R1 enables commands, B stops");
    RCLCPP_INFO(get_logger(), "Joystick buttons: A=%d B=%d L1=%d R1=%d",
                config_.joystick_buttons[0], config_.joystick_buttons[1],
                config_.joystick_buttons[2], config_.joystick_buttons[3]);
  }

  ~PolicyNode() {
    if (joystick_fd_ >= 0) close(joystick_fd_);
  }

 private:
  void reset_controls() {
    for (int i = 0; i < 3; i++) joystick_command_[i] = 0;
    for (int i = 0; i < 256; i++) joystick_buttons_[i] = false;
    started_ = false;
    publish_commands_ = false;
    combo_was_held_ = false;
    reset_history_ = true;
  }

  void read_joystick() {
    bool start_pressed = false;
    bool stop_pressed = false;
    if (joystick_fd_ < 0) {
      struct timespec time;
      clock_gettime(CLOCK_MONOTONIC, &time);
      double now = time.tv_sec + time.tv_nsec * 1e-9;
      if (now < next_joystick_retry_) return;
      next_joystick_retry_ = now + 1.0;
      joystick_fd_ = open(config_.joystick_device.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
      if (joystick_fd_ < 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                            "Cannot open joystick %s: %s; commands are disabled",
                            config_.joystick_device.c_str(), std::strerror(errno));
        return;
      }
      RCLCPP_INFO(get_logger(), "Joystick connected: %s", config_.joystick_device.c_str());
    }

    struct js_event event;
    while (true) {
      ssize_t count = read(joystick_fd_, &event, sizeof(event));
      if (count == static_cast<ssize_t>(sizeof(event))) {
        int type = event.type & ~JS_EVENT_INIT;
        if (type == JS_EVENT_BUTTON) {
          bool pressed = event.value != 0;
          bool was_pressed = joystick_buttons_[event.number];
          joystick_buttons_[event.number] = pressed;
          // Buttons held during connection must not start the runner.
          if (!(event.type & JS_EVENT_INIT) && pressed && !was_pressed) {
            if (event.number == config_.joystick_buttons[0]) start_pressed = true;
            if (event.number == config_.joystick_buttons[1]) stop_pressed = true;
          }
          continue;
        }
        // Initial axis events also describe the current stick position.
        if (type != JS_EVENT_AXIS) continue;
        float value = event.value / 32767.0F;
        if (value < -1) value = -1;
        if (value > 1) value = 1;
        float magnitude = std::fabs(value);
        if (magnitude <= config_.joystick_deadzone) value = 0;
        else {
          magnitude = (magnitude - config_.joystick_deadzone) / (1 - config_.joystick_deadzone);
          if (value < 0) value = -magnitude;
          else value = magnitude;
        }
        for (int i = 0; i < 3; i++) {
          if (event.number == config_.joystick_axes[i]) {
            joystick_command_[i] = value * config_.joystick_scales[i];
          }
        }
      } else if (count < 0 && errno == EINTR) {
        continue;
      } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
      } else {
        close(joystick_fd_);
        joystick_fd_ = -1;
        reset_controls();
        RCLCPP_WARN(get_logger(), "Joystick disconnected; commands are disabled");
        return;
      }
    }

    bool combo_held = joystick_buttons_[config_.joystick_buttons[2]] &&
                      joystick_buttons_[config_.joystick_buttons[3]];
    // Stop takes priority even if A and the combination are pressed in the same batch.
    if (stop_pressed || joystick_buttons_[config_.joystick_buttons[1]]) {
      if (started_) RCLCPP_INFO(get_logger(), "Stopped: joint commands disabled");
      started_ = false;
      publish_commands_ = false;
      reset_history_ = true;
    } else {
      if (start_pressed && !started_) {
        started_ = true;
        reset_history_ = true;
        RCLCPP_INFO(get_logger(), "Armed: press L1+R1 to send joint commands");
      }
      if (started_ && combo_held && !combo_was_held_ && !publish_commands_) {
        publish_commands_ = true;
        RCLCPP_INFO(get_logger(), "Running: joint commands enabled");
      }
    }
    combo_was_held_ = combo_held;
  }

  void on_state(const policy_deployment::msg::RobotState& message) {
    try {
      read_joystick();
      if (!publish_commands_) return;
      RobotState state;
      for (int i = 0; i < 3; i++) {
        state.angular_velocity_body[i] = message.angular_velocity_body[i];
        state.command[i] = joystick_command_[i];
      }
      for (int i = 0; i < 4; i++) state.orientation_xyzw[i] = message.orientation_xyzw[i];
      reorder_joints(message.joint_names, message.joint_positions, config_, state.positions);
      reorder_joints(message.joint_names, message.joint_velocities, config_, state.velocities);

      float actions[kJoints];
      float targets[kJoints];
      policy_.step(state, message.reset_history || reset_history_, actions);
      reset_history_ = false;
      action_targets(actions, config_, targets);

      policy_deployment::msg::JointCommand output;
      output.header = message.header;
      for (int i = 0; i < kJoints; i++) {
        output.joint_names[i] = config_.joint_names[i];
        output.actions[i] = actions[i];
        output.target_positions[i] = targets[i];
        output.stiffness[i] = config_.stiffness[i];
        output.damping[i] = config_.damping[i];
      }
      publisher_->publish(output);
    } catch (const std::exception& error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Rejected state: %s", error.what());
    }
  }

  PolicyConfig config_;
  Policy policy_;
  int joystick_fd_;
  double next_joystick_retry_;
  float joystick_command_[3];
  bool joystick_buttons_[256];
  bool started_;
  bool publish_commands_;
  bool combo_was_held_;
  bool reset_history_;
  rclcpp::TimerBase::SharedPtr joystick_timer_;
  rclcpp::Subscription<policy_deployment::msg::RobotState>::SharedPtr subscription_;
  rclcpp::Publisher<policy_deployment::msg::JointCommand>::SharedPtr publisher_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    PolicyNode node;
    rclcpp::spin(node.get_node_base_interface());
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("policy_deployment"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
