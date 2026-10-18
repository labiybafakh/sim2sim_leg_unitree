#include "unitree_leg_mujoco/simulation.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <policy_deployment/msg/joint_command.hpp>
#include <policy_deployment/msg/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#ifdef UNITREE_LEG_MUJOCO_WITH_VIEWER
#include "unitree_leg_mujoco/viewer.hpp"
#endif
#include <chrono>
#include <thread>

using unitree_leg_mujoco::Simulation;
using unitree_leg_mujoco::Settings;
using unitree_leg_mujoco::JointValues;

class SimulationNode : public rclcpp::Node {
 public:
  SimulationNode() : Node("unitree_leg_mujoco") {
    auto share = ament_index_cpp::get_package_share_directory("unitree_leg_mujoco");
    auto policy_share = ament_index_cpp::get_package_share_directory("policy_deployment");
    auto scene = declare_parameter<std::string>("scene_path", share + "/scene_29dof.xml");
    auto config = declare_parameter<std::string>("config_path", policy_share + "/config/policy.json");
    Settings settings;
    settings.timestep = declare_parameter<double>("timestep", settings.timestep);
    settings.command_timeout = declare_parameter<double>("command_timeout", settings.command_timeout);
    settings.elastic.hold = declare_parameter<double>("elastic_hold", settings.elastic.hold);
    settings.elastic.release = declare_parameter<double>("elastic_release", settings.elastic.release);
    settings.elastic.stiffness = declare_parameter<double>("elastic_stiffness", settings.elastic.stiffness);
    settings.elastic.damping = declare_parameter<double>("elastic_damping", settings.elastic.damping);
    settings.elastic.auto_release = declare_parameter<bool>("auto_release", settings.elastic.auto_release);
#ifdef UNITREE_LEG_MUJOCO_WITH_VIEWER
    headless_ = declare_parameter<bool>("headless", false);
#else
    headless_ = declare_parameter<bool>("headless", true);
    if (!headless_) throw std::runtime_error("This build has no viewer; use headless:=true");
#endif
    timeout_ = settings.command_timeout;
    sim_ = std::make_unique<Simulation>(scene, config, settings);
    rclcpp::QoS qos(1);
    qos.reliable().durability_volatile();
    publisher_ = create_publisher<policy_deployment::msg::RobotState>("/policy_deployment/state", qos);
    subscription_ = create_subscription<policy_deployment::msg::JointCommand>(
        "/policy_deployment/command", qos,
        [this](const policy_deployment::msg::JointCommand& msg) { on_command(msg); });
    elastic_service_ = create_service<std_srvs::srv::SetBool>("~/elastic",
        [this](const std_srvs::srv::SetBool::Request::SharedPtr request,
               std_srvs::srv::SetBool::Response::SharedPtr response) {
          on_elastic(request->data, *response);
        });
    reset_service_ = create_service<std_srvs::srv::Trigger>("~/reset",
        [this](const std_srvs::srv::Trigger::Request::SharedPtr,
               std_srvs::srv::Trigger::Response::SharedPtr response) {
          reset();
          response->success = true;
          response->message = "Simulation reset with elastic support attached";
        });
    reset();
    timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(settings.timestep)), [this]() { tick(); });
    RCLCPP_INFO(get_logger(), "MuJoCo C++ ready: physics %.0f Hz, states %g Hz; elastic support attached",
                1.0/settings.timestep, sim_->frequency());
    RCLCPP_INFO(get_logger(), "Elastic release: %s, hold %.2f s, ramp %.2f s after fresh policy commands",
                settings.elastic.auto_release ? "automatic" : "manual", settings.elastic.hold, settings.elastic.release);
  }

  void reset() {
    sim_->reset();
    ticks_ = 0;
    reset_stamp_ = now().nanoseconds();
    last_stamp_ = 0;
    previous_weight_ = 1;
  }
  bool headless() const { return headless_; }
  Simulation& sim() { return *sim_; }

 private:
  void on_elastic(bool enabled, std_srvs::srv::SetBool::Response& response) {
    try {
      sim_->set_elastic(enabled);
      response.success = true;
      response.message = enabled ? "Elastic support attached; automatic release paused" :
          "Elastic release scheduled after the hold interval";
    } catch (const std::exception& error) {
      response.success = false;
      response.message = error.what();
    }
  }

  void on_command(const policy_deployment::msg::JointCommand& msg) {
    try {
      int64_t stamp = rclcpp::Time(msg.header.stamp).nanoseconds();
      double age = (now().nanoseconds() - stamp) * 1e-9;
      if (stamp < reset_stamp_ || stamp <= last_stamp_ || age < 0 || age > timeout_)
        throw std::runtime_error("Command timestamp is stale, repeated, or from before reset");
      JointValues targets{}, kp{}, kd{};
      for (size_t i = 0; i < targets.size(); ++i) {
        targets[i] = msg.target_positions[i];
        kp[i] = msg.stiffness[i];
        kd[i] = msg.damping[i];
      }
      sim_->command({msg.joint_names.begin(), msg.joint_names.end()}, targets, kp, kd);
      last_stamp_ = stamp;
    } catch (const std::exception& error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Rejected command: %s", error.what());
    }
  }

  void publish_state() {
    policy_deployment::msg::RobotState msg;
    msg.header.stamp = now();
    msg.header.frame_id = "pelvis";
    msg.reset_history = sim_->take_reset_history();
    auto positions = sim_->positions();
    auto velocities = sim_->velocities();
    auto orientation = sim_->orientation_xyzw();
    auto angular = sim_->angular_velocity_body();
    msg.joint_names = sim_->names();
    msg.joint_positions.assign(positions.begin(), positions.end());
    msg.joint_velocities.assign(velocities.begin(), velocities.end());
    std::copy(orientation.begin(), orientation.end(), msg.orientation_xyzw.begin());
    std::copy(angular.begin(), angular.end(), msg.angular_velocity_body.begin());
    publisher_->publish(msg);
  }

  void tick() {
    bool was_active = sim_->commands_active();
    sim_->step();
    if (was_active && !sim_->commands_active())
      RCLCPP_WARN(get_logger(), "Policy commands timed out; elastic support reattached and standing targets restored");
    double weight = sim_->elastic_weight();
    if (previous_weight_ >= 1 && weight < 1)
      RCLCPP_INFO(get_logger(), "Releasing elastic support gradually");
    if (previous_weight_ > 0 && weight == 0)
      RCLCPP_INFO(get_logger(), "Elastic support deactivated");
    previous_weight_ = weight;
    ticks_ = (ticks_ + 1) % sim_->decimation();
    if (ticks_ == 0) publish_state();
  }
  std::unique_ptr<Simulation> sim_;
  double timeout_ = 0, previous_weight_ = 1;
  bool headless_ = false;
  int ticks_ = 0;
  int64_t reset_stamp_ = 0, last_stamp_ = 0;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<policy_deployment::msg::RobotState>::SharedPtr publisher_;
  rclcpp::Subscription<policy_deployment::msg::JointCommand>::SharedPtr subscription_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr elastic_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    auto node = std::make_shared<SimulationNode>();
    if (node->headless()) {
      rclcpp::spin(node);
    }
#ifdef UNITREE_LEG_MUJOCO_WITH_VIEWER
    else {
      unitree_leg_mujoco::Viewer viewer(node->sim(), [node]() { node->reset(); });
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(node);
      auto next_frame = std::chrono::steady_clock::now();
      while (rclcpp::ok() && viewer.open()) {
        executor.spin_some(std::chrono::milliseconds(1));
        auto now = std::chrono::steady_clock::now();
        if (now >= next_frame) {
          viewer.render();
          next_frame = now + std::chrono::milliseconds(16);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
#endif
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("unitree_leg_mujoco"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
