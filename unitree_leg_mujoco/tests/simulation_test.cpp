#include "unitree_leg_mujoco/simulation.hpp"
#include <json/json.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

using namespace unitree_leg_mujoco;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "Invalid input was accepted");
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto path = std::filesystem::temp_directory_path() /
      ("unitree-mujoco-test-" + std::to_string(getpid()) + ".json");
  try {
    std::vector<std::string> names = {
        "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"};
    JointValues standing{-0.1, -0.1, 0, 0, 0, 0, 0.3, 0.3, -0.2, -0.2, 0, 0};
    JointValues kp{100, 100, 100, 100, 100, 100, 150, 150, 40, 40, 40, 40};
    JointValues kd{2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2};
    Json::Value config;
    config["policy_frequency_hz"] = 50;
    for (size_t i = 0; i < kJoints; ++i) {
      Json::Value joint;
      joint["name"] = names[i];
      joint["default_position"] = standing[i];
      joint["stiffness"] = kp[i];
      joint["damping"] = kd[i];
      config["joints"].append(joint);
    }
    { std::ofstream output(path); output << config; }
    Settings settings;
    settings.elastic.hold = 0.1;
    settings.elastic.release = 0.2;
    Simulation sim(std::string(argv[1]) + "/scene_29dof.xml", path.string(), settings);
    require(sim.positions() == standing, "Standing pose/order mismatch");
    require(std::abs(sim.decimation() * settings.timestep - 0.02) < 1e-9, "State rate mismatch");
    require(sim.take_reset_history() && !sim.take_reset_history(), "Initial history reset missing");
    require(std::abs(sim.orientation_xyzw()[3] - 1) < 1e-9, "Quaternion order mismatch");
    // Check the body-frame contract with a rotated root and nonzero angular velocity.
    sim.data()->qpos[3] = sim.data()->qpos[6] = std::sqrt(0.5);
    sim.data()->qvel[3] = 0.2;
    sim.data()->qvel[4] = -0.1;
    sim.data()->qvel[5] = 0.3;
    mj_forward(sim.model(), sim.data());
    auto orientation = sim.orientation_xyzw();
    auto angular = sim.angular_velocity_body();
    require(std::abs(orientation[2] - std::sqrt(0.5)) < 1e-9 &&
            std::abs(orientation[3] - std::sqrt(0.5)) < 1e-9, "Rotated quaternion mismatch");
    require(std::abs(angular[0] - 0.2) < 1e-9 && std::abs(angular[1] + 0.1) < 1e-9 &&
            std::abs(angular[2] - 0.3) < 1e-9, "Angular velocity is not in the body frame");
    sim.reset();
    auto advance = [&](double seconds, bool send_commands = false) {
      int steps = static_cast<int>(std::round(seconds / settings.timestep));
      for (int i = 0; i < steps; ++i) {
        if (send_commands && i % sim.decimation() == 0) sim.command(names, standing, kp, kd);
        double before = sim.data()->time;
        sim.step();
        require(sim.data()->time > before, "MuJoCo reset after unstable dynamics");
      }
    };
    rejects([&] { sim.set_elastic(false); });
    advance(1.0);
    require(sim.elastic_weight() == 1, "Support released before commands");
    require(sim.data()->qpos[2] > 0.6, "Supported robot collapsed");
    // Reverse all arrays to verify mapping by joint name, then inspect motor torques.
    sim.reset();
    auto reversed_names = names;
    auto reversed_targets = standing;
    JointValues zero{};
    JointValues one{};
    one.fill(1);
    for (size_t i = 0; i < kJoints; ++i) reversed_targets[i] += 0.01*(i+1);
    std::reverse(reversed_names.begin(), reversed_names.end());
    std::reverse(reversed_targets.begin(), reversed_targets.end());
    sim.command(reversed_names, reversed_targets, one, zero);
    sim.step();
    for (size_t i = 0; i < kJoints; ++i) {
      std::string motor = names[i].substr(0, names[i].size()-6);
      int actuator = mj_name2id(sim.model(), mjOBJ_ACTUATOR, motor.c_str());
      require(actuator >= 0 && std::abs(sim.data()->ctrl[actuator] - 0.01*(i+1)) < 1e-9,
              "Reordered command mapped to wrong motor");
    }
    auto bad_names = names;
    bad_names[0] = bad_names[1];
    rejects([&] { sim.command(bad_names, standing, kp, kd); });
    auto invalid = standing;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { sim.command(names, invalid, kp, kd); });
    invalid = kp;
    invalid[0] = -1;
    rejects([&] { sim.command(names, standing, invalid, kd); });
    sim.reset();
    advance(0.08, true);
    require(sim.elastic_weight() == 1, "Support released before hold completed");
    advance(0.12, true);
    require(sim.elastic_weight() > 0 && sim.elastic_weight() < 1, "Release did not ramp");
    advance(0.14, true);
    require(sim.elastic_weight() == 0, "Elastic support did not deactivate");
    sim.take_reset_history();
    advance(0.3);
    require(sim.elastic_weight() == 1 && !sim.commands_active(), "Timeout did not reattach support");
    require(sim.take_reset_history(), "Timeout did not reset policy history");
    sim.command(names, standing, kp, kd);
    sim.step();
    require(sim.elastic_weight() == 1, "Restart skipped hold interval");
    sim.set_elastic(true);
    advance(0.4, true);
    require(sim.elastic_weight() == 1, "Manual attachment did not pause release");
    sim.set_elastic(false);
    advance(0.22, true);
    require(sim.elastic_weight() == 0, "Manual release failed");
    sim.reset();
    require(sim.data()->time == 0 && sim.positions() == standing && sim.elastic_weight() == 1,
            "Reset did not restore simulation");
    // Extreme targets still respect training effort limits.
    JointValues extreme{};
    extreme.fill(10000);
    sim.command(names, extreme, kp, kd);
    sim.step();
    int ankle = mj_name2id(sim.model(), mjOBJ_ACTUATOR, "left_ankle_pitch");
    require(sim.data()->ctrl[ankle] == 25, "Ankle effort was not clipped");
    std::filesystem::remove(path);
    std::cout << "Simulation startup, mapping, elastic lifecycle, timeout, reset and effort checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove(path);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
