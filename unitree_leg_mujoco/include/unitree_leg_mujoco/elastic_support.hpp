#pragma once

#include <mujoco/mujoco.h>
#include <array>

namespace unitree_leg_mujoco {
struct ElasticSettings {
  double hold = 2.0;
  double release = 2.0;
  double stiffness = 2000.0;
  double damping = 200.0;
  bool auto_release = true;
};

// Owns support timing and forces; the simulation owns command freshness.
class ElasticSupport {
 public:
  explicit ElasticSupport(ElasticSettings settings = {});
  void reset(const mjData& data, int body);
  void on_command(double time);
  void on_timeout();
  void attach(double time);
  void request_release(double time);
  void apply(const mjModel& model, mjData& data, int body);
  double weight() const { return weight_; }

 private:
  ElasticSettings settings_;
  std::array<double, 3> anchor_{};
  double first_command_ = -1, weight_ = 1;
  bool release_requested_ = false;
};
}  // namespace unitree_leg_mujoco
