#include "unitree_leg_mujoco/elastic_support.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace unitree_leg_mujoco {
ElasticSupport::ElasticSupport(ElasticSettings settings) : settings_(settings) {
  for (double value : {settings.release, settings.stiffness, settings.damping}) {
    if (!std::isfinite(value) || value <= 0)
      throw std::runtime_error("Elastic release duration and gains must be positive");
  }
  if (!std::isfinite(settings.hold) || settings.hold < 0)
    throw std::runtime_error("Elastic hold must be nonnegative");
}

void ElasticSupport::reset(const mjData& data, int body) {
  std::copy_n(data.xipos + 3*body, 3, anchor_.begin());
  release_requested_ = settings_.auto_release;
  on_timeout();
}

void ElasticSupport::on_command(double time) {
  if (first_command_ < 0) first_command_ = time;
}

void ElasticSupport::on_timeout() {
  first_command_ = -1;
  weight_ = 1;
}

void ElasticSupport::attach(double time) {
  weight_ = 1;
  release_requested_ = false;
  if (first_command_ >= 0) first_command_ = time;
}

void ElasticSupport::request_release(double time) {
  if (!release_requested_) first_command_ = std::max(first_command_, time - settings_.hold);
  release_requested_ = true;
}

void ElasticSupport::apply(const mjModel& model, mjData& data, int body) {
  if (release_requested_ && first_command_ >= 0) {
    double elapsed = data.time - first_command_ - settings_.hold;
    weight_ = std::clamp(1.0 - elapsed / settings_.release, 0.0, 1.0);
  }
  if (weight_ == 0) return;
  mjtNum velocity[6], error[3], quat[4];
  mj_objectVelocity(&model, &data, mjOBJ_BODY, body, velocity, 0);
  mju_copy4(quat, data.xquat + 4*body);
  if (quat[0] < 0) mju_scl(quat, quat, -1, 4);
  mju_quat2Vel(error, quat, 1);
  for (int i = 0; i < 3; ++i) {
    double force = settings_.stiffness * (anchor_[i] - data.xipos[3*body+i])
                 - settings_.damping * velocity[3+i]
                 - model.body_subtreemass[body] * model.opt.gravity[i];
    data.xfrc_applied[6*body+i] = weight_ * force;
    data.xfrc_applied[6*body+3+i] = weight_ * (-20*error[i] - 2*velocity[i]);
  }
}
}  // namespace unitree_leg_mujoco
