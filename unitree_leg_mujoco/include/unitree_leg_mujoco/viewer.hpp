#pragma once

#include "unitree_leg_mujoco/simulation.hpp"
#include <GLFW/glfw3.h>
#include <functional>

namespace unitree_leg_mujoco {
// Called on the same thread as the simulation; owns only rendering resources.
class Viewer {
 public:
  Viewer(Simulation& sim, std::function<void()> on_reset);
  ~Viewer();
  Viewer(const Viewer&) = delete;
  Viewer& operator=(const Viewer&) = delete;
  bool open() const;
  void render();

 private:
  Simulation& sim_;
  std::function<void()> on_reset_;
  GLFWwindow* window_ = nullptr;
  mjvCamera camera_{};
  mjvOption option_{};
  mjvScene scene_{};
  mjrContext context_{};
  double last_x_ = 0, last_y_ = 0;
};
}  // namespace unitree_leg_mujoco
