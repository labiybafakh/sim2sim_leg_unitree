#include "unitree_leg_mujoco/viewer.hpp"
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace unitree_leg_mujoco {
Viewer::Viewer(Simulation& sim, std::function<void()> on_reset)
    : sim_(sim), on_reset_(std::move(on_reset)) {
  if (!glfwInit()) throw std::runtime_error("GLFW initialization failed; use -p headless:=true without a display");
  window_ = glfwCreateWindow(1200, 900, "Unitree leg MuJoCo", nullptr, nullptr);
  if (!window_) { glfwTerminate(); throw std::runtime_error("Cannot create viewer window"); }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(0);
  mjv_defaultCamera(&camera_);
  camera_.type = mjCAMERA_TRACKING;
  camera_.trackbodyid = mj_name2id(sim_.model(), mjOBJ_BODY, "pelvis");
  camera_.distance = 2.5;
  camera_.azimuth = 135;
  camera_.elevation = -20;
  mjv_defaultOption(&option_);
  mjv_defaultScene(&scene_);
  mjr_defaultContext(&context_);
  mjv_makeScene(sim_.model(), &scene_, 2000);
  mjr_makeContext(sim_.model(), &context_, mjFONTSCALE_150);
  glfwSetWindowUserPointer(window_, this);
  glfwSetScrollCallback(window_, [](GLFWwindow* window, double, double y) {
    auto* self = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    mjv_moveCamera(self->sim_.model(), mjMOUSE_ZOOM, 0, -0.05*y, &self->scene_, &self->camera_);
  });
  glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* self = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
    if (key == GLFW_KEY_BACKSPACE) self->on_reset_();
  });
}
Viewer::~Viewer() {
  mjr_freeContext(&context_);
  mjv_freeScene(&scene_);
  glfwDestroyWindow(window_);
  glfwTerminate();
}
bool Viewer::open() const { return !glfwWindowShouldClose(window_); }
void Viewer::render() {
  glfwPollEvents();
  double x, y;
  glfwGetCursorPos(window_, &x, &y);
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  if (height > 0 && glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    mjv_moveCamera(sim_.model(), mjMOUSE_ROTATE_V, (x-last_x_)/height,
                   (y-last_y_)/height, &scene_, &camera_);
  last_x_ = x;
  last_y_ = y;
  mjrRect viewport{0, 0, 0, 0};
  glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
  mjv_updateScene(sim_.model(), sim_.data(), &option_, nullptr, &camera_, mjCAT_ALL, &scene_);
  mjr_render(viewport, &scene_, &context_);
  char status[128];
  std::snprintf(status, sizeof(status), "Elastic support: %.0f%%\nPolicy commands: %s",
                100*sim_.elastic_weight(), sim_.commands_active() ? "active" : "waiting");
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, status,
              "Drag: rotate | Scroll: zoom\nBackspace: reset | Esc: quit", &context_);
  glfwSwapBuffers(window_);
}
}  // namespace unitree_leg_mujoco
