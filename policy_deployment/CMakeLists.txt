cmake_minimum_required(VERSION 3.28)
project(policy_deployment VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/install" CACHE PATH
    "Source-built ONNX Runtime install prefix")
find_package(onnxruntime 1.22.2 EXACT CONFIG REQUIRED
  PATHS "${ONNXRUNTIME_ROOT}/lib/cmake/onnxruntime" NO_DEFAULT_PATH)
find_package(jsoncpp CONFIG REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg" "msg/JointCommand.msg" DEPENDENCIES std_msgs)
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(policy_node src/policy_node.cpp src/policy.cpp)
target_include_directories(policy_node PRIVATE include)
ament_target_dependencies(policy_node PUBLIC ament_index_cpp)
target_link_libraries(policy_node PRIVATE
  onnxruntime::onnxruntime JsonCpp::JsonCpp rclcpp::rclcpp "${typesupport_target}")
set_target_properties(policy_node PROPERTIES INSTALL_RPATH "$ORIGIN/..")

install(TARGETS policy_node DESTINATION lib/${PROJECT_NAME})
# Bundle runtime shared libraries so installed programs find them with a relative RPATH.
install(DIRECTORY "${ONNXRUNTIME_ROOT}/lib/" DESTINATION lib FILES_MATCHING PATTERN "libonnxruntime*.so*")
install(FILES third_party/onnxruntime/source/LICENSE third_party/onnxruntime/source/ThirdPartyNotices.txt
  DESTINATION share/${PROJECT_NAME}/licenses)
install(FILES README.MD LICENSE DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
