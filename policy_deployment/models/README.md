# Exported policy

Place `policy.onnx` and `policy.onnx.data` here before building the ROS package.
The default JSON resolves `../models` relative to its own location. CMake installs
this directory beside `config/`, so the installed package uses the same layout.
Model files are ignored by Git. Rebuild after replacing them.

To keep model files elsewhere, set `policy_directory` in a custom JSON to an
absolute path or a path relative to that JSON, and pass `config_path` at launch.
