#!/usr/bin/env bash
# Initialize and build third-party dependencies only; run colcon separately.
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
package_root="$workspace_root/policy_deployment"
ort_commit=5630b081cd25e4eccc7516a652ff956e51676794
mujoco_commit=f1d45bd5422c74beddfb0d1deb590a02583d21de
jobs="${JOBS:-4}"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64|Linux:aarch64|Linux:arm64) ;;
  *) echo 'Only native Linux x86_64 and ARM64 builds are supported.' >&2; exit 1 ;;
esac
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'JOBS must be a positive integer.' >&2
  exit 1
fi

source_dir="$package_root/third_party/onnxruntime/source"
build_dir="$package_root/third_party/onnxruntime/build"
install_dir="$package_root/third_party/onnxruntime/install"

git -C "$workspace_root" submodule update --init --recursive -- \
  policy_deployment/third_party/onnxruntime/source \
  unitree_leg_mujoco/third_party/mujoco

check_source() {
  local name="$1" directory="$2" expected_commit="$3"
  if [[ "$(git -C "$directory" rev-parse HEAD)" != "$expected_commit" ]]; then
    echo "$name must be pinned to $expected_commit." >&2
    exit 1
  fi
  if ! git -C "$directory" diff --quiet HEAD --; then
    echo "$name sources are modified." >&2
    exit 1
  fi
}

mujoco_root="$workspace_root/unitree_leg_mujoco/third_party"
check_source 'ONNX Runtime' "$source_dir" "$ort_commit"
check_source 'MuJoCo' "$mujoco_root/mujoco" "$mujoco_commit"

"$source_dir/build.sh" --config Release --build_dir "$build_dir" \
  --update --build --build_shared_lib --skip_tests \
  --parallel "$jobs" --compile_no_warning_as_error \
  --cmake_extra_defines "CMAKE_INSTALL_PREFIX=$install_dir" onnxruntime_BUILD_UNIT_TESTS=OFF
cmake --install "$build_dir/Release" --prefix "$install_dir"
printf 'Installed ONNX Runtime 1.22.2 in %s\n' "$install_dir"

cmake -S "$mujoco_root/mujoco" -B "$mujoco_root/build" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--version-script=$workspace_root/scripts/mujoco.exports" \
  -DCMAKE_INSTALL_PREFIX="$mujoco_root/install" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF
cmake --build "$mujoco_root/build" --parallel "$jobs"
cmake --install "$mujoco_root/build"
printf 'Installed MuJoCo 3.3.7 in %s\n' "$mujoco_root/install"
