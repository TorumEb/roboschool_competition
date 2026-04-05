#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RVIZ_CONFIG="/workspace/aliengo_competition/ros2_isaac_bridge/e2e_check/camera_topics.rviz"
OPEN_RVIZ=1
DO_BUILD=1

for arg in "$@"; do
  case "${arg}" in
    --no-rviz)
      OPEN_RVIZ=0
      ;;
    --skip-build)
      DO_BUILD=0
      ;;
    *)
      echo "Unknown argument: ${arg}"
      echo "Usage: run_everything.sh [--skip-build] [--no-rviz]"
      exit 1
      ;;
  esac
done

sim_compose() {
  docker compose -p aliengo-sim -f "${ROOT_DIR}/docker/compose.local.yml" -f "${ROOT_DIR}/docker/compose.viz.yml" "$@"
}

ros_compose() {
  docker compose -p aliengo-ros2 -f "${ROOT_DIR}/docker/compose.ros2.yml" "$@"
}

cleanup_legacy_containers() {
  docker ps -aq --filter "name=aliengo-competition" | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker ps -aq --filter "name=ros2-jazzy" | xargs -r docker rm -f >/dev/null 2>&1 || true
}

show_logs() {
  echo
  echo "[E2E] ---- Isaac Controller Log (tail) ----"
  sim_compose exec -T aliengo-competition bash -lc "tail -n 80 /tmp/e2e_isaac_controller.log || true"
  echo
  echo "[E2E] ---- ROS Bridge Log (tail) ----"
  ros_compose exec -T ros2-jazzy bash -lc "tail -n 80 /tmp/e2e_bridge_node.log || true"
}

echo "[E2E] 1/7 Starting simulation container (with visualization)..."
cleanup_legacy_containers
if [[ "${DO_BUILD}" -eq 1 ]]; then
  "${ROOT_DIR}/docker/ctl.sh" up
else
  sim_compose up -d
fi

echo "[E2E] 2/7 Building ROS 2 Jazzy container..."
if [[ "${DO_BUILD}" -eq 1 ]]; then
  "${ROOT_DIR}/docker/ctl.sh" ros2-build
else
  echo "[E2E] Skipping image build (--skip-build)."
fi

echo "[E2E] 3/7 Starting ROS 2 Jazzy container..."
"${ROOT_DIR}/docker/ctl.sh" ros2-up

echo "[E2E] 4/7 Restarting sim-side controller..."
sim_compose exec -T aliengo-competition bash -lc \
  "if [[ -f /tmp/e2e_isaac_controller.pid ]]; then kill \"\$(cat /tmp/e2e_isaac_controller.pid)\" >/dev/null 2>&1 || true; rm -f /tmp/e2e_isaac_controller.pid; fi"
sim_compose exec -T aliengo-competition bash -lc \
  "cd /workspace/aliengo_competition && nohup python ros2_isaac_bridge/sim_side/isaac_controller.py >/tmp/e2e_isaac_controller.log 2>&1 & echo \$! >/tmp/e2e_isaac_controller.pid"

echo "[E2E] 5/7 Restarting ROS bridge node..."
ros_compose exec -T ros2-jazzy bash -lc \
  "if [[ -f /tmp/e2e_bridge_node.pid ]]; then kill \"\$(cat /tmp/e2e_bridge_node.pid)\" >/dev/null 2>&1 || true; rm -f /tmp/e2e_bridge_node.pid; fi"
ros_compose exec -T ros2-jazzy bash -lc \
  "cd /workspace/aliengo_competition && nohup bash ros2_isaac_bridge/run_bridge_node.sh >/tmp/e2e_bridge_node.log 2>&1 & echo \$! >/tmp/e2e_bridge_node.pid"

echo "[E2E] 6/7 Waiting for data and checking camera topics..."
sleep 10

if ! ros_compose exec -T ros2-jazzy bash -lc \
  "set -e; set +u; source /opt/ros/jazzy/setup.bash; set -u; timeout 180 ros2 topic echo /aliengo/camera/color/image_raw --once >/tmp/e2e_camera_color_msg.txt"; then
  echo "[E2E] Color camera topic check failed."
  show_logs
  exit 1
fi

if ! ros_compose exec -T ros2-jazzy bash -lc \
  "set -e; set +u; source /opt/ros/jazzy/setup.bash; set -u; timeout 60 ros2 topic echo /aliengo/camera/depth/image_raw --once >/tmp/e2e_camera_depth_msg.txt"; then
  echo "[E2E] Depth camera topic check failed."
  show_logs
  exit 1
fi

if [[ "${OPEN_RVIZ}" -eq 1 ]]; then
  echo "[E2E] 7/7 Camera topics are alive. Opening RViz..."
  ros_compose exec ros2-jazzy bash -lc \
    "set -e; set +u; source /opt/ros/jazzy/setup.bash; set -u; rviz2 -d '${RVIZ_CONFIG}'"
else
  echo "[E2E] 7/7 Camera topics are alive. RViz skipped (--no-rviz)."
fi
