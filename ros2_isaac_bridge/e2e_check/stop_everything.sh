#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

echo "[E2E] Stopping bridge processes..."
sim_compose exec -T aliengo-competition bash -lc \
  "if [[ -f /tmp/e2e_isaac_controller.pid ]]; then kill \"\$(cat /tmp/e2e_isaac_controller.pid)\" >/dev/null 2>&1 || true; rm -f /tmp/e2e_isaac_controller.pid; fi" || true
ros_compose exec -T ros2-jazzy bash -lc \
  "if [[ -f /tmp/e2e_bridge_node.pid ]]; then kill \"\$(cat /tmp/e2e_bridge_node.pid)\" >/dev/null 2>&1 || true; rm -f /tmp/e2e_bridge_node.pid; fi" || true

echo "[E2E] Stopping ROS 2 Jazzy container..."
"${ROOT_DIR}/docker/ctl.sh" ros2-down || true

echo "[E2E] Stopping simulation container..."
"${ROOT_DIR}/docker/ctl.sh" down || true

cleanup_legacy_containers

echo "[E2E] Done."
