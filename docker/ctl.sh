#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.local.yml"

usage() {
  cat <<'EOF'
Usage:
  docker/ctl.sh build   # build local docker layers and the competition image
  docker/ctl.sh up      # build then start the competition container detached
  docker/ctl.sh down    # stop and remove the competition container
  docker/ctl.sh enter   # open a shell inside the running container
  docker/ctl.sh logs    # follow container logs
EOF
}

build_layers() {
  bash "${SCRIPT_DIR}/build.sh"
}

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

cmd="${1:-}"
case "${cmd}" in
  build)
    build_layers
    compose build
    ;;
  up)
    build_layers
    compose up -d
    ;;
  down)
    compose down
    ;;
  enter)
    compose exec aliengo-competition bash
    ;;
  logs)
    compose logs -f aliengo-competition
    ;;
  *)
    usage
    exit 1
    ;;
esac
