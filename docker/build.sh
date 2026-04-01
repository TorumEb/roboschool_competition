#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

docker build -t aliengo-base docker/base
docker build -t aliengo-isaac-gym docker/isaac-gym
docker build -t aliengo-competition -f docker/Dockerfile .
