FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_WS=/workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-opencv \
    python3-pip \
    python3-numpy \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-rqt \
    ros-jazzy-rqt-common-plugins \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/aliengo_competition

CMD ["bash", "-lc", "tail -f /dev/null"]
