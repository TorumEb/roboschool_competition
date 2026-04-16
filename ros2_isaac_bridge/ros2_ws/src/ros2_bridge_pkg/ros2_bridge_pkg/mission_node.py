"""
ROS2 Mission Node — semantic-frontier exploration pipeline.

Builds an occupancy grid from depth, detects all object classes,
plans paths to frontiers or known objects via A*, and follows
a strict task-order FSM.
"""

import json
import math
import os
from enum import Enum, auto

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32

from ros2_bridge_pkg.reference_detector import (
    ReferenceImageDetectorBackend,
    Detection,
)
from ros2_bridge_pkg.local_map import OccupancyGrid, SemanticObjectMap
from ros2_bridge_pkg.perception import (
    MultiClassScanner,
    ConfirmationBuffer,
    LocalizedDetection,
    bbox_to_world_pos,
    _median_depth_at_bbox,
    ALL_OBJECTS,
    RGB_W,
    RGB_CX,
    RGB_FX,
)
from ros2_bridge_pkg.frontier_explorer import select_best_frontier
from ros2_bridge_pkg.path_planner import (
    PlannedPath,
    PathFollower,
    plan_path,
    obstacle_safety,
)


class State(Enum):
    WAIT_MISSION = auto()
    EXPLORE = auto()
    NAV_TO_OBJECT = auto()
    CONFIRM = auto()
    APPROACH = auto()
    REPORT = auto()
    WAIT_NEXT = auto()
    FINISHED = auto()


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        # ---- ROS parameters ----
        self.declare_parameter("refs_dir", "")
        self.declare_parameter("debug", False)
        self.declare_parameter("align_pixel_tolerance", 40.0)
        self.declare_parameter("align_kp", 0.005)
        self.declare_parameter("approach_distance_threshold", 0.4)
        self.declare_parameter("approach_max_vx", 1.0)
        self.declare_parameter("approach_kp_dist", 0.4)
        self.declare_parameter("max_vx", 1.5)
        self.declare_parameter("max_wz", 1.5)
        self.declare_parameter("min_detection_score", 0.12)
        self.declare_parameter("approach_lost_timeout_s", 5.0)
        self.declare_parameter("confirm_frames", 3)
        self.declare_parameter("confirm_visual_timeout_s", 10.0)
        self.declare_parameter("grid_size_m", 60.0)
        self.declare_parameter("grid_resolution", 0.5)
        self.declare_parameter("path_replan_interval_s", 3.0)
        self.declare_parameter("nav_arrive_threshold_m", 1.5)
        self.declare_parameter("explore_vx", 1.0)
        self.declare_parameter("fallback_wz", 0.5)

        self.refs_dir = self.get_parameter("refs_dir").value
        self.debug = self.get_parameter("debug").value
        self.align_tol = self.get_parameter("align_pixel_tolerance").value
        self.align_kp = self.get_parameter("align_kp").value
        self.approach_dist = self.get_parameter("approach_distance_threshold").value
        self.approach_max_vx = self.get_parameter("approach_max_vx").value
        self.approach_kp_dist = self.get_parameter("approach_kp_dist").value
        self.max_vx = self.get_parameter("max_vx").value
        self.max_wz = self.get_parameter("max_wz").value
        self.min_score = self.get_parameter("min_detection_score").value
        self.approach_lost_timeout = self.get_parameter("approach_lost_timeout_s").value
        self.confirm_frames_n = self.get_parameter("confirm_frames").value
        self.confirm_visual_timeout = self.get_parameter("confirm_visual_timeout_s").value
        self.grid_size = self.get_parameter("grid_size_m").value
        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.replan_interval = self.get_parameter("path_replan_interval_s").value
        self.nav_arrive_thresh = self.get_parameter("nav_arrive_threshold_m").value
        self.explore_vx = self.get_parameter("explore_vx").value
        self.fallback_wz = self.get_parameter("fallback_wz").value

        if not self.refs_dir:
            for candidate in [
                "/workspace/aliengo_competition/resources/assets/objects",
                os.path.expanduser(
                    "~/workspace/roboschool_competition/resources/assets/objects"
                ),
            ]:
                if os.path.isdir(candidate):
                    self.refs_dir = candidate
                    break

        self.get_logger().info(f"refs_dir = {self.refs_dir}")
        self.get_logger().info(f"debug    = {self.debug}")

        # ---- Detector + perception ----
        self.detector = ReferenceImageDetectorBackend(
            refs_dir=self.refs_dir,
            min_matches=10,
            min_inliers=8,
            ransac_reproj_threshold=4.0,
            min_bbox_area=600,
            debug_visualization=self.debug,
        )
        self.scanner = MultiClassScanner(self.detector, self.min_score)
        self.confirm_buf = ConfirmationBuffer(
            required=self.confirm_frames_n,
        )

        # ---- Mapping ----
        self.occ_grid = OccupancyGrid(
            size_m=self.grid_size,
            resolution=self.grid_resolution,
        )
        self.sem_map = SemanticObjectMap()

        # ---- Path planner + follower ----
        self.path_follower = PathFollower(
            grid=self.occ_grid,
            lookahead_cells=5,
            heading_kp=2.5,
            max_vx=self.explore_vx,
        )
        self.current_path: PlannedPath | None = None
        self._last_plan_time = 0.0

        # ---- Publishers ----
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.det_pub = self.create_publisher(
            Int32, "/aliengo/detected_object_id", 10,
        )

        # ---- Subscribers ----
        self.create_subscription(
            Image, "/aliengo/camera/color/image_raw", self._rgb_cb, 10,
        )
        self.create_subscription(
            Image, "/aliengo/camera/depth/image_raw", self._depth_cb, 10,
        )
        self.create_subscription(
            TwistStamped, "/aliengo/base_velocity", self._vel_cb, 10,
        )
        self.create_subscription(
            String, "/aliengo/mission_queue", self._queue_cb, 10,
        )
        self.create_subscription(
            Int32, "/aliengo/current_target_id", self._target_cb, 10,
        )
        self.create_subscription(
            String, "/aliengo/mission_status", self._status_cb, 10,
        )

        # ---- Cached sensor data ----
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None

        # ---- Odometry (dead reckoning) ----
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_heading = 0.0
        self._odom_last_t: float | None = None

        # ---- Mission state ----
        self.mission_queue: list | None = None
        self.sim_target_id: int = -1
        self.mission_status: dict = {}
        self.id_to_name: dict = {}

        # ---- FSM ----
        self.state = State.WAIT_MISSION
        self.last_detection: Detection | None = None
        self.last_detection_time = 0.0
        self._last_approach_depth: float | None = None
        self.report_sent_id: int | None = None
        self.report_time = 0.0
        self._confirm_enter_time = 0.0
        self._fallback_dir = 1.0
        self._tick_count = 0

        self._stuck_check_x = 0.0
        self._stuck_check_y = 0.0
        self._stuck_check_time = 0.0
        self._stuck_backing = False
        self._stuck_back_until = 0.0

        self._cb_counts = {
            "rgb": 0, "depth": 0, "vel": 0,
            "queue": 0, "target": 0, "status": 0,
        }

        if self.debug:
            os.makedirs("debug_mission", exist_ok=True)
            self._dbg_frame = 0

        self.create_timer(0.1, self._tick)
        self.get_logger().info("Mission node started (semantic frontier mode).")

    # ==================================================================
    # Callbacks
    # ==================================================================
    def _rgb_cb(self, msg: Image):
        self._cb_counts["rgb"] += 1
        try:
            self.latest_rgb = np.frombuffer(
                msg.data, dtype=np.uint8,
            ).reshape((msg.height, msg.width, 3))
        except ValueError:
            pass

    def _depth_cb(self, msg: Image):
        self._cb_counts["depth"] += 1
        try:
            self.latest_depth = np.frombuffer(
                msg.data, dtype=np.float32,
            ).reshape((msg.height, msg.width))
        except ValueError:
            pass

    def _vel_cb(self, msg: TwistStamped):
        self._cb_counts["vel"] += 1
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        wz = msg.twist.angular.z

        now = self._now()
        if self._odom_last_t is not None:
            dt = now - self._odom_last_t
            if 0.0 < dt < 0.5:
                self._odom_heading += wz * dt
                c = math.cos(self._odom_heading)
                s = math.sin(self._odom_heading)
                self._odom_x += (vx * c - vy * s) * dt
                self._odom_y += (vx * s + vy * c) * dt
        self._odom_last_t = now

    def _queue_cb(self, msg: String):
        self._cb_counts["queue"] += 1
        if self._cb_counts["queue"] == 1:
            self.get_logger().info(f"[CB] First mission_queue: {msg.data[:120]}")
        try:
            parsed = json.loads(msg.data)
            if isinstance(parsed, list) and len(parsed) > 0:
                self.mission_queue = parsed
                self.id_to_name = {
                    int(item[0]): str(item[1]) for item in parsed
                }
        except (json.JSONDecodeError, IndexError) as e:
            self.get_logger().warn(f"[CB] queue parse error: {e}")

    def _target_cb(self, msg: Int32):
        self._cb_counts["target"] += 1
        if self._cb_counts["target"] == 1:
            self.get_logger().info(f"[CB] First target_id: {msg.data}")
        self.sim_target_id = msg.data

    def _status_cb(self, msg: String):
        self._cb_counts["status"] += 1
        try:
            self.mission_status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    # ==================================================================
    # Helpers
    # ==================================================================
    def _send_cmd(self, vx: float, vy: float, wz: float):
        now = self._now()
        if self._stuck_backing and now < self._stuck_back_until:
            vx, vy, wz = -0.4, 0.0, 0.7
        elif self._stuck_backing:
            self._stuck_backing = False
            self._stuck_check_x = self._odom_x
            self._stuck_check_y = self._odom_y
            self._stuck_check_time = now

        vx = max(-self.max_vx, min(self.max_vx, vx))
        vy = max(-self.max_vx, min(self.max_vx, vy))
        wz = max(-self.max_wz, min(self.max_wz, wz))
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _check_stuck(self):
        """If robot barely moved in 3s while exploring, back up."""
        now = self._now()
        if self._stuck_backing:
            return
        elapsed = now - self._stuck_check_time
        if elapsed < 3.0:
            return
        moved = math.hypot(
            self._odom_x - self._stuck_check_x,
            self._odom_y - self._stuck_check_y,
        )
        if moved < 0.15:
            self.get_logger().warn(
                f"[STUCK] Moved only {moved:.2f}m in {elapsed:.0f}s, backing up"
            )
            self._stuck_backing = True
            self._stuck_back_until = now + 1.5
        self._stuck_check_x = self._odom_x
        self._stuck_check_y = self._odom_y
        self._stuck_check_time = now

    def _stop(self):
        self._send_cmd(0.0, 0.0, 0.0)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _forward_depth(self) -> float | None:
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        cy1, cy2 = int(h * 0.25), int(h * 0.75)
        cx1, cx2 = int(w * 0.3), int(w * 0.7)
        roi = np.abs(self.latest_depth[cy1:cy2, cx1:cx2])
        valid = roi[(roi > 0.05) & (roi < 20.0) & np.isfinite(roi)]
        if valid.size < 4:
            close = roi[(roi > 0.001) & (roi <= 0.05)]
            if close.size > 10:
                return 0.05
            return None
        return float(np.percentile(valid, 15))

    def _estimate_depth_det(self, det: Detection) -> float | None:
        if self.latest_depth is None:
            return None
        result = _median_depth_at_bbox(self.latest_depth, det.bbox_xyxy)
        return result

    # ==================================================================
    # Per-tick perception + mapping (runs every state)
    # ==================================================================
    def _update_world_model(self) -> list[LocalizedDetection]:
        """Update occupancy grid, run multi-class scan, update semantic map."""
        self.occ_grid.mark_visited(self._odom_x, self._odom_y)

        if self.latest_depth is not None:
            self.occ_grid.update_from_depth(
                self.latest_depth,
                self._odom_x, self._odom_y, self._odom_heading,
            )

        detections: list[LocalizedDetection] = []
        if self.latest_rgb is not None:
            priority = (
                [self.sim_target_id] if self.sim_target_id >= 0 else None
            )
            detections = self.scanner.scan(
                self.latest_rgb,
                self.latest_depth,
                self._odom_x, self._odom_y, self._odom_heading,
                priority_ids=priority,
            )
            now = self._now()
            for ld in detections:
                if math.isfinite(ld.world_x):
                    self.sem_map.update_detection(
                        ld.object_id, ld.detection.class_name,
                        ld.world_x, ld.world_y, now,
                    )

        return detections

    def _find_target_in_detections(
        self, detections: list[LocalizedDetection],
    ) -> LocalizedDetection | None:
        for ld in detections:
            if ld.object_id == self.sim_target_id:
                return ld
        return None

    # ==================================================================
    # Main tick
    # ==================================================================
    def _tick(self):
        self._tick_count += 1
        self._check_stuck()

        detections = self._update_world_model()

        if self._tick_count % 100 == 0:
            sem_found = list(self.sem_map.instances.keys())
            self.get_logger().info(
                f"[NAV] state={self.state.name} "
                f"pos=({self._odom_x:.1f},{self._odom_y:.1f}) "
                f"hdg={math.degrees(self._odom_heading):.0f}° "
                f"sem_objects={sem_found} "
                f"target={self.sim_target_id}"
            )

        dispatch = {
            State.WAIT_MISSION: self._tick_wait,
            State.EXPLORE: self._tick_explore,
            State.NAV_TO_OBJECT: self._tick_nav_to_object,
            State.CONFIRM: self._tick_confirm,
            State.APPROACH: self._tick_approach,
            State.REPORT: self._tick_report,
            State.WAIT_NEXT: self._tick_wait_next,
            State.FINISHED: self._tick_finished,
        }
        dispatch[self.state](detections)

    # ==================================================================
    # FSM states
    # ==================================================================

    # ---- WAIT_MISSION ----
    def _tick_wait(self, detections):
        self._stop()
        if self.mission_queue is not None and self.sim_target_id >= 0:
            self.get_logger().info(
                f"Mission received. Queue={self.mission_queue}, "
                f"first target={self.sim_target_id}"
            )
            self._transition(State.EXPLORE)

    # ---- EXPLORE ----
    def _tick_explore(self, detections):
        target_det = self._find_target_in_detections(detections)
        if target_det is not None:
            self.last_detection = target_det.detection
            self.last_detection_time = self._now()
            self.confirm_buf.push(target_det)

        inst = self.sem_map.get(self.sim_target_id)
        if inst is not None and inst.confirmed:
            self.get_logger().info(
                f"[EXPLORE] Target {self.sim_target_id} "
                f"({inst.class_name}) found on map at "
                f"({inst.world_x:.1f},{inst.world_y:.1f}), navigating"
            )
            self.current_path = plan_path(
                self.occ_grid,
                self._odom_x, self._odom_y,
                inst.world_x, inst.world_y,
                goal_type="object",
                now=self._now(),
            )
            self._transition(State.NAV_TO_OBJECT)
            return

        if target_det is not None and self.confirm_buf.push(target_det):
            self.get_logger().info(
                f"[EXPLORE] Target {self.sim_target_id} confirmed visually, "
                f"switching to APPROACH"
            )
            self._transition(State.APPROACH)
            return

        now = self._now()
        need_replan = (
            self.current_path is None
            or self.current_path.is_empty()
            or self.path_follower.is_path_complete(
                self.current_path, self._odom_x, self._odom_y,
            )
            or now - self._last_plan_time > self.replan_interval
        )

        if need_replan:
            frontier = select_best_frontier(
                self.occ_grid,
                self._odom_x, self._odom_y,
                self.sem_map,
                self.sim_target_id,
            )
            if frontier is not None:
                self.current_path = plan_path(
                    self.occ_grid,
                    self._odom_x, self._odom_y,
                    frontier.centroid_wx, frontier.centroid_wy,
                    goal_type="frontier",
                    now=now,
                )
                self._last_plan_time = now
                if self._tick_count % 50 == 0:
                    self.get_logger().info(
                        f"[EXPLORE] Frontier at "
                        f"({frontier.centroid_wx:.1f},{frontier.centroid_wy:.1f}) "
                        f"score={frontier.score:.1f}"
                    )
            else:
                self._send_cmd(0.0, 0.0, self.fallback_wz * self._fallback_dir)
                self._fallback_dir *= -1
                return

        if self.current_path is not None and not self.current_path.is_empty():
            vx, vy, wz = self.path_follower.follow(
                self.current_path,
                self._odom_x, self._odom_y, self._odom_heading,
            )
            fwd = self._forward_depth()
            vx, vy, wz = obstacle_safety(vx, vy, wz, fwd)
            self._send_cmd(vx, vy, wz)
        else:
            self._send_cmd(0.0, 0.0, self.fallback_wz)

    # ---- NAV_TO_OBJECT ----
    def _tick_nav_to_object(self, detections):
        target_det = self._find_target_in_detections(detections)
        if target_det is not None:
            self.last_detection = target_det.detection
            self.last_detection_time = self._now()
            self.confirm_buf.push(target_det)

        inst = self.sem_map.get(self.sim_target_id)
        if inst is None:
            self.get_logger().warn("[NAV] Target removed from map, back to EXPLORE")
            self._transition(State.EXPLORE)
            return

        dist_to_goal = math.hypot(
            inst.world_x - self._odom_x,
            inst.world_y - self._odom_y,
        )
        if dist_to_goal < self.nav_arrive_thresh:
            self.get_logger().info(
                f"[NAV] Arrived near target ({dist_to_goal:.1f}m), confirming"
            )
            self._transition(State.CONFIRM)
            return

        if target_det is not None and self.confirm_buf.push(target_det):
            self.get_logger().info("[NAV] Target confirmed visually en route")
            self._transition(State.APPROACH)
            return

        now = self._now()
        if (
            self.current_path is None
            or self.current_path.is_empty()
            or self.path_follower.is_path_complete(
                self.current_path, self._odom_x, self._odom_y,
            )
            or now - self._last_plan_time > self.replan_interval
        ):
            self.current_path = plan_path(
                self.occ_grid,
                self._odom_x, self._odom_y,
                inst.world_x, inst.world_y,
                goal_type="object",
                now=now,
            )
            self._last_plan_time = now

        if self.current_path is not None and not self.current_path.is_empty():
            vx, vy, wz = self.path_follower.follow(
                self.current_path,
                self._odom_x, self._odom_y, self._odom_heading,
            )
            fwd = self._forward_depth()
            vx, vy, wz = obstacle_safety(vx, vy, wz, fwd)
            self._send_cmd(vx, vy, wz)
        else:
            heading_to = math.atan2(
                inst.world_y - self._odom_y,
                inst.world_x - self._odom_x,
            )
            err = _angle_diff(heading_to, self._odom_heading)
            vx, vy, wz = 0.3, 0.0, 2.0 * err
            fwd = self._forward_depth()
            vx, vy, wz = obstacle_safety(vx, vy, wz, fwd)
            self._send_cmd(vx, vy, wz)

    # ---- CONFIRM ----
    def _tick_confirm(self, detections):
        target_det = self._find_target_in_detections(detections)

        if target_det is not None:
            self.last_detection = target_det.detection
            self.last_detection_time = self._now()
            if self.confirm_buf.push(target_det):
                self.get_logger().info("[CONFIRM] Visual confirmation OK")
                self._transition(State.APPROACH)
                return

        inst = self.sem_map.get(self.sim_target_id)
        if inst is not None:
            heading_to = math.atan2(
                inst.world_y - self._odom_y,
                inst.world_x - self._odom_x,
            )
            err = _angle_diff(heading_to, self._odom_heading)
            vx, vy, wz = 0.15, 0.0, 2.0 * err
            fwd = self._forward_depth()
            vx, vy, wz = obstacle_safety(
                vx, vy, wz, fwd,
                soft_threshold=1.15,
                hard_threshold=0.42,
                reverse_threshold=0.22,
            )
            self._send_cmd(vx, vy, wz)
        else:
            self._send_cmd(0.0, 0.0, 0.4)

        if self._now() - self._confirm_enter_time > self.confirm_visual_timeout:
            self.get_logger().warn(
                "[CONFIRM] Timeout — object not visible, removing from map"
            )
            self.sem_map.remove(self.sim_target_id)
            self._transition(State.EXPLORE)

    # ---- APPROACH ----
    def _tick_approach(self, detections):
        target_det = self._find_target_in_detections(detections)
        if target_det is not None:
            self.last_detection = target_det.detection
            self.last_detection_time = self._now()
            depth_now = self._estimate_depth_det(target_det.detection)
            if depth_now is not None:
                self._last_approach_depth = depth_now
        elif self._now() - self.last_detection_time > self.approach_lost_timeout:
            self.get_logger().warn("[APPROACH] Target lost, back to EXPLORE")
            self._last_approach_depth = None
            self._transition(State.EXPLORE)
            return

        if self.last_detection is None:
            self._stop()
            return

        x1, _, x2, _ = self.last_detection.bbox_xyxy
        bbox_cx = (x1 + x2) / 2.0
        error_x = bbox_cx - RGB_CX
        wz = -self.align_kp * error_x

        dist = self._estimate_depth_det(self.last_detection)
        if dist is None:
            dist = self._last_approach_depth

        if dist is not None and self._tick_count % 10 == 0:
            self.get_logger().info(
                f"[APPROACH] dist={dist:.2f}m  threshold={self.approach_dist:.2f}m"
            )

        if dist is None:
            vx, vy, fwd_v = 0.3, 0.0, self._forward_depth()
            vx, _, wz = obstacle_safety(
                vx, vy, wz, fwd_v,
                soft_threshold=1.05,
                hard_threshold=0.38,
                reverse_threshold=0.2,
            )
            self._send_cmd(vx, 0.0, wz)
            return

        if dist <= self.approach_dist:
            self._stop()
            self.get_logger().info(
                f"[APPROACH] Close enough: dist={dist:.2f}m"
            )
            self._transition(State.REPORT)
            return

        vx = min(self.approach_max_vx,
                 self.approach_kp_dist * (dist - self.approach_dist))
        vx = max(0.15, vx)
        fwd_v = self._forward_depth()
        vx, _, wz = obstacle_safety(
            vx, 0.0, wz, fwd_v,
            soft_threshold=1.05,
            hard_threshold=0.38,
            reverse_threshold=0.2,
        )
        self._send_cmd(vx, 0.0, wz)

    # ---- REPORT ----
    def _tick_report(self, detections):
        self._stop()
        now = self._now()

        if not hasattr(self, '_report_enter_time') or self._report_enter_time == 0.0:
            self._report_enter_time = now
            self.get_logger().info("[REPORT] Standing near target for 2s...")
            return

        if now - self._report_enter_time < 2.0:
            return

        target_id = self.sim_target_id
        target_name = self.id_to_name.get(target_id, "?")

        msg = Int32()
        msg.data = target_id
        self.det_pub.publish(msg)

        self.report_sent_id = target_id
        self.report_time = now
        self._report_enter_time = 0.0
        self.scanner.mark_found(target_id)
        self.get_logger().info(
            f"[REPORT] Reported object_id={target_id} ({target_name})"
        )
        self._transition(State.WAIT_NEXT)

    # ---- WAIT_NEXT ----
    def _tick_wait_next(self, detections):
        self._stop()

        if self.mission_status.get("finished", False):
            self._transition(State.FINISHED)
            return

        if self.sim_target_id != self.report_sent_id:
            new_name = self.id_to_name.get(self.sim_target_id, "?")
            self.get_logger().info(
                f"[WAIT] Next target: id={self.sim_target_id} ({new_name})"
            )
            self.last_detection = None
            self._transition(State.EXPLORE)
            return

        if self._now() - self.report_time > 5.0:
            self.get_logger().warn("[WAIT] Timeout, re-publishing report")
            msg = Int32()
            msg.data = self.report_sent_id
            self.det_pub.publish(msg)
            self.report_time = self._now()

    # ---- FINISHED ----
    def _tick_finished(self, detections):
        self._stop()

    # ==================================================================
    # Transition helper
    # ==================================================================
    def _transition(self, new_state: State):
        self.get_logger().info(f"FSM: {self.state.name} -> {new_state.name}")
        if new_state == State.EXPLORE:
            self.current_path = None
            self._last_plan_time = 0.0
            self.confirm_buf.reset()
        if new_state == State.CONFIRM:
            self._confirm_enter_time = self._now()
            self.confirm_buf.reset()
        if new_state == State.APPROACH:
            self._last_approach_depth = None
        self.state = new_state


def _angle_diff(target: float, current: float) -> float:
    d = target - current
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
