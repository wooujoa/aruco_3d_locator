#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage


DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class Aruco3DNodeZed(Node):
    def __init__(self):
        super().__init__("aruco_3d_node_zed")

        self.declare_parameter("color_topic", "/zedm/zed_node/left/image_rect_color/compressed")
        self.declare_parameter("depth_topic", "/zedm/zed_node/depth/depth_registered")
        self.declare_parameter("camera_info_topic", "/zedm/zed_node/left/camera_info")

        self.declare_parameter("point_output_topic", "/aruco/marker_3d_zed")
        self.declare_parameter("debug_output_topic", "/aruco/debug_image_zed")

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 1)
        self.declare_parameter("depth_patch_size", 5)

        self.declare_parameter("debug", True)
        self.declare_parameter("log_every_n_frames", 30)
        self.declare_parameter("warn_if_time_diff_ms", 80.0)

        self.declare_parameter("skip_if_time_diff_too_large", True)
        self.declare_parameter("max_color_depth_dt_ms", 180.0)

        self.declare_parameter("depth_buffer_size", 30)
        self.declare_parameter("max_depth_buffer_age_ms", 1000.0)

        self.declare_parameter("draw_camera_center", True)
        self.declare_parameter("camera_center_cross_size", 15)
        self.declare_parameter("output_frame_override", "")

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.point_output_topic = self.get_parameter("point_output_topic").value
        self.debug_output_topic = self.get_parameter("debug_output_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

        self.debug = bool(self.get_parameter("debug").value)
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)
        self.warn_if_time_diff_ms = float(self.get_parameter("warn_if_time_diff_ms").value)

        self.skip_if_time_diff_too_large = bool(self.get_parameter("skip_if_time_diff_too_large").value)
        self.max_color_depth_dt_ms = float(self.get_parameter("max_color_depth_dt_ms").value)

        self.depth_buffer_size = int(self.get_parameter("depth_buffer_size").value)
        self.max_depth_buffer_age_ms = float(self.get_parameter("max_depth_buffer_age_ms").value)

        self.draw_camera_center = bool(self.get_parameter("draw_camera_center").value)
        self.camera_center_cross_size = int(self.get_parameter("camera_center_cross_size").value)
        self.output_frame_override = self.get_parameter("output_frame_override").value

        dict_name = self.get_parameter("aruco_dict").value
        dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self.use_new_detector_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_new_detector_api = False

        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_buffer = deque(maxlen=self.depth_buffer_size)

        self.color_frame_count = 0
        self.depth_frame_count = 0
        self.detect_success_count = 0
        self.detect_fail_count = 0

        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_cb, qos_profile_sensor_data
        )

        self.pub_point = self.create_publisher(PointStamped, self.point_output_topic, 10)
        self.pub_debug = self.create_publisher(Image, self.debug_output_topic, 10)

        self.get_logger().info("========================================")
        self.get_logger().info("✅ ArUco 3D Node Started (ZED, nearest-depth sync)")
        self.get_logger().info(f"  max_color_depth_dt_ms : {self.max_color_depth_dt_ms}")
        self.get_logger().info(f"  depth_buffer_size     : {self.depth_buffer_size}")
        self.get_logger().info(f"  max_depth_buffer_age  : {self.max_depth_buffer_age_ms} ms")
        self.get_logger().info("========================================")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg
        if self.debug:
            self.get_logger().info(
                "✅ CameraInfo received | "
                f"frame_id={msg.header.frame_id}, "
                f"size=({msg.width}x{msg.height}), "
                f"K=[fx={msg.k[0]:.3f}, fy={msg.k[4]:.3f}, cx={msg.k[2]:.3f}, cy={msg.k[5]:.3f}]"
            )

    def depth_cb(self, msg: Image):
        self.depth_frame_count += 1
        self.depth_buffer.append(msg)
        self._prune_depth_buffer()

    def color_cb(self, msg: CompressedImage):
        self.color_frame_count += 1

        if self.camera_info is None:
            return
        if len(self.depth_buffer) == 0:
            return

        depth_msg, dt_ms = self._get_nearest_depth_msg(msg.header)
        if depth_msg is None:
            return

        if dt_ms is not None and dt_ms > self.warn_if_time_diff_ms:
            self.get_logger().warn(f"⚠️ Color/Depth timestamp diff is large: {dt_ms:.2f} ms")

        if self.skip_if_time_diff_too_large and dt_ms is not None and dt_ms > self.max_color_depth_dt_ms:
            self.get_logger().warn(
                f"⏭️ Skip publish: color-depth dt too large ({dt_ms:.2f} ms > {self.max_color_depth_dt_ms:.2f} ms)"
            )
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            if np_arr.size == 0:
                return
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding failed: {repr(e)}")
            return

        overlay = bgr.copy()
        self._draw_camera_center(overlay)

        try:
            if self.use_new_detector_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.detector_params
                )
        except Exception as e:
            self.get_logger().error(f"❌ ArUco detection failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        if ids is None or len(ids) == 0:
            self.detect_fail_count += 1
            self._publish_debug(overlay, msg.header)
            return

        try:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        except Exception:
            pass

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"❌ Depth conversion failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_marker_id:
                continue

            c = corners[idx][0]
            u = int(np.mean(c[:, 0]))
            v = int(np.mean(c[:, 1]))

            h, w = cv_depth.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                continue

            self._draw_marker_vs_camera_center(overlay, u, v, cx, cy)

            depth_m = self._median_depth_m(cv_depth, depth_msg.encoding, u, v)
            if depth_m is None:
                continue

            x_m = (u - cx) * depth_m / fx
            y_m = (v - cy) * depth_m / fy
            z_m = depth_m

            out = PointStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.output_frame_override if self.output_frame_override else depth_msg.header.frame_id
            out.point.x = float(x_m)
            out.point.y = float(y_m)
            out.point.z = float(z_m)
            self.pub_point.publish(out)

            self.detect_success_count += 1
            self.get_logger().info(
                f"📍 Published marker_3d_zed | "
                f"ID={marker_id}, pixel=({u},{v}), "
                f"xyz=({x_m:.4f}, {y_m:.4f}, {z_m:.4f}) m | "
                f"frame={out.header.frame_id}, dt={dt_ms:.2f} ms"
            )

            cv2.circle(overlay, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f"ID:{marker_id} X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f}",
                (u, max(20, v - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        self._publish_debug(overlay, msg.header)

    def _prune_depth_buffer(self):
        if len(self.depth_buffer) == 0:
            return
        newest = self._header_to_sec(self.depth_buffer[-1].header)
        while len(self.depth_buffer) > 0:
            oldest = self._header_to_sec(self.depth_buffer[0].header)
            if (newest - oldest) * 1000.0 > self.max_depth_buffer_age_ms:
                self.depth_buffer.popleft()
            else:
                break

    def _get_nearest_depth_msg(self, color_header):
        if len(self.depth_buffer) == 0:
            return None, None
        color_t = self._header_to_sec(color_header)

        best_msg = None
        best_dt = None
        for dmsg in self.depth_buffer:
            dt = abs(color_t - self._header_to_sec(dmsg.header)) * 1000.0
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_msg = dmsg
        return best_msg, best_dt

    def _header_to_sec(self, header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _draw_camera_center(self, img):
        if not self.draw_camera_center or self.camera_info is None:
            return
        cx = int(round(float(self.camera_info.k[2])))
        cy = int(round(float(self.camera_info.k[5])))
        h, w = img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return
        s = self.camera_center_cross_size
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(img, (cx - s, cy), (cx + s, cy), (0, 0, 255), 2)
        cv2.line(img, (cx, cy - s), (cx, cy + s), (0, 0, 255), 2)

    def _draw_marker_vs_camera_center(self, img, u, v, cx, cy):
        try:
            cx_i = int(round(cx))
            cy_i = int(round(cy))
            cv2.line(img, (cx_i, cy_i), (u, v), (255, 0, 255), 2)
            cv2.circle(img, (u, v), 8, (0, 255, 255), 2)
        except Exception:
            pass

    def _median_depth_m(self, cv_depth, encoding, u, v):
        try:
            r = self.depth_patch_size // 2
            h, w = cv_depth.shape[:2]

            u0 = max(0, u - r)
            u1 = min(w, u + r + 1)
            v0 = max(0, v - r)
            v1 = min(h, v + r + 1)

            patch = cv_depth[v0:v1, u0:u1]
            if patch.size == 0:
                return None

            if "16UC1" in encoding:
                vals = patch[patch > 0]
                if vals.size == 0:
                    return None
                return float(np.median(vals)) / 1000.0

            if "32FC1" in encoding:
                vals = patch[np.isfinite(patch) & (patch > 0)]
                if vals.size == 0:
                    return None
                return float(np.median(vals))

            return None
        except Exception:
            return None

    def _publish_debug(self, img, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ Debug image publish failed: {repr(e)}")


def main():
    rclpy.init()
    node = Aruco3DNodeZed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()