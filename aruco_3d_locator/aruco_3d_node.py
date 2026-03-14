#!/usr/bin/env python3
import cv2
import numpy as np

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


class Aruco3DNode(Node):
    def __init__(self):
        super().__init__("aruco_3d_node")

        # ===== Parameters =====
        self.declare_parameter(
            "color_topic",
            "/camera_r/camera_r/color/image_rect_raw/compressed"
        )
        self.declare_parameter(
            "depth_topic",
            "/camera_r/camera_r/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera_r/camera_r/aligned_depth_to_color/camera_info"
        )

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 0)
        self.declare_parameter("depth_patch_size", 5)

        # 디버그 파라미터
        self.declare_parameter("debug", True)
        self.declare_parameter("log_every_n_frames", 30)
        self.declare_parameter("warn_if_time_diff_ms", 100.0)

        # optical center 표시 관련
        self.declare_parameter("draw_camera_center", True)
        self.declare_parameter("camera_center_cross_size", 15)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

        self.debug = bool(self.get_parameter("debug").value)
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)
        self.warn_if_time_diff_ms = float(self.get_parameter("warn_if_time_diff_ms").value)

        self.draw_camera_center = bool(self.get_parameter("draw_camera_center").value)
        self.camera_center_cross_size = int(self.get_parameter("camera_center_cross_size").value)

        dict_name = self.get_parameter("aruco_dict").value
        dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        # ===== Detector API compatibility =====
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.detector_params
            )
            self.use_new_detector_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_new_detector_api = False

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.camera_info = None

        self.color_frame_count = 0
        self.depth_frame_count = 0
        self.detect_success_count = 0
        self.detect_fail_count = 0

        # ===== Subscriptions =====
        self.sub_color = self.create_subscription(
            CompressedImage,
            self.color_topic,
            self.color_cb,
            qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_cb,
            qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.info_cb,
            qos_profile_sensor_data
        )

        # ===== Publishers =====
        self.pub_point = self.create_publisher(PointStamped, "/aruco/marker_3d", 10)
        self.pub_debug = self.create_publisher(Image, "/aruco/debug_image", 10)

        self.get_logger().info("========================================")
        self.get_logger().info("✅ ArUco 3D Node Started")
        self.get_logger().info(f"  color_topic         : {self.color_topic}")
        self.get_logger().info(f"  depth_topic         : {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic   : {self.camera_info_topic}")
        self.get_logger().info(f"  aruco_dict          : {dict_name}")
        self.get_logger().info(f"  target_marker_id    : {self.target_marker_id}")
        self.get_logger().info(f"  depth_patch_size    : {self.depth_patch_size}")
        self.get_logger().info(f"  draw_camera_center  : {self.draw_camera_center}")
        self.get_logger().info(f"  debug               : {self.debug}")
        self.get_logger().info(
            f"  detector_api        : {'new' if self.use_new_detector_api else 'legacy'}"
        )
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
        self.latest_depth_msg = msg
        self.depth_frame_count += 1

        if self.debug and (self.depth_frame_count % self.log_every_n_frames == 0):
            self.get_logger().info(
                "📥 Depth frame received | "
                f"count={self.depth_frame_count}, "
                f"encoding={msg.encoding}, "
                f"size=({msg.width}x{msg.height}), "
                f"frame_id={msg.header.frame_id}"
            )

    def color_cb(self, msg: CompressedImage):
        self.color_frame_count += 1

        if self.latest_depth_msg is None:
            if self.debug and (self.color_frame_count % self.log_every_n_frames == 0):
                self.get_logger().warn("⏳ Skip color frame: latest_depth_msg is None")
            return

        if self.camera_info is None:
            if self.debug and (self.color_frame_count % self.log_every_n_frames == 0):
                self.get_logger().warn("⏳ Skip color frame: camera_info is None")
            return

        if self.debug and (self.color_frame_count % self.log_every_n_frames == 0):
            self.get_logger().info(
                "📷 Color frame received | "
                f"count={self.color_frame_count}, "
                f"format={msg.format}, "
                f"data_size={len(msg.data)}, "
                f"frame_id={msg.header.frame_id}"
            )

        # Color-Depth timestamp 차이 체크
        self._log_time_diff_if_needed(msg, self.latest_depth_msg)

        # 1) Compressed image decode
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            if np_arr.size == 0:
                self.get_logger().error("❌ Decode failed: compressed color buffer is empty")
                return

            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().error(
                    "❌ Decode failed: cv2.imdecode returned None "
                    f"(format={msg.format}, data_size={len(msg.data)})"
                )
                return

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            self.get_logger().error(f"❌ Decoding failed: {repr(e)}")
            return

        overlay = bgr.copy()

        # optical center 먼저 표시
        self._draw_camera_center(overlay)

        # 2) ArUco detection
        try:
            if self.use_new_detector_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.detector_params
                )
        except Exception as e:
            self.get_logger().error(f"❌ ArUco detection failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        if ids is None or len(ids) == 0:
            self.detect_fail_count += 1

            if self.debug and (self.detect_fail_count % self.log_every_n_frames == 0):
                self.get_logger().info(
                    "🔍 No markers detected | "
                    f"color_count={self.color_frame_count}, "
                    f"detect_fail_count={self.detect_fail_count}"
                )

            self._publish_debug(overlay, msg.header)
            return

        try:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        except Exception as e:
            self.get_logger().warn(f"⚠️ drawDetectedMarkers failed: {repr(e)}")

        detected_ids = ids.flatten().tolist()
        if self.debug:
            self.get_logger().info(f"✅ Detected marker IDs: {detected_ids}")

        # 3) Depth image conversion
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth_msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(
                "❌ Depth conversion failed in CvBridge: "
                f"{repr(e)} | encoding={self.latest_depth_msg.encoding}"
            )
            self._publish_debug(overlay, msg.header)
            return

        if cv_depth is None:
            self.get_logger().error("❌ Depth conversion failed: cv_depth is None")
            self._publish_debug(overlay, msg.header)
            return

        if len(cv_depth.shape) < 2:
            self.get_logger().error(
                f"❌ Depth image shape invalid: shape={cv_depth.shape}"
            )
            self._publish_debug(overlay, msg.header)
            return

        # 4) Camera intrinsics
        try:
            fx = float(self.camera_info.k[0])
            fy = float(self.camera_info.k[4])
            cx = float(self.camera_info.k[2])
            cy = float(self.camera_info.k[5])

            if fx == 0.0 or fy == 0.0:
                self.get_logger().error(f"❌ Invalid intrinsics: fx={fx}, fy={fy}")
                self._publish_debug(overlay, msg.header)
                return
        except Exception as e:
            self.get_logger().error(f"❌ CameraInfo parse failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        # 5) Process target marker only
        found_target = False

        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_marker_id:
                continue

            found_target = True

            try:
                c = corners[idx][0]
                u = int(np.mean(c[:, 0]))
                v = int(np.mean(c[:, 1]))
            except Exception as e:
                self.get_logger().error(
                    f"❌ Marker center computation failed for ID={marker_id}: {repr(e)}"
                )
                continue

            h, w = cv_depth.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                self.get_logger().warn(
                    f"⚠️ Marker center out of depth bounds: "
                    f"(u,v)=({u},{v}), depth_size=({w},{h})"
                )
                continue

            # optical center와 marker center 선 연결
            self._draw_marker_vs_camera_center(overlay, u, v, cx, cy)

            # 6) Depth extraction
            depth_m = self._median_depth_m(
                cv_depth,
                self.latest_depth_msg.encoding,
                u,
                v
            )

            if depth_m is None:
                self.get_logger().warn(
                    f"⚠️ No valid depth at marker center | ID={marker_id}, (u,v)=({u},{v}), "
                    f"encoding={self.latest_depth_msg.encoding}"
                )
                cv2.putText(
                    overlay,
                    f"ID:{marker_id} depth=None",
                    (u, max(0, v - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                continue

            # 7) 3D point computation
            try:
                x_m = (u - cx) * depth_m / fx
                y_m = (v - cy) * depth_m / fy
                z_m = depth_m
            except Exception as e:
                self.get_logger().error(
                    f"❌ 3D projection failed for ID={marker_id}: {repr(e)}"
                )
                continue

            # 8) Publish point
            try:
                out = PointStamped()
                out.header = self.latest_depth_msg.header
                out.point.x = float(x_m)
                out.point.y = float(y_m)
                out.point.z = float(z_m)
                self.pub_point.publish(out)

                self.detect_success_count += 1
                self.get_logger().info(
                    f"📍 Published marker_3d | "
                    f"ID={marker_id}, pixel=({u},{v}), "
                    f"xyz=({x_m:.4f}, {y_m:.4f}, {z_m:.4f}) m"
                )
            except Exception as e:
                self.get_logger().error(
                    f"❌ Point publish failed for ID={marker_id}: {repr(e)}"
                )
                continue

            # 9) Debug overlay
            try:
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

                pixel_dx = u - int(round(cx))
                pixel_dy = v - int(round(cy))
                cv2.putText(
                    overlay,
                    f"dPix=({pixel_dx},{pixel_dy})",
                    (u, min(overlay.shape[0] - 10, v + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )
            except Exception as e:
                self.get_logger().warn(
                    f"⚠️ Debug overlay draw failed for ID={marker_id}: {repr(e)}"
                )

        if not found_target:
            self.get_logger().info(
                f"ℹ️ Markers detected {detected_ids}, but target ID {self.target_marker_id} not found"
            )

        self._publish_debug(overlay, msg.header)

    def _draw_camera_center(self, img):
        if not self.draw_camera_center or self.camera_info is None:
            return

        try:
            cx = int(round(float(self.camera_info.k[2])))
            cy = int(round(float(self.camera_info.k[5])))

            h, w = img.shape[:2]
            if not (0 <= cx < w and 0 <= cy < h):
                self.get_logger().warn(
                    f"⚠️ Camera center out of image bounds: (cx,cy)=({cx},{cy}), size=({w},{h})"
                )
                return

            s = self.camera_center_cross_size

            # 중심점 표시: 빨간 점 + 십자선
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(img, (cx - s, cy), (cx + s, cy), (0, 0, 255), 2)
            cv2.line(img, (cx, cy - s), (cx, cy + s), (0, 0, 255), 2)

            cv2.putText(
                img,
                f"CAM (0,0) = ({cx},{cy})",
                (max(5, cx + 10), max(20, cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to draw camera center: {repr(e)}")

    def _draw_marker_vs_camera_center(self, img, u, v, cx, cy):
        try:
            cx_i = int(round(cx))
            cy_i = int(round(cy))

            # optical center -> marker center 선
            cv2.line(img, (cx_i, cy_i), (u, v), (255, 0, 255), 2)

            # marker 중심 강조
            cv2.circle(img, (u, v), 8, (0, 255, 255), 2)
        except Exception as e:
            self.get_logger().warn(
                f"⚠️ Failed to draw marker-center relation: {repr(e)}"
            )

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
                self.get_logger().warn(f"⚠️ Empty depth patch at (u,v)=({u},{v})")
                return None

            if "16UC1" in encoding:
                vals = patch[patch > 0]
                if vals.size == 0:
                    return None
                return float(np.median(vals)) / 1000.0

            elif "32FC1" in encoding:
                vals = patch[np.isfinite(patch) & (patch > 0)]
                if vals.size == 0:
                    return None
                return float(np.median(vals))

            else:
                self.get_logger().error(f"❌ Unsupported depth encoding: {encoding}")
                return None

        except Exception as e:
            self.get_logger().error(f"❌ Depth median extraction failed: {repr(e)}")
            return None

    def _publish_debug(self, img, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception as e:
            self.get_logger().error(f"❌ Debug image publish failed: {repr(e)}")

    def _log_time_diff_if_needed(self, color_msg, depth_msg):
        try:
            color_t = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9
            depth_t = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            diff_ms = abs(color_t - depth_t) * 1000.0

            if diff_ms > self.warn_if_time_diff_ms:
                self.get_logger().warn(
                    f"⚠️ Color/Depth timestamp diff is large: {diff_ms:.2f} ms"
                )
        except Exception as e:
            self.get_logger().warn(f"⚠️ Timestamp diff check failed: {repr(e)}")


def main():
    rclpy.init()
    node = Aruco3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 KeyboardInterrupt received, shutting down")
    except Exception as e:
        node.get_logger().error(f"❌ Fatal error in spin: {repr(e)}")
    finally:
        node.get_logger().info(
            "📊 Summary | "
            f"color_frames={node.color_frame_count}, "
            f"depth_frames={node.depth_frame_count}, "
            f"detect_success={node.detect_success_count}, "
            f"detect_fail={node.detect_fail_count}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()