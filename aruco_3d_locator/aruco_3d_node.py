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
}


class Aruco3DNode(Node):
    def __init__(self):
        super().__init__("aruco_3d_node")

        # 토픽 설정
        self.declare_parameter("color_topic", "/camera_r/camera_r/color/image_rect_raw/compressed")
        self.declare_parameter("depth_topic", "/camera_r/camera_r/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_r/camera_r/color/camera_info")

        self.declare_parameter("target_marker_id", 0)
        self.declare_parameter("depth_patch_size", 15)
        self.declare_parameter("enable_verbose_log", True)

        # solvePnP fallback용
        self.declare_parameter("marker_size_m", 0.0625)  # 실제 마커 한 변 길이(m)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)
        self.enable_verbose_log = bool(self.get_parameter("enable_verbose_log").value)
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self.use_new_detector_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_new_detector_api = False

        self.bridge = CvBridge()
        self.latest_depth_msg = None
        self.camera_info = None
        self._depth_info_logged = False

        # Subscription
        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_cb, qos_profile_sensor_data
        )

        self.pub_point = self.create_publisher(PointStamped, "/aruco/marker_3d", 10)
        self.pub_debug = self.create_publisher(Image, "/aruco/debug_image", 10)

        self.get_logger().info(
            f"🚀 ArUco 3D Node Started | target_id={self.target_marker_id}, "
            f"marker_size={self.marker_size_m:.4f}m, patch={self.depth_patch_size}, "
            f"color_topic={self.color_topic}, depth_topic={self.depth_topic}, "
            f"info_topic={self.camera_info_topic}"
        )

    def info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("✅ [INFO] Camera intrinsics received.")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

        if not self._depth_info_logged:
            self.get_logger().info(
                f"[DEPTH_CB] encoding={msg.encoding}, width={msg.width}, height={msg.height}"
            )
            self._depth_info_logged = True

    def color_cb(self, msg: CompressedImage):
        if self.camera_info is None:
            self.get_logger().warn("⏳ Waiting for Camera Info...")
            return
        if self.latest_depth_msg is None:
            self.get_logger().warn("⏳ Waiting for Depth Image...")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Image Decoding Error: {e}")
            return

        if self.use_new_detector_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )

        overlay = bgr.copy()

        if ids is not None:
            self.get_logger().info(f"🔍 Found Marker IDs: {ids.flatten().tolist()}")
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

            try:
                # 네가 원한 방식 유지: passthrough
                cv_depth = self.bridge.imgmsg_to_cv2(
                    self.latest_depth_msg, desired_encoding="passthrough"
                )

                fx, fy = self.camera_info.k[0], self.camera_info.k[4]
                cx, cy = self.camera_info.k[2], self.camera_info.k[5]

                color_h, color_w = bgr.shape[:2]
                depth_h, depth_w = cv_depth.shape[:2]

                if self.enable_verbose_log:
                    self.get_logger().info(
                        f"[SHAPE] color=({color_w}x{color_h}) "
                        f"depth=({depth_w}x{depth_h}) "
                        f"encoding={self.latest_depth_msg.encoding} "
                        f"dtype={cv_depth.dtype} "
                        f"shape={cv_depth.shape}"
                    )
                    self.get_logger().info(
                        f"[STAMP] color={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                        f"depth={self.latest_depth_msg.header.stamp.sec}.{self.latest_depth_msg.header.stamp.nanosec:09d}"
                    )

                use_raw_depth = self.latest_depth_msg.encoding in ("16UC1", "32FC1")

                if not use_raw_depth:
                    self.get_logger().warn(
                        f"[DEPTH_WARNING] depth topic encoding is '{self.latest_depth_msg.encoding}'. "
                        "Raw depth(16UC1/32FC1)가 아니므로 solvePnP tvec fallback을 사용합니다."
                    )

                for idx, marker_id in enumerate(ids.flatten()):
                    if int(marker_id) != self.target_marker_id:
                        continue

                    c = corners[idx][0]  # (4,2)
                    u = int(np.mean(c[:, 0]))
                    v = int(np.mean(c[:, 1]))

                    self.get_logger().info(f"🎯 Target ID {marker_id} at Pixel: ({u}, {v})")
                    cv2.circle(overlay, (u, v), 4, (0, 255, 255), -1)

                    depth_m = None
                    sample_points = []

                    if use_raw_depth:
                        depth_m, sample_points = self._sample_marker_depth_m(
                            cv_depth=cv_depth,
                            encoding=self.latest_depth_msg.encoding,
                            corners=c,
                            color_shape=(color_h, color_w),
                            depth_shape=(depth_h, depth_w),
                        )

                        for i, (cu, cvv, du, dv) in enumerate(sample_points):
                            cv2.circle(overlay, (cu, cvv), 3, (255, 0, 255), -1)
                            cv2.putText(
                                overlay,
                                f"s{i}",
                                (cu + 3, cvv - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 0, 255),
                                1,
                            )

                    if depth_m is not None:
                        # raw depth 성공
                        x_m = (u - cx) * depth_m / fx
                        y_m = (v - cy) * depth_m / fy
                        z_m = depth_m
                        source = "raw_depth"
                    else:
                        # fallback: solvePnP tvec 사용
                        tvec = self._estimate_marker_tvec(c)
                        if tvec is None:
                            self.get_logger().warn(
                                f"⚠️ [POSE_FAIL] solvePnP fallback also failed for ID:{marker_id}"
                            )
                            cv2.putText(
                                overlay,
                                f"ID:{marker_id} pose_fail",
                                (u, max(v - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2,
                            )
                            continue

                        x_m = float(tvec[0])
                        y_m = float(tvec[1])
                        z_m = float(tvec[2])
                        source = "solvePnP"

                    out = PointStamped()
                    out.header = self.latest_depth_msg.header
                    out.point.x = float(x_m)
                    out.point.y = float(y_m)
                    out.point.z = float(z_m)
                    self.pub_point.publish(out)

                    self.get_logger().info(
                        f"🟢 [SUCCESS:{source}] ID:{marker_id} -> "
                        f"X:{x_m:.3f}, Y:{y_m:.3f}, Z:{z_m:.3f}m"
                    )

                    text_color = (0, 255, 0) if source == "raw_depth" else (0, 200, 255)
                    cv2.putText(
                        overlay,
                        f"ID:{marker_id} {source} Z:{z_m:.3f}m",
                        (u, max(v - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        2,
                    )

            except Exception as e:
                self.get_logger().error(f"❌ Processing Error: {e}")

        self._publish_debug(overlay, msg.header)

    def _estimate_marker_tvec(self, corners):
        """
        depth를 못 쓸 때 solvePnP로 마커 중심까지의 tvec 추정.
        반환값: np.array([x, y, z]) in meters
        """
        if self.camera_info is None:
            return None

        camera_matrix = np.array(self.camera_info.k, dtype=np.float32).reshape(3, 3)
        dist_coeffs = np.array(self.camera_info.d, dtype=np.float32)

        marker_size = self.marker_size_m
        object_points = np.array([
            [-marker_size / 2.0,  marker_size / 2.0, 0.0],
            [ marker_size / 2.0,  marker_size / 2.0, 0.0],
            [ marker_size / 2.0, -marker_size / 2.0, 0.0],
            [-marker_size / 2.0, -marker_size / 2.0, 0.0],
        ], dtype=np.float32)

        image_points = corners.astype(np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                return None
            return tvec.reshape(3)
        except Exception as e:
            self.get_logger().warn(f"[solvePnP] failed: {e}")
            return None

    def _scale_to_depth(self, u, v, color_shape, depth_shape):
        color_h, color_w = color_shape
        depth_h, depth_w = depth_shape

        if color_w == depth_w and color_h == depth_h:
            return int(u), int(v)

        u_d = int(round(u * depth_w / float(color_w)))
        v_d = int(round(v * depth_h / float(color_h)))
        return u_d, v_d

    def _sample_marker_depth_m(self, cv_depth, encoding, corners, color_shape, depth_shape):
        pts = corners.astype(np.float32)

        center = np.mean(pts, axis=0)
        p0, p1, p2, p3 = pts

        sample_points_color = [
            center,
            0.7 * center + 0.3 * p0,
            0.7 * center + 0.3 * p1,
            0.7 * center + 0.3 * p2,
            0.7 * center + 0.3 * p3,
        ]

        depths = []
        debug_points = []

        for p in sample_points_color:
            cu, cvv = int(round(p[0])), int(round(p[1]))
            du, dv = self._scale_to_depth(cu, cvv, color_shape, depth_shape)
            debug_points.append((cu, cvv, du, dv))

            d = self._median_depth_m(cv_depth, encoding, du, dv)
            if d is not None:
                depths.append(d)

        if self.enable_verbose_log:
            self.get_logger().info(
                f"[SAMPLES] valid_depth_count={len(depths)}/{len(sample_points_color)}"
            )

        if len(depths) == 0:
            return None, debug_points

        return float(np.median(depths)), debug_points

    def _median_depth_m(self, cv_depth, encoding, u, v):
        r = self.depth_patch_size // 2
        h, w = cv_depth.shape[:2]

        if not (0 <= u < w and 0 <= v < h):
            self.get_logger().warn(
                f"[DEPTH_FAIL] pixel ({u}, {v}) out of range for depth image ({w}, {h})"
            )
            return None

        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = cv_depth[y0:y1, x0:x1]

        if self.enable_verbose_log:
            self.get_logger().info(
                f"[PATCH] center=({u},{v}) range=({x0}:{x1}, {y0}:{y1}) shape={patch.shape}"
            )

        try:
            if "16UC1" in encoding:
                vals = patch[patch > 0]
                if self.enable_verbose_log:
                    raw_min = int(np.min(patch)) if patch.size > 0 else -1
                    raw_max = int(np.max(patch)) if patch.size > 0 else -1
                    self.get_logger().info(
                        f"[PATCH-16U] valid={vals.size}, raw_min={raw_min}, raw_max={raw_max}"
                    )
                if vals.size == 0:
                    return None
                return float(np.median(vals)) / 1000.0

            elif "32FC1" in encoding:
                valid_mask = np.isfinite(patch) & (patch > 0)
                vals = patch[valid_mask]
                if self.enable_verbose_log:
                    raw_min = float(np.nanmin(patch)) if patch.size > 0 else float("nan")
                    raw_max = float(np.nanmax(patch)) if patch.size > 0 else float("nan")
                    self.get_logger().info(
                        f"[PATCH-32F] valid={vals.size}, raw_min={raw_min:.6f}, raw_max={raw_max:.6f}"
                    )
                if vals.size == 0:
                    return None
                return float(np.median(vals))

            else:
                return None

        except Exception as e:
            self.get_logger().error(f"[PATCH_ERROR] {e}")
            return None

    def _publish_debug(self, img, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception:
            pass


def main():
    rclpy.init()
    node = Aruco3DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()