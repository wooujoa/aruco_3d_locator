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
        self.declare_parameter("target_marker_id", 0)   # ID 0번 추적
        self.declare_parameter("depth_patch_size", 15)  # 기존 5 -> 15
        self.declare_parameter("enable_verbose_log", True)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)
        self.enable_verbose_log = bool(self.get_parameter("enable_verbose_log").value)

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
            f"patch={self.depth_patch_size}, color_topic={self.color_topic}, "
            f"depth_topic={self.depth_topic}, info_topic={self.camera_info_topic}"
        )

    def info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("✅ [INFO] Camera intrinsics received.")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

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
                        f"encoding={self.latest_depth_msg.encoding}"
                    )
                    self.get_logger().info(
                        f"[STAMP] color={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
                        f"depth={self.latest_depth_msg.header.stamp.sec}.{self.latest_depth_msg.header.stamp.nanosec:09d}"
                    )

                for idx, marker_id in enumerate(ids.flatten()):
                    if int(marker_id) != self.target_marker_id:
                        continue

                    c = corners[idx][0]  # shape: (4,2)
                    u = int(np.mean(c[:, 0]))
                    v = int(np.mean(c[:, 1]))

                    self.get_logger().info(f"🎯 Target ID {marker_id} at Pixel: ({u}, {v})")

                    # 중심점 표시
                    cv2.circle(overlay, (u, v), 4, (0, 255, 255), -1)

                    # depth 추출: 마커 내부 5지점 샘플링
                    depth_m, sample_points = self._sample_marker_depth_m(
                        cv_depth=cv_depth,
                        encoding=self.latest_depth_msg.encoding,
                        corners=c,
                        color_shape=(color_h, color_w),
                        depth_shape=(depth_h, depth_w),
                    )

                    # 샘플 포인트 시각화
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
                        x_m = (u - cx) * depth_m / fx
                        y_m = (v - cy) * depth_m / fy

                        out = PointStamped()
                        out.header = self.latest_depth_msg.header
                        out.point.x = float(x_m)
                        out.point.y = float(y_m)
                        out.point.z = float(depth_m)
                        self.pub_point.publish(out)

                        self.get_logger().info(
                            f"🟢 [SUCCESS] ID:{marker_id} -> X:{x_m:.3f}, Y:{y_m:.3f}, Z:{depth_m:.3f}m"
                        )

                        cv2.putText(
                            overlay,
                            f"ID:{marker_id} Z:{depth_m:.3f}m",
                            (u, max(v - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        self.get_logger().warn(
                            f"⚠️ [DEPTH_FAIL] No valid depth around marker center ({u}, {v}). "
                            f"Check patch/raw depth/timestamp alignment."
                        )
                        cv2.putText(
                            overlay,
                            f"ID:{marker_id} depth_fail",
                            (u, max(v - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

            except Exception as e:
                self.get_logger().error(f"❌ Processing Error: {e}")

        self._publish_debug(overlay, msg.header)

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

        # corners: [top-left, top-right, bottom-right, bottom-left] 형태를 기대
        center = np.mean(pts, axis=0)
        p0, p1, p2, p3 = pts

        # 중심 + 각 코너 방향 안쪽 4개
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
                self.get_logger().warn(f"[DEPTH_FAIL] Unsupported depth encoding: {encoding}")
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