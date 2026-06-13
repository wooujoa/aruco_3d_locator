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
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoPoseNodeZed(Node):
    def __init__(self):
        super().__init__("aruco_pose_node_zed")

        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter(
            "color_topic",
            "/zedm/zed_node/left/image_rect_color/compressed"
        )
        self.declare_parameter(
            "camera_info_topic",
            "/zedm/zed_node/left/camera_info"
        )

        # ZED depth image topic
        # wrapper 설정에서 publish_depth_map: true 여야 나옴
        self.declare_parameter(
            "depth_topic",
            "/zedm/zed_node/depth/depth_registered"
        )

        self.declare_parameter("point_output_topic", "/aruco/marker_3d_zed")
        self.declare_parameter("debug_output_topic", "/aruco/debug_image_zed")

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 0)

        # 실제 인쇄된 마커의 검은 테두리 바깥쪽 한 변 길이 [m]
        self.declare_parameter("marker_size", 0.047)

        # ==================================================
        # Depth fusion options
        # ==================================================
        # False면 기존처럼 solvePnP만 사용
        # True면 ArUco 중심 주변 Neural depth median으로 XYZ를 재계산
        self.declare_parameter("use_depth_fusion", True)

        # depth가 없거나 invalid일 때 기존 solvePnP 값을 publish할지 여부
        self.declare_parameter("fallback_to_pnp", True)

        # ArUco 중심 주변 depth ROI 반경 [pixel]
        # roi_radius=5이면 11x11 영역 사용
        self.declare_parameter("depth_roi_radius", 5)

        # depth 유효 범위 [m]
        self.declare_parameter("valid_depth_min", 0.1)
        self.declare_parameter("valid_depth_max", 10.0)

        # PnP Z와 depth Z 차이가 너무 크면 reject할지 여부
        self.declare_parameter("enable_z_diff_check", True)
        self.declare_parameter("max_z_diff", 0.30)

        # 최종값 선택 방식
        # "depth": depth 기반 XYZ 사용
        # "pnp": solvePnP XYZ 사용
        # "blend": PnP와 depth를 섞음
        self.declare_parameter("output_mode", "depth")
        self.declare_parameter("blend_alpha", 0.5)

        self.declare_parameter("debug", True)
        self.declare_parameter("draw_camera_center", True)
        self.declare_parameter("camera_center_cross_size", 15)

        # ==================================================
        # Fetch Parameters
        # ==================================================
        self.color_topic = self.get_parameter("color_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value

        self.point_output_topic = self.get_parameter("point_output_topic").value
        self.debug_output_topic = self.get_parameter("debug_output_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.marker_size = float(self.get_parameter("marker_size").value)

        self.use_depth_fusion = bool(self.get_parameter("use_depth_fusion").value)
        self.fallback_to_pnp = bool(self.get_parameter("fallback_to_pnp").value)

        self.depth_roi_radius = int(self.get_parameter("depth_roi_radius").value)
        self.valid_depth_min = float(self.get_parameter("valid_depth_min").value)
        self.valid_depth_max = float(self.get_parameter("valid_depth_max").value)

        self.enable_z_diff_check = bool(self.get_parameter("enable_z_diff_check").value)
        self.max_z_diff = float(self.get_parameter("max_z_diff").value)

        self.output_mode = str(self.get_parameter("output_mode").value).lower()
        self.blend_alpha = float(self.get_parameter("blend_alpha").value)

        self.debug = bool(self.get_parameter("debug").value)
        self.draw_camera_center = bool(self.get_parameter("draw_camera_center").value)
        self.camera_center_cross_size = int(
            self.get_parameter("camera_center_cross_size").value
        )

        if self.output_mode not in ["depth", "pnp", "blend"]:
            self.get_logger().warn(
                f"Unknown output_mode='{self.output_mode}', forced to 'depth'"
            )
            self.output_mode = "depth"

        self.blend_alpha = max(0.0, min(1.0, self.blend_alpha))

        # ==================================================
        # OpenCV ArUco Setup
        # ==================================================
        dict_name = self.get_parameter("aruco_dict").value
        dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.detector_params
            )
            self.use_new_detector_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.use_new_detector_api = False

        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_depth = None
        self.latest_depth_stamp = None

        # ==================================================
        # Object Points for solvePnP
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # ==================================================
        s = self.marker_size / 2.0
        self.obj_points = np.array(
            [
                [-s, -s, 0],
                [s, -s, 0],
                [s, s, 0],
                [-s, s, 0],
            ],
            dtype=np.float32,
        )

        # ==================================================
        # Sub / Pub
        # ==================================================
        self.sub_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.info_cb,
            qos_profile_sensor_data,
        )

        self.sub_color = self.create_subscription(
            CompressedImage,
            self.color_topic,
            self.color_cb,
            qos_profile_sensor_data,
        )

        # depth fusion을 켜든 끄든 일단 구독은 생성
        # output_mode=pnp로 두면 기존 PnP 실험도 가능
        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_cb,
            qos_profile_sensor_data,
        )

        self.pub_point = self.create_publisher(
            PointStamped,
            self.point_output_topic,
            10,
        )

        self.pub_debug = self.create_publisher(
            Image,
            self.debug_output_topic,
            10,
        )

        self.get_logger().info("========================================")
        self.get_logger().info("✅ ArUco Pose Node Started")
        self.get_logger().info("   [RGB ArUco + solvePnP + optional ZED depth ROI]")
        self.get_logger().info(f"  color_topic       : {self.color_topic}")
        self.get_logger().info(f"  camera_info_topic : {self.camera_info_topic}")
        self.get_logger().info(f"  depth_topic       : {self.depth_topic}")
        self.get_logger().info(f"  marker_size       : {self.marker_size} m")
        self.get_logger().info(f"  use_depth_fusion  : {self.use_depth_fusion}")
        self.get_logger().info(f"  output_mode       : {self.output_mode}")
        self.get_logger().info(f"  depth_roi_radius  : {self.depth_roi_radius} px")
        self.get_logger().info("========================================")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def depth_cb(self, msg: Image):
        try:
            # ZED depth image는 보통 32FC1, 단위 m
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.latest_depth = np.array(depth, dtype=np.float32)
            self.latest_depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"⚠️ Depth conversion failed: {repr(e)}")
            self.latest_depth = None
            self.latest_depth_stamp = None

    def color_cb(self, msg: CompressedImage):
        if self.camera_info is None:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding failed: {repr(e)}")
            return

        overlay = bgr.copy()
        self._draw_camera_center(overlay)

        # ==================================================
        # 1. ArUco 2D detection
        # ==================================================
        try:
            if self.use_new_detector_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.detector_params,
                )
        except Exception as e:
            self.get_logger().error(f"❌ ArUco detection failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        if ids is None or len(ids) == 0:
            self._publish_debug(overlay, msg.header)
            return

        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

        # ==================================================
        # 2. Camera intrinsics
        # ==================================================
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # rectified image 기준이므로 왜곡 계수 0
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # ==================================================
        # 3. Target marker processing
        # ==================================================
        for idx, marker_id in enumerate(ids.flatten()):
            if int(marker_id) != self.target_marker_id:
                continue

            corner_points = corners[idx][0].astype(np.float32)

            # ------------------------------
            # 3-1. solvePnP
            # ------------------------------
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corner_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                self.get_logger().warn(f"⚠️ solvePnP failed for ID={marker_id}")
                continue

            x_pnp = float(tvec[0][0])
            y_pnp = float(tvec[1][0])
            z_pnp = float(tvec[2][0])

            # ArUco center pixel
            u = int(round(float(np.mean(corner_points[:, 0]))))
            v = int(round(float(np.mean(corner_points[:, 1]))))

            # ------------------------------
            # 3-2. depth ROI median
            # ------------------------------
            depth_ok = False
            z_depth = None
            x_depth = None
            y_depth = None
            valid_count = 0

            if self.use_depth_fusion:
                depth_ok, z_depth, valid_count = self._get_depth_median(u, v)

                if depth_ok:
                    x_depth = (float(u) - cx) * z_depth / fx
                    y_depth = (float(v) - cy) * z_depth / fy

            # ------------------------------
            # 3-3. final output decision
            # ------------------------------
            final_source = "pnp"
            x_final = x_pnp
            y_final = y_pnp
            z_final = z_pnp

            if self.use_depth_fusion and depth_ok:
                z_diff = abs(z_pnp - z_depth)

                if self.enable_z_diff_check and z_diff > self.max_z_diff:
                    # 차이가 너무 크면 상황에 따라 depth 또는 pnp 선택
                    # 실험 목적에서는 depth를 써보는 게 핵심이라 depth 사용
                    self.get_logger().warn(
                        f"⚠️ Z diff large | "
                        f"PnP Z={z_pnp:.3f}, Depth Z={z_depth:.3f}, "
                        f"diff={z_diff:.3f} > {self.max_z_diff:.3f}"
                    )

                if self.output_mode == "depth":
                    x_final = x_depth
                    y_final = y_depth
                    z_final = z_depth
                    final_source = "depth"

                elif self.output_mode == "blend":
                    a = self.blend_alpha
                    x_final = a * x_depth + (1.0 - a) * x_pnp
                    y_final = a * y_depth + (1.0 - a) * y_pnp
                    z_final = a * z_depth + (1.0 - a) * z_pnp
                    final_source = f"blend(a={a:.2f})"

                else:
                    x_final = x_pnp
                    y_final = y_pnp
                    z_final = z_pnp
                    final_source = "pnp"

            elif self.use_depth_fusion and not depth_ok:
                if not self.fallback_to_pnp:
                    self.get_logger().warn(
                        f"⚠️ No valid depth at marker ROI. Skip publish. "
                        f"ID={marker_id}, center=({u},{v})"
                    )
                    continue

                x_final = x_pnp
                y_final = y_pnp
                z_final = z_pnp
                final_source = "pnp_fallback"

            # ------------------------------
            # 3-4. publish PointStamped
            # ------------------------------
            out = PointStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.camera_info.header.frame_id
            out.point.x = float(x_final)
            out.point.y = float(y_final)
            out.point.z = float(z_final)
            self.pub_point.publish(out)

            # ------------------------------
            # 3-5. debug drawing
            # ------------------------------
            cv2.drawFrameAxes(
                overlay,
                K,
                dist_coeffs,
                rvec,
                tvec,
                self.marker_size * 0.8,
            )

            # depth ROI rectangle
            self._draw_depth_roi(overlay, u, v)

            text_y = max(20, v - 45)
            cv2.putText(
                overlay,
                f"src: {final_source}",
                (u, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                f"PnP Z: {z_pnp:.3f}m",
                (u, text_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            if depth_ok:
                cv2.putText(
                    overlay,
                    f"Depth Z: {z_depth:.3f}m n={valid_count}",
                    (u, text_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    overlay,
                    "Depth Z: invalid",
                    (u, text_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

            cv2.circle(overlay, (u, v), 4, (0, 255, 255), -1)

            if self.debug:
                if depth_ok:
                    self.get_logger().info(
                        f"📍 ID={marker_id} | src={final_source} | "
                        f"PnP=({x_pnp:.3f}, {y_pnp:.3f}, {z_pnp:.3f}) m | "
                        f"Depth=({x_depth:.3f}, {y_depth:.3f}, {z_depth:.3f}) m | "
                        f"Final=({x_final:.3f}, {y_final:.3f}, {z_final:.3f}) m | "
                        f"center=({u},{v}) valid_depth={valid_count}"
                    )
                else:
                    self.get_logger().info(
                        f"📍 ID={marker_id} | src={final_source} | "
                        f"PnP=({x_pnp:.3f}, {y_pnp:.3f}, {z_pnp:.3f}) m | "
                        f"Depth=invalid | "
                        f"Final=({x_final:.3f}, {y_final:.3f}, {z_final:.3f}) m | "
                        f"center=({u},{v})"
                    )

        self._publish_debug(overlay, msg.header)

    def _get_depth_median(self, u: int, v: int):
        """
        ArUco center 주변 ROI에서 depth median을 구함.
        ZED depth image는 보통 32FC1, 단위 m.
        """
        if self.latest_depth is None:
            return False, None, 0

        depth = self.latest_depth

        if depth.ndim != 2:
            return False, None, 0

        h, w = depth.shape[:2]

        if u < 0 or u >= w or v < 0 or v >= h:
            return False, None, 0

        r = max(0, int(self.depth_roi_radius))

        u0 = max(0, u - r)
        u1 = min(w, u + r + 1)
        v0 = max(0, v - r)
        v1 = min(h, v + r + 1)

        roi = depth[v0:v1, u0:u1].astype(np.float32)

        valid = roi[np.isfinite(roi)]
        valid = valid[valid > self.valid_depth_min]
        valid = valid[valid < self.valid_depth_max]

        if valid.size == 0:
            return False, None, 0

        z = float(np.median(valid))
        return True, z, int(valid.size)

    def _draw_depth_roi(self, img, u: int, v: int):
        r = max(0, int(self.depth_roi_radius))
        h, w = img.shape[:2]

        u0 = max(0, u - r)
        u1 = min(w - 1, u + r)
        v0 = max(0, v - r)
        v1 = min(h - 1, v + r)

        cv2.rectangle(img, (u0, v0), (u1, v1), (255, 255, 0), 2)

    def _draw_camera_center(self, img):
        if not self.draw_camera_center or self.camera_info is None:
            return

        cx = int(round(float(self.camera_info.k[2])))
        cy = int(round(float(self.camera_info.k[5])))
        s = self.camera_center_cross_size

        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(img, (cx - s, cy), (cx + s, cy), (0, 0, 255), 2)
        cv2.line(img, (cx, cy - s), (cx, cy + s), (0, 0, 255), 2)

    def _publish_debug(self, img, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except Exception:
            pass


def main():
    rclpy.init()
    node = ArucoPoseNodeZed()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()