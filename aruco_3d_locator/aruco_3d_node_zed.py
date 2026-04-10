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
        self.declare_parameter("color_topic", "/zedm/zed_node/left/image_rect_color/compressed")
        self.declare_parameter("camera_info_topic", "/zedm/zed_node/left/camera_info")
        
        self.declare_parameter("point_output_topic", "/aruco/marker_3d_zed")
        self.declare_parameter("debug_output_topic", "/aruco/debug_image_zed")

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 1)
        
        # [핵심] 실제 인쇄된 마커의 검은 테두리 바깥쪽 한 변의 길이 (단위: 미터)
        self.declare_parameter("marker_size", 0.047) 

        self.declare_parameter("debug", True)
        self.declare_parameter("draw_camera_center", True)
        self.declare_parameter("camera_center_cross_size", 15)

        # ==================================================
        # Fetch Parameters
        # ==================================================
        self.color_topic = self.get_parameter("color_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.point_output_topic = self.get_parameter("point_output_topic").value
        self.debug_output_topic = self.get_parameter("debug_output_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.marker_size = float(self.get_parameter("marker_size").value)

        self.debug = bool(self.get_parameter("debug").value)
        self.draw_camera_center = bool(self.get_parameter("draw_camera_center").value)
        self.camera_center_cross_size = int(self.get_parameter("camera_center_cross_size").value)

        # ==================================================
        # OpenCV ArUco Setup
        # ==================================================
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

        # ==================================================
        # Object Points for solvePnP
        # (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # ==================================================
        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ], dtype=np.float32)

        # ==================================================
        # Sub / Pub
        # ==================================================
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.info_cb, qos_profile_sensor_data
        )
        self.sub_color = self.create_subscription(
            CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data
        )

        self.pub_point = self.create_publisher(PointStamped, self.point_output_topic, 10)
        self.pub_debug = self.create_publisher(Image, self.debug_output_topic, 10)

        self.get_logger().info("========================================")
        self.get_logger().info("✅ ArUco Pose Node Started (RGB + solvePnP)")
        self.get_logger().info("   [Depth 토픽 동기화 제거, 기하학 기반 추출]")
        self.get_logger().info(f"  color_topic   : {self.color_topic}")
        self.get_logger().info(f"  marker_size   : {self.marker_size} m")
        self.get_logger().info("========================================")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def color_cb(self, msg: CompressedImage):
        if self.camera_info is None:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None: return
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding failed: {repr(e)}")
            return

        overlay = bgr.copy()
        self._draw_camera_center(overlay)

        # 1. 아루코 마커 2D 검출
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
            self._publish_debug(overlay, msg.header)
            return

        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

        # 2. 카메라 내부 파라미터 (Intrinsics) 준비
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]
        
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Rectified 이미지 기준이므로 왜곡 계수는 0으로 처리
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # 3. 타겟 마커에 대해 solvePnP 수행
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_marker_id:
                continue

            # corners[idx][0] 은 마커의 4개 픽셀 코너점 (N, 2)
            corner_points = corners[idx][0]

            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corner_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                self.get_logger().warn(f"⚠️ solvePnP failed for ID={marker_id}")
                continue

            # tvec 은 카메라 좌표계 기준 마커의 3D 위치 [X, Y, Z]
            x_m = float(tvec[0][0])
            y_m = float(tvec[1][0])
            z_m = float(tvec[2][0])

            # 다음 단계 TF 노드(ZedPointBaseTransformer)와의 호환성을 위해 PointStamped 발행
            out = PointStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.camera_info.header.frame_id
            out.point.x = x_m
            out.point.y = y_m
            out.point.z = z_m
            self.pub_point.publish(out)

            # 3D 축 렌더링 (디버그 뷰 확인용)
            # Z축(파란색)이 마커 표면에서 튀어나오는 방향으로 그려집니다.
            cv2.drawFrameAxes(overlay, K, dist_coeffs, rvec, tvec, self.marker_size * 0.8)

            u = int(np.mean(corner_points[:, 0]))
            v = int(np.mean(corner_points[:, 1]))
            
            cv2.putText(
                overlay,
                f"PnP Z: {z_m:.3f}m",
                (u, max(20, v - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            if self.debug:
                self.get_logger().info(
                    f"📍 solvePnP Success | ID={marker_id} | "
                    f"XYZ: ({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) m"
                )

        self._publish_debug(overlay, msg.header)

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
        except Exception as e:
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