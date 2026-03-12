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
        self.declare_parameter("target_marker_id", 0) # ID 0번 추적
        self.declare_parameter("depth_patch_size", 5)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

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
        self.sub_color = self.create_subscription(CompressedImage, self.color_topic, self.color_cb, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, qos_profile_sensor_data)

        self.pub_point = self.create_publisher(PointStamped, "/aruco/marker_3d", 10)
        self.pub_debug = self.create_publisher(Image, "/aruco/debug_image", 10)

        self.get_logger().info(f"🚀 ArUco Debug Node Started. Target ID: {self.target_marker_id}")

    def info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("✅ [INFO] Camera intrinsics received.")

    def depth_cb(self, msg: Image):
        # Depth 수신 여부 로그 (너무 자주 찍히지 않게 1초에 한 번 정도의 주기로 짐작)
        self.latest_depth_msg = msg

    def color_cb(self, msg: CompressedImage):
        # 상태 체크 로그
        if self.camera_info is None:
            self.get_logger().warn("⏳ Waiting for Camera Info...")
            return
        if self.latest_depth_msg is None:
            self.get_logger().warn("⏳ Waiting for Depth Image...")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Image Decoding Error: {e}")
            return

        # 마커 검출 시작
        if self.use_new_detector_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

        overlay = bgr.copy()

        if ids is not None:
            # 1단계 로그: 마커 발견
            self.get_logger().info(f"🔍 Found Marker IDs: {ids.flatten().tolist()}")
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            
            try:
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
                fx, fy = self.camera_info.k[0], self.camera_info.k[4]
                cx, cy = self.camera_info.k[2], self.camera_info.k[5]

                for idx, marker_id in enumerate(ids.flatten()):
                    if int(marker_id) == self.target_marker_id:
                        # 2단계 로그: 타겟 마커 일치 확인
                        c = corners[idx][0]
                        u, v = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                        self.get_logger().info(f"🎯 Target ID {marker_id} at Pixel: ({u}, {v})")

                        # Depth 추출
                        depth_m = self._median_depth_m(cv_depth, self.latest_depth_msg.encoding, u, v)
                        
                        if depth_m is not None:
                            # 3단계 로그: 최종 좌표 계산 및 발행 성공
                            x_m = (u - cx) * depth_m / fx
                            y_m = (v - cy) * depth_m / fy
                            
                            out = PointStamped()
                            out.header = self.latest_depth_msg.header
                            out.point.x, out.point.y, out.point.z = float(x_m), float(y_m), float(depth_m)
                            self.pub_point.publish(out)

                            self.get_logger().info(f"🟢 [SUCCESS] ID:{marker_id} -> X:{x_m:.3f}, Y:{y_m:.3f}, Z:{depth_m:.3f}m")
                            
                            cv2.putText(overlay, f"ID:{marker_id} Z:{depth_m:.3f}m", (u, v-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # 실패 로그: Depth 값이 없음
                            self.get_logger().warn(f"⚠️ [DEPTH_FAIL] No valid depth at ({u}, {v}). Check distance or lighting.")
            except Exception as e:
                self.get_logger().error(f"❌ Processing Error: {e}")
        else:
            # 마커가 아예 안 보이는 경우
            pass

        self._publish_debug(overlay, msg.header)

    def _median_depth_m(self, cv_depth, encoding, u, v):
        r = self.depth_patch_size // 2
        h, w = cv_depth.shape[:2]
        
        # 패치 범위 로그 (좌표가 튀는지 확인)
        y0, y1 = max(0, v-r), min(h, v+r+1)
        x0, x1 = max(0, u-r), min(w, u+r+1)
        patch = cv_depth[y0:y1, x0:x1]

        if "16UC1" in encoding:
            vals = patch[patch > 0]
            if vals.size == 0: return None
            return float(np.median(vals)) / 1000.0
        elif "32FC1" in encoding:
            vals = patch[np.isfinite(patch) & (patch > 0)]
            if vals.size == 0: return None
            return float(np.median(vals))
        return None

    def _publish_debug(self, img, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except:
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