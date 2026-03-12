#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage # CompressedImage 추가

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    # ... (기타 딕셔너리 생략) ...
}

class Aruco3DNode(Node):
    def __init__(self):
        super().__init__("aruco_3d_node")

        # 1. 로봇 환경에 맞는 토픽 설정 (camera_r 및 compressed 반영)
        self.declare_parameter("color_topic", "/camera_r/camera_r/color/image_rect_raw/compressed")
        self.declare_parameter("depth_topic", "/camera_r/camera_r/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_r/camera_r/depth/camera_info")
        
        # 2. 마커 설정 (올려주신 이미지 반영: 4X4_50, ID 0)
        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 0) # ID 0번만 추적
        self.declare_parameter("depth_patch_size", 5)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

        dict_id = DICT_MAP.get(self.get_parameter("aruco_dict").value, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        # OpenCV 버전 호환성 처리
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

        # 3. Subscription 설정 (CompressedImage 대응)
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

        self.get_logger().info(f"✅ ArUco Node Started: ID {self.target_marker_id}")

    def info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("✅ Camera info received")

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def color_cb(self, msg: CompressedImage):
        if self.latest_depth_msg is None or self.camera_info is None:
            return

        try:
            # CompressedImage 디코딩 (YOLO 노드와 동일 방식)
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"❌ Decoding failed: {e}")
            return

        # 마커 검출
        if self.use_new_detector_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

        overlay = bgr.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            
            try:
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, desired_encoding="passthrough")
                fx, fy = self.camera_info.k[0], self.camera_info.k[4]
                cx, cy = self.camera_info.k[2], self.camera_info.k[5]

                for idx, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_marker_id:
                        # 마커 중심점 계산
                        c = corners[idx][0]
                        u, v = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))

                        # Depth 추출
                        depth_m = self._median_depth_m(cv_depth, self.latest_depth_msg.encoding, u, v)
                        if depth_m:
                            # 3D 좌표 계산
                            x_m = (u - cx) * depth_m / fx
                            y_m = (v - cy) * depth_m / fy
                            
                            # 발행
                            out = PointStamped()
                            out.header = self.latest_depth_msg.header
                            out.point.x, out.point.y, out.point.z = float(x_m), float(y_m), float(depth_m)
                            self.pub_point.publish(out)

                            # 디버그 텍스트
                            cv2.putText(overlay, f"ID:{marker_id} Z:{depth_m:.3f}m", (u, v-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                self.get_logger().error(f"❌ Depth processing error: {e}")

        self._publish_debug(overlay, msg.header)

    def _median_depth_m(self, cv_depth, encoding, u, v):
        r = self.depth_patch_size // 2
        h, w = cv_depth.shape[:2]
        patch = cv_depth[max(0, v-r):min(h, v+r+1), max(0, u-r):min(w, u+r+1)]

        if "16UC1" in encoding:
            vals = patch[patch > 0]
            return float(np.median(vals)) / 1000.0 if vals.size > 0 else None
        elif "32FC1" in encoding:
            vals = patch[np.isfinite(patch) & (patch > 0)]
            return float(np.median(vals)) if vals.size > 0 else None
        return None

    def _publish_debug(self, img, header):
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header = header
        self.pub_debug.publish(msg)

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