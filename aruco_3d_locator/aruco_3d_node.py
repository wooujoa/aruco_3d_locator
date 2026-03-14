#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import tf2_ros
from scipy.spatial.transform import Rotation as R


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

        # camera_r 입력 유지
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

        # base 변환용
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("gripper_frame", "arm_l_link7")   # camera_r 이지만 left TF 사용
        self.declare_parameter("use_msg_timestamp", False)
        self.declare_parameter("tf_timeout_sec", 0.2)

        # Hand-Eye: Camera -> Gripper
        self.T_cam_to_gripper = np.array([
            [0.70655016, -0.70427039, -0.06921048,  0.30271700],
            [-0.13697267, -0.04014969, -0.98976082, 0.01067388],
            [0.69428046,  0.70879561, -0.12483357, -0.08213032],
            [0.0,         0.0,         0.0,         1.0]
        ], dtype=np.float64)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

        self.base_frame = self.get_parameter("base_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.use_msg_timestamp = bool(self.get_parameter("use_msg_timestamp").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        dict_name = self.get_parameter("aruco_dict").value
        dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        self.pub_point = self.create_publisher(PointStamped, "/aruco/marker_3d", 10)
        self.pub_base_pose = self.create_publisher(PointStamped, "/aruco/marker_base_pose", 10)

        self.get_logger().info("Aruco3DNode started")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def depth_cb(self, msg: Image):
        self.latest_depth_msg = msg

    def color_cb(self, msg: CompressedImage):
        if self.latest_depth_msg is None or self.camera_info is None:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return

        try:
            if self.use_new_detector_api:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    self.aruco_dict,
                    parameters=self.detector_params
                )
        except Exception:
            return

        if ids is None or len(ids) == 0:
            return

        try:
            cv_depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth_msg,
                desired_encoding="passthrough"
            )
        except Exception:
            return

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        if fx == 0.0 or fy == 0.0:
            return

        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_marker_id:
                continue

            c = corners[idx][0]
            u = int(np.mean(c[:, 0]))
            v = int(np.mean(c[:, 1]))

            h, w = cv_depth.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_m = self._median_depth_m(
                cv_depth,
                self.latest_depth_msg.encoding,
                u,
                v
            )
            if depth_m is None:
                continue

            x_m = (u - cx) * depth_m / fx
            y_m = (v - cy) * depth_m / fy
            z_m = depth_m

            cam_msg = PointStamped()
            cam_msg.header = self.latest_depth_msg.header
            cam_msg.point.x = float(x_m)
            cam_msg.point.y = float(y_m)
            cam_msg.point.z = float(z_m)
            self.pub_point.publish(cam_msg)

            self.publish_base_pose(cam_msg)

    def publish_base_pose(self, cam_msg: PointStamped):
        p_cam = np.array(
            [cam_msg.point.x, cam_msg.point.y, cam_msg.point.z, 1.0],
            dtype=np.float64
        )

        p_gripper = self.T_cam_to_gripper @ p_cam

        transform = self.lookup_base_from_gripper(cam_msg)
        if transform is None:
            return

        T_gripper_to_base = self.make_transform_matrix(transform)
        p_base = T_gripper_to_base @ p_gripper

        out_msg = PointStamped()
        out_msg.header.stamp = cam_msg.header.stamp
        out_msg.header.frame_id = self.base_frame
        out_msg.point.x = float(p_base[0])
        out_msg.point.y = float(p_base[1])
        out_msg.point.z = float(p_base[2])

        self.pub_base_pose.publish(out_msg)

    def lookup_base_from_gripper(self, msg: PointStamped):
        if self.use_msg_timestamp:
            try:
                target_time = rclpy.time.Time.from_msg(msg.header.stamp)
                return self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.gripper_frame,
                    target_time,
                    timeout=Duration(seconds=self.tf_timeout_sec)
                )
            except Exception:
                pass

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec)
            )
        except Exception:
            return None

    def make_transform_matrix(self, transform_stamped):
        q = transform_stamped.transform.rotation
        t = transform_stamped.transform.translation

        rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _median_depth_m(self, cv_depth, encoding, u, v):
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
            return float(np.median(vals)) / 1000.0 if vals.size > 0 else None

        if "32FC1" in encoding:
            vals = patch[np.isfinite(patch) & (patch > 0)]
            return float(np.median(vals)) if vals.size > 0 else None

        return None


def main():
    rclpy.init()
    node = Aruco3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()