#!/usr/bin/env python3

import os
import cv2
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from std_msgs.msg import Bool, String, Int32
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


class ArucoZedNode(Node):
    """
    master_2 연동용 ARUCO_ZED 노드.

    Subscribe:
      /aruco_zed_start       std_msgs/Bool
      /target_item_name      std_msgs/String
      /target_aruco_id       std_msgs/Int32
      ZED image
      ZED camera_info

    Publish:
      /aruco/marker_3d_zed   geometry_msgs/PointStamped

    동작:
      - /aruco_zed_start true일 때만 검출 수행
      - /target_aruco_id가 들어오면 그 ID를 우선 사용
      - /target_item_name은 로그용으로만 사용하고, target ID는 /target_aruco_id만 신뢰
      - target ID 마커를 찾으면 solvePnP로 camera frame 기준 marker center publish
    """

    def __init__(self):
        super().__init__("aruco_zed_node")

        # ==================================================
        # Parameters
        # ==================================================
        self.declare_parameter("color_topic", "/zedm/zed_node/left/image_rect_color/compressed")
        self.declare_parameter("camera_info_topic", "/zedm/zed_node/left/camera_info")

        self.declare_parameter("start_topic", "/aruco_zed_start")
        self.declare_parameter("target_item_topic", "/target_item_name")
        self.declare_parameter("target_aruco_id_topic", "/target_aruco_id")

        self.declare_parameter("point_output_topic", "/aruco/marker_3d_zed")
        self.declare_parameter("debug_output_topic", "/aruco/debug_image_zed")

        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("default_target_marker_id", -1)

        # 실제 인쇄된 마커의 검은 테두리 바깥쪽 한 변 길이 [m]
        self.declare_parameter("marker_size", 0.04)

        # item DB fallback용. Local PC에 DB가 없더라도 /target_aruco_id를 받으면 동작함.
        self.declare_parameter("use_item_database", False)
        self.declare_parameter("item_database_path", "")
        self.declare_parameter("config_package", "master_capstone")
        self.declare_parameter("item_database_file", "item_database.yaml")

        self.declare_parameter("debug", True)
        self.declare_parameter("draw_camera_center", True)
        self.declare_parameter("camera_center_cross_size", 15)

        # ==================================================
        # Fetch Parameters
        # ==================================================
        self.color_topic = self.get_parameter("color_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.start_topic = self.get_parameter("start_topic").value
        self.target_item_topic = self.get_parameter("target_item_topic").value
        self.target_aruco_id_topic = self.get_parameter("target_aruco_id_topic").value

        self.point_output_topic = self.get_parameter("point_output_topic").value
        self.debug_output_topic = self.get_parameter("debug_output_topic").value

        self.target_marker_id = int(self.get_parameter("default_target_marker_id").value)
        self.marker_size = float(self.get_parameter("marker_size").value)

        self.use_item_database = bool(self.get_parameter("use_item_database").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.draw_camera_center = bool(self.get_parameter("draw_camera_center").value)
        self.camera_center_cross_size = int(self.get_parameter("camera_center_cross_size").value)

        self.active = False
        self.current_item_name = ""
        self.current_item = None
        self.camera_info = None

        # ==================================================
        # QoS
        # ==================================================
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================================================
        # DB
        # ==================================================
        self.item_db = {}
        if self.use_item_database:
            self.item_db = self.load_item_database()

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

        # ==================================================
        # Object Points for solvePnP
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # ==================================================
        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s, -s, 0.0],
            [ s, -s, 0.0],
            [ s,  s, 0.0],
            [-s,  s, 0.0],
        ], dtype=np.float32)

        # ==================================================
        # Subscribers
        # ==================================================
        self.sub_start = self.create_subscription(
            Bool,
            self.start_topic,
            self.start_cb,
            self.qos_cmd,
        )

        self.sub_target_item = self.create_subscription(
            String,
            self.target_item_topic,
            self.target_item_cb,
            self.qos_cmd,
        )

        self.sub_target_aruco_id = self.create_subscription(
            Int32,
            self.target_aruco_id_topic,
            self.target_aruco_id_cb,
            self.qos_cmd,
        )

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

        # ==================================================
        # Publishers
        # ==================================================
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
        self.get_logger().info("ARUCO_ZED node started")
        self.get_logger().info("Master-controlled ArUco detection enabled")
        self.get_logger().info(f"start_topic           : {self.start_topic}")
        self.get_logger().info(f"target_item_topic     : {self.target_item_topic}")
        self.get_logger().info(f"target_aruco_id_topic : {self.target_aruco_id_topic}")
        self.get_logger().info(f"color_topic           : {self.color_topic}")
        self.get_logger().info(f"camera_info_topic     : {self.camera_info_topic}")
        self.get_logger().info(f"point_output_topic    : {self.point_output_topic}")
        self.get_logger().info(f"marker_size           : {self.marker_size} m")
        self.get_logger().info(f"default target ID     : {self.target_marker_id}")
        self.get_logger().info(f"use_item_database     : {self.use_item_database}")
        self.get_logger().info("========================================")

    # ==================================================
    # DB
    # ==================================================
    def resolve_database_path(self):
        explicit_path = str(self.get_parameter("item_database_path").value).strip()
        if explicit_path:
            return os.path.expanduser(explicit_path)

        config_package = self.get_parameter("config_package").value
        item_database_file = self.get_parameter("item_database_file").value

        package_share = get_package_share_directory(config_package)
        return os.path.join(package_share, "config", item_database_file)

    def load_item_database(self):
        try:
            yaml_path = self.resolve_database_path()
        except Exception as e:
            self.get_logger().warn(f"Cannot resolve item database path: {repr(e)}")
            return {}

        if not os.path.exists(yaml_path):
            self.get_logger().warn(
                f"item_database.yaml not found: {yaml_path}. "
                f"ARUCO_ZED ignores local DB for runtime target fields; /target_aruco_id from master is authoritative."
            )
            return {}

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            items = data.get("items", {})
            self.get_logger().info(f"Loaded item database: {yaml_path}")
            self.get_logger().info(f"Number of items: {len(items)}")
            return items

        except Exception as e:
            self.get_logger().warn(f"Failed to load item database: {repr(e)}")
            return {}

    def find_item_by_name(self, query: str):
        q = query.strip().lower()
        if not q:
            return None

        for item_key, info in self.item_db.items():
            candidates = [
                str(item_key),
                str(info.get("item_id", "")),
                str(info.get("product_name", "")),
            ]

            for alias in info.get("aliases", []):
                candidates.append(str(alias))

            candidates_norm = [
                c.strip().lower()
                for c in candidates
                if c is not None and str(c).strip() != ""
            ]

            if q in candidates_norm:
                out = dict(info)
                out["item_key"] = item_key
                return out

        return None

    # ==================================================
    # Master callbacks
    # ==================================================
    def start_cb(self, msg: Bool):
        self.active = bool(msg.data)

        if self.active:
            self.get_logger().info(
                f"[START] /aruco_zed_start true. "
                f"target_marker_id={self.target_marker_id}, item='{self.current_item_name}'"
            )
        else:
            self.get_logger().info("[STOP] /aruco_zed_start false. Detection paused.")

    def target_item_cb(self, msg: String):
        """Store item name only.

        IMPORTANT:
          This node must trust the master target topics as the single source
          of truth. Therefore /target_item_name is used only for logging.
          The target marker ID is updated only by /target_aruco_id.

        This prevents robot PC / operator PC YAML mismatches from overwriting
        the target marker id after the master already published it.
        """
        item_name = msg.data.strip()
        self.current_item_name = item_name
        self.current_item = {
            "item_key": item_name,
            "item_id": item_name,
            "product_name": item_name,
        }

        self.get_logger().info(
            "\n"
            "[TARGET ITEM UPDATED - MASTER TOPIC ONLY]\n"
            f"input          : {item_name}\n"
            "source         : /target_item_name only; local item DB ignored\n"
            f"target aruco_id: {self.target_marker_id}"
        )

    def target_aruco_id_cb(self, msg: Int32):
        self.target_marker_id = int(msg.data)

        self.get_logger().info(
            f"[TARGET ARUCO ID UPDATED] target_marker_id={self.target_marker_id}"
        )

    # ==================================================
    # Sensor callbacks
    # ==================================================
    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def color_cb(self, msg: CompressedImage):
        if not self.active:
            return

        if self.camera_info is None:
            return

        if self.target_marker_id < 0:
            if self.debug:
                self.get_logger().warn(
                    "No target_marker_id. Publish /target_aruco_id or valid /target_item_name first."
                )
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f"Decoding failed: {repr(e)}")
            return

        overlay = bgr.copy()
        self._draw_camera_center(overlay)

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
            self.get_logger().error(f"ArUco detection failed: {repr(e)}")
            self._publish_debug(overlay, msg.header)
            return

        if ids is None or len(ids) == 0:
            self._publish_debug(overlay, msg.header)
            return

        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # rectified image 기준
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        found_target = False

        for idx, marker_id_raw in enumerate(ids.flatten()):
            marker_id = int(marker_id_raw)

            if marker_id != self.target_marker_id:
                continue

            found_target = True
            corner_points = corners[idx][0]

            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corner_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                self.get_logger().warn(f"solvePnP failed for ID={marker_id}")
                continue

            x_m = float(tvec[0][0])
            y_m = float(tvec[1][0])
            z_m = float(tvec[2][0])

            out = PointStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.camera_info.header.frame_id
            out.point.x = x_m
            out.point.y = y_m
            out.point.z = z_m
            self.pub_point.publish(out)

            cv2.drawFrameAxes(
                overlay,
                K,
                dist_coeffs,
                rvec,
                tvec,
                self.marker_size * 0.8,
            )

            u = int(np.mean(corner_points[:, 0]))
            v = int(np.mean(corner_points[:, 1]))

            cv2.putText(
                overlay,
                f"Target ID {marker_id} Z: {z_m:.3f}m",
                (u, max(20, v - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if self.debug:
                self.get_logger().info(
                    f"[ARUCO TARGET FOUND] "
                    f"ID={marker_id}, "
                    f"camera_xyz=({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) m, "
                    f"published={self.point_output_topic}"
                )

        if self.debug and not found_target:
            detected_ids = [int(x) for x in ids.flatten()]
            self.get_logger().info(
                f"Target ID={self.target_marker_id} not found. detected_ids={detected_ids}"
            )

        self._publish_debug(overlay, msg.header)

    # ==================================================
    # Debug image
    # ==================================================
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
            debug_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            debug_msg.header = header
            self.pub_debug.publish(debug_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ArucoZedNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()