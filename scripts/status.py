#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
morai_sensor_publisher.py
────────────────────────────────────────────────────────────────────────────
• MORAI GPSMessage   → status.now_position   (PoseStamped)
• IMU                → status.now_heading    (deg)
• EgoVehicleStatus   → status.now_speed      (kph)
• RViz 화살표        → 차량의 현재 위치·방향 시각화  (visualization_msgs/Marker)

출력:
  • /status           (custom_msgs/msg/status)
  • /vehicle_heading  (visualization_msgs/Marker)  ← RViz에서 “Add → Marker” 로 확인
ROS 1 Noetic 전용
"""

import math
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from morai_msgs.msg import GPSMessage, EgoVehicleStatus
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
from pyproj import CRS, Transformer

# 파일명이 소문자 status.msg 이므로, 메시지 클래스도 소문자 status
from custom_msgs.msg import status


# ────────────────────────────────────────────────
#                보조 함수
# ────────────────────────────────────────────────
def compute_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def wrap_pm180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# ────────────────────────────────────────────────
#                메인 클래스
# ────────────────────────────────────────────────
class MoraiSensorPublisher(object):
    def __init__(self):
        rospy.init_node("morai_sensor_publisher", anonymous=False)

        # ── 파라미터 ───────────────────────────
        self.gps_topic_in         = rospy.get_param("~gps_topic_in",    "/gps")
        self.imu_topic_in         = rospy.get_param("~imu_topic_in",    "/imu")
        self.status_in_topic      = rospy.get_param("~status_topic_in", "/Competition_topic")
        self.frame_id             = rospy.get_param("~frame_id",        "map")
        self.yaw_offset_deg       = rospy.get_param("~yaw_offset_deg",  0.0)
        self.use_first_alt_as_ref = rospy.get_param("~use_first_alt_as_ref", True)
        self.arrow_length         = rospy.get_param("~arrow_length",    2.0)   # RViz 화살표 길이 [m]

        # ── 내부 상태 ───────────────────────────
        self.current_pose  = None   # PoseStamped
        self.current_yaw   = None   # float (deg)
        self.current_speed = None   # float (kph)
        self._have_alt_ref = not self.use_first_alt_as_ref
        self._east_offset  = None
        self._north_offset = None
        self._transformer  = None

        # ── 퍼블리셔 ────────────────────────────
        self.pub_status = rospy.Publisher("/status", status, queue_size=10)
        self.pub_marker = rospy.Publisher("/vehicle_heading", Marker, queue_size=1)

        # ── 서브스크라이버 ──────────────────────
        rospy.Subscriber(self.gps_topic_in,    GPSMessage,       self.cb_gps,   queue_size=50)
        rospy.Subscriber(self.imu_topic_in,    Imu,              self.cb_imu,   queue_size=50)
        rospy.Subscriber(self.status_in_topic, EgoVehicleStatus, self.cb_speed, queue_size=50)

        # ── 주기적 발행 타이머 (10 Hz) ───────────
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_status)

        rospy.loginfo("[morai_sensor_publisher] 노드 시작")
        rospy.spin()

    # ────────────────────────────────────────
    #                콜백
    # ────────────────────────────────────────
    @staticmethod
    def _get_field(msg, camel: str, snake: str):
        """MORAI 메시지 버전에 따라 필드명이 다를 수 있어 대응"""
        return getattr(msg, camel) if hasattr(msg, camel) else getattr(msg, snake, None)

    def _ensure_transformer(self, lon: float, lat: float):
        if self._transformer is None:
            epsg = compute_epsg_from_lonlat(lon, lat)
            self._transformer = Transformer.from_crs(
                CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True
            )
            rospy.loginfo("[GPS] UTM EPSG:%d 변환기 생성", epsg)

    def cb_gps(self, msg: GPSMessage):
        if self._east_offset is None or self._north_offset is None:
            self._east_offset  = self._get_field(msg, "eastOffset",  "east_offset")  or 0.0
            self._north_offset = self._get_field(msg, "northOffset", "north_offset") or 0.0
        if not self._have_alt_ref:
            self.alt_ref      = float(msg.altitude)
            self._have_alt_ref = True

        lon, lat, alt = float(msg.longitude), float(msg.latitude), float(msg.altitude)
        self._ensure_transformer(lon, lat)
        E, N = self._transformer.transform(lon, lat)
        x = E - self._east_offset
        y = N - self._north_offset
        z = alt - self.alt_ref

        pose = PoseStamped()
        pose.header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z  # 필요하면 0.0 으로 고정 가능
        pose.pose.orientation.w = 1.0  # 방향은 별도로 화살표로 표시

        self.current_pose = pose

    def cb_imu(self, msg: Imu):
        qx, qy, qz, qw = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        if (qx, qy, qz, qw) == (0.0, 0.0, 0.0, 0.0):
            return
        yaw_rad = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        self.current_yaw = wrap_pm180(math.degrees(yaw_rad) + self.yaw_offset_deg)

    def cb_speed(self, msg: EgoVehicleStatus):
        self.current_speed = float(msg.velocity.x) * 3.6  # m/s → kph

    # ────────────────────────────────────────
    #           상태·Marker 퍼블리시
    # ────────────────────────────────────────
    def publish_status(self, event):
        if (
            self.current_pose is None
            or self.current_yaw is None
            or self.current_speed is None
        ):
            return

        # 1) /status 메시지
        st = status()
        st.now_position = self.current_pose
        st.now_heading  = self.current_yaw
        st.now_speed    = self.current_speed
        self.pub_status.publish(st)

        # 2) RViz 화살표 Marker
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        marker.ns = "vehicle_heading"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 위치
        marker.pose.position.x = self.current_pose.pose.position.x
        marker.pose.position.y = self.current_pose.pose.position.y
        marker.pose.position.z = 0.0  # z 고정

        # 방향
        yaw_rad = math.radians(self.current_yaw)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw_rad)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        # 크기: scale.x = 길이, scale.y = 화살폭, scale.z = 화살높이
        marker.scale.x = self.arrow_length
        marker.scale.y = 0.2 * self.arrow_length
        marker.scale.z = 0.2 * self.arrow_length

        # 색상 (빨강)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0)  # 0 = 영구
        self.pub_marker.publish(marker)


# ────────────────────────────────────────────────
#                    main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        MoraiSensorPublisher()
    except rospy.ROSInterruptException:
        pass

