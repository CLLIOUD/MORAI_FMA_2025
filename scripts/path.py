#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
path_publisher.py — WaypointArray + (RViz 시각화: Path & MarkerArray)
"""

from pathlib import Path
import math
import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped, Pose
from nav_msgs.msg import Path as PathMsg
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from scipy.interpolate import interp1d

from custom_msgs.msg import waypoint, waypointarray

ABS_MAX_SPEED = 49.5 / 3.6  # ≈10.97 m/s  (원본 유지)


def find_catkin_ws() -> Path:
    """
    스크립트 위치 및 CWD 기준으로 'catkin_ws' 디렉터리를 찾아 반환.
    못 찾으면 None 반환.
    """
    script_dir = Path(__file__).resolve().parent
    for p in (script_dir, *script_dir.parents):
        if p.name == 'catkin_ws':
            return p
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if p.name == 'catkin_ws':
            return p
        if (p / 'catkin_ws').is_dir():
            return p / 'catkin_ws'
    return None


def load_path(file_path: Path):
    """
    텍스트 파일에서 x y [z] 형식 읽어 geometry_msgs/Point 리스트로 반환.
    z 값이 없으면 0.0으로 설정.
    """
    pts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) >= 3 else 0.0
            pts.append(Point(x=x, y=y, z=z))
    return pts


def resample_by_distance(pts, step=0.5):
    """
    pts: Point 리스트
    step: 재샘플링 간격(m)
    → 등간격으로 보간된 Point 리스트 반환.
    """
    if len(pts) < 2:
        return pts[:]

    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    ds = np.hypot(np.diff(xs), np.diff(ys))
    ss = np.concatenate(([0.0], np.cumsum(ds)))
    total_len = ss[-1]
    num = int(math.ceil(total_len / step))
    s_sample = np.linspace(0.0, total_len, num + 1)
    fx = interp1d(ss, xs)
    fy = interp1d(ss, ys)

    resampled = []
    for s in s_sample:
        x = float(fx(s))
        y = float(fy(s))
        resampled.append(Point(x=x, y=y, z=0.0))
    return resampled


def compute_curvature_radius(p_prev, p, p_next):
    """
    세 점으로부터 곡률 반경 계산.
    p_prev, p, p_next: geometry_msgs/Point
    """
    a = math.hypot(p.x - p_prev.x, p.y - p_prev.y)
    b = math.hypot(p_next.x - p.x, p_next.y - p.y)
    c = math.hypot(p_prev.x - p_next.x, p_prev.y - p_next.y)
    s = (a + b + c) / 2.0
    area = max(s * (s - a) * (s - b) * (s - c), 0.0)
    area = math.sqrt(area)
    if area < 1e-6:
        return float('inf')
    return (a * b * c) / (4.0 * area)


def plan_speed_spatial(pts, max_speed, min_speed,
                       a_lat_max, a_acc, a_dec,
                       curv_window_m, start_speed, goal_speed):
    """
    공간 기반 속도 플래닝
    - 횡가속도 제한(a_lat_max)
    - 전체 속도 한계
    - 가속/감속 제한
    - 시작/목표 속도
    """
    N = len(pts)
    speeds = [max_speed] * N

    # 횡가속도 제한
    for i in range(1, N - 1):
        R = compute_curvature_radius(pts[i - 1], pts[i], pts[i + 1])
        v_lat = max_speed if math.isinf(R) else math.sqrt(a_lat_max * R)
        speeds[i] = min(speeds[i], v_lat)

    # 전체 속도 한계 적용
    speeds = [min(max(v, min_speed), max_speed) for v in speeds]

    # 시작/목표 속도 설정
    speeds[0] = start_speed
    speeds[-1] = goal_speed

    # 전진 패스: 가속도 제한
    for i in range(1, N):
        ds = math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
        v_prev = speeds[i - 1]
        v_max = math.sqrt(v_prev * v_prev + 2.0 * a_acc * ds)
        speeds[i] = min(speeds[i], v_max)

    # 후진 패스: 감속도 제한
    for i in range(N - 2, -1, -1):
        ds = math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y)
        v_next = speeds[i + 1]
        v_max = math.sqrt(v_next * v_next + 2.0 * a_dec * ds)
        speeds[i] = min(speeds[i], v_max)

    return speeds


def main():
    rospy.init_node('path_publisher', anonymous=False)

    # 파라미터 읽기 (원본 유지)
    resample_step = rospy.get_param('~resample_step', 0.5)
    max_sp = rospy.get_param('~max_speed', ABS_MAX_SPEED)
    min_sp = rospy.get_param('~min_speed', ABS_MAX_SPEED * 0.3)
    a_lat = rospy.get_param('~a_lat_max', 0.8)   # 원본 주석 유지
    a_acc = rospy.get_param('~a_acc', 4.0)
    a_dec = rospy.get_param('~a_dec', 2.0)       # 원본 2.0 유지
    curv_win = rospy.get_param('~curv_window_m', 20.0)
    start_sp = rospy.get_param('~start_speed', ABS_MAX_SPEED)
    goal_sp = rospy.get_param('~goal_speed', 0.0)

    # 경로 파일 읽기 (원본 경로 유지)
    ws = find_catkin_ws()
    if ws is None:
        rospy.logerr("catkin_ws 폴더를 찾을 수 없습니다.")
        return
    path_file = ws / 'path' / '25hl_global_path_ver2.txt'
    if not path_file.is_file():
        rospy.logerr(f"파일이 없습니다: {path_file}")
        return

    raw_pts = load_path(path_file)
    dense_pts = resample_by_distance(raw_pts, step=resample_step)

    # 속도 플래닝 (원본 로직 유지)
    speeds_dense = plan_speed_spatial(
        dense_pts,
        max_speed=max_sp, min_speed=min_sp,
        a_lat_max=a_lat, a_acc=a_acc, a_dec=a_dec,
        curv_window_m=curv_win,
        start_speed=start_sp, goal_speed=goal_sp
    )

    # ─────────────────────────────────────
    # 퍼블리셔: WaypointArray + (추가) Path/MarkerArray
    # ─────────────────────────────────────
    pub_wparr = rospy.Publisher('global_waypoints_src', waypointarray, queue_size=1, latch=True)
    pub_path = rospy.Publisher('global_path', PathMsg, queue_size=1, latch=True)
    pub_speed_markers = rospy.Publisher('path_speed_markers', MarkerArray, queue_size=1, latch=True)

    # WaypointArray 퍼블리시 (원본 유지)
    wa = waypointarray()
    for pt, v in zip(dense_pts, speeds_dense):
        wa.waypoints.append(waypoint(x=pt.x, y=pt.y, speed=v * 3.6))
    rospy.sleep(0.2)
    pub_wparr.publish(wa)

    # Path 메시지 생성 (예시 코드 방식)
    def make_path_msg(points):
        path = PathMsg()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for p in points:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose = Pose()
            ps.pose.position = p
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    # MarkerArray로 속도 바 시각화 (예시 코드 방식)
    def make_speed_marker_array(points, speeds, v2z_scale=0.3):
        ma = MarkerArray()
        header = Header(frame_id='map', stamp=rospy.Time.now())
        for i, (p, v) in enumerate(zip(points, speeds)):
            m = Marker()
            m.header = header
            m.ns = 'speed_bars'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position = Point(p.x, p.y, v * v2z_scale / 2.0)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = v * v2z_scale
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.6
            ma.markers.append(m)
        return ma

    # 한 번만 퍼블리시 (latch)
    rospy.sleep(0.3)
    pub_path.publish(make_path_msg(dense_pts))
    pub_speed_markers.publish(make_speed_marker_array(dense_pts, speeds_dense))

    rospy.loginfo("Published global_waypoints (WaypointArray), global_path (Path), path_speed_markers (MarkerArray).")
    rospy.spin()


if __name__ == '__main__':
    main()

