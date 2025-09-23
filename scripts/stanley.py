#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stanley_controller_seq_dynamic_path.py
────────────────────────────────────────────────────────────────────────────
• Stanley 조향 + PI-D 속도 제어
• /global_waypoints: **매 수신 시 갱신**
    - 새 경로 수신 시: (1) 경로 배열 재계산, (2) j_idx '현재 위치 근처'로 재시드,
      (3) STOP_POINTS를 해당 경로 인덱스로 **재계산**, (4) DWELL 지점 경로 인덱스 매칭
• 신호 정지(2-단계) + GREEN 해제 옵션
• /stop(String), /warn(Bool): 구독
• 전방 웨이포인트 탐색: 이전 인덱스 이후의 짧은 창
• 헤딩 고정 모드(기본): 기준점 반경 진입 시 켜고, 이탈/거리 이상치 시 자동 해제 가드
• (중요) DWELL(정지 후 재출발)은 **최우선**: 진행 중이면 신호/헤딩고정/일반주행 모두 무시
• (옵션) HH 중 신호/정지 무시 스위치
• (수정) HH 진입 시 **파라미터로 지정한 고정 조향각을 래치**하여 유지
    - Pure Pursuit 계산 제거, 사용자가 각도를 파라미터로 지정
    - /stop TRUE 들어오면 기존 정지 래치/브레이크 로직 그대로 작동
• (신규) /status 좌표 글리치 가드(범위/점프속도) 추가
ROS 1 Noetic
"""

import math
import numpy as np
import rospy
import threading
from math import cos, sin, atan2, sqrt, pi, radians, degrees

from geometry_msgs.msg import Point
from custom_msgs.msg import status, waypointarray
from std_msgs.msg import String, Bool, Float32
from morai_msgs.msg import CtrlCmd


def wrap_angle(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


# 신호 제어용 정지 후보 (경로 인덱스로 매핑)
STOP_POINTS = [
    (771.2772, -827.5197),
    (773.2772, -827.5197),
    (775.2772, -827.5197),
    (777.2772, -827.5197),
    (779.2772, -827.5197),
    (616.0377, -725.0411),
    (615.0377, -722.0411),
    (612.5377, -721.0411),

]


class StanleyController:
    def __init__(self):
        rospy.init_node("stanley_node", anonymous=True)

        # 경로/상태
        self.path_received = False
        self.path = None
        self.xs = self.ys = self.vs = self.s = None
        self.s_end = None
        self.last_xy = None
        self.N = 0
        self.j_idx = 0
        self._path_lock = threading.RLock()

        # 신호/정지 파라미터
        self.stop_points_xy = rospy.get_param("~stop_points", STOP_POINTS)
        self.stop_assoc_thr = float(rospy.get_param("~stop_assoc_threshold_m", 3.0))

        # 신호 2-단계
        self.tl_active_radius_m = float(rospy.get_param("~tl_active_radius_m", 6.0))
        self.tl_decel_start_m  = float(rospy.get_param("~tl_decel_start_m", 6.0))
        self.tl_hard_zone_m    = float(rospy.get_param("~tl_hard_zone_m", 0.8))
        self.tl_hard_brake_cmd = float(rospy.get_param("~tl_hard_brake_cmd", 1.0))
        self.tl_pass_margin_m  = float(rospy.get_param("~tl_pass_margin_m", 2.0))
        self.tl_require_green  = bool(rospy.get_param("~tl_require_green", True))
        self.tl_lock_active = False
        self.tl_lock_idx = None

        # 옵션 (정책 플래그)
        self.treat_unknown_as_red = bool(rospy.get_param("~treat_unknown_as_red", True))
        self.treat_yellow_as_red  = bool(rospy.get_param("~treat_yellow_as_red",  True))

        self.stop_indices = []
        self.stop_idx_to_xy = {}

        # 런타임 상태/제어
        self.traffic_state = None
        self.stop_line_state = None
        self.warn_state = None

        self.search_ahead = int(rospy.get_param("~search_ahead", 30))

        self.pub = rospy.Publisher("/ctrl_cmd_0", CtrlCmd, queue_size=1)
        self.cmd = CtrlCmd()
        self.cmd.longlCmdType = 1  # accel/brake 모드

        self.status_received = False
        self.cur_pos = None
        self.yaw = 0.0
        self.kph = 0.0

        # Stanley
        self.k = float(rospy.get_param("~stanley_k", 1.0))
        self.v_eps = float(rospy.get_param("~vt_eps", 1.0))
        self.max_deg = float(rospy.get_param("~max_wheel_deg", 40.0))
        self.max_rad = self.max_deg * pi / 180.0
        self.invert_steer = bool(rospy.get_param("~invert_steer", False))

        # PI-D 속도 제어기
        self.v_kp = float(rospy.get_param("~v_kp", 1.0 / 30.0))
        self.v_ki = float(rospy.get_param("~v_ki", 0.01))
        self.v_kd = float(rospy.get_param("~v_kd", 0.005))
        self.u_dead = float(rospy.get_param("~u_dead", 0.03))
        self.u_slew_rate = float(rospy.get_param("~u_slew_rate", 2.0))
        self.i_limit = float(rospy.get_param("~i_limit", 5.0))

        self.abs_kph_cap = float(rospy.get_param("~abs_kph_cap", 35.0))
        self.loop_hz = float(rospy.get_param("~loop_hz", 15.0))

        # 종점 정지/브레이크 홀드
        self.stop_window_m = float(rospy.get_param("~stop_window_m", 5.0))
        self.stop_radius_m = float(rospy.get_param("~stop_radius_m", 2.5))
        self.brake_hold_kph = float(rospy.get_param("~brake_hold_kph", 1.5))
        self.brake_hold_cmd = float(rospy.get_param("~brake_hold_cmd", 1.0))

        # PI-D 내부 상태
        self.int_v = 0.0
        self.prev_kph = None
        self.prev_t = None
        self.last_u = 0.0
        self.last_sat = False

        # 헤딩 고정 모드
        self.heading_hold_enabled = bool(rospy.get_param("~heading_hold_enabled", True))
        hh_xy = rospy.get_param("~heading_hold_xy", [609.0, -878.60])
        self.heading_hold_xy = (float(hh_xy[0]), float(hh_xy[1]))
        self.heading_hold_radius_m = float(rospy.get_param("~heading_hold_radius_m", 1.0))
        self.heading_hold_exit_radius_m = float(rospy.get_param("~heading_hold_exit_radius_m", 10.0))
        self.heading_hold_disable_if_distance_huge = bool(rospy.get_param("~heading_hold_disable_if_distance_huge", True))
        self.heading_hold_huge_distance_m = float(rospy.get_param("~heading_hold_huge_distance_m", 100000.0))
        self.heading_hold_deg = float(rospy.get_param("~heading_hold_deg", 51.2))
        self.heading_hold_active = False
        self.heading_hold_target_kph_param = float(rospy.get_param("~heading_hold_speed_kph", -1.0))
        self.heading_hold_target_kph = None
        self.heading_hold_k_gain = float(rospy.get_param("~heading_hold_k_gain", 1.0))

        # ── HH 중 /stop 래치 정책
        self.hh_stop_delay_s    = float(rospy.get_param("~heading_hold_stop_delay_s", 5.0))  # 기본: 즉시 래치
        self.hh_stop_ignore_s   = float(rospy.get_param("~heading_hold_stop_ignore_s", 2.0)) # 기본: 무시 구간 없음
        self.hh_stop_permanent  = bool(rospy.get_param("~heading_hold_stop_permanent", True))
        self.hh_stop_armed      = False
        self.hh_stop_deadline   = None
        self.hh_stop_engaged    = False
        self.hh_enter_time      = None

        # ── HH 중 신호/정지 무시 스위치 (요청 사항: /stop은 그대로 → 무시 OFF가 기본)
        self.hh_ignore_stop_in_hh = bool(rospy.get_param("~hh_ignore_stop_in_hh", False))  # False: HH 중에도 /stop 처리
        self.tl_disable_in_hh     = bool(rospy.get_param("~tl_disable_in_hh", True))       # True면 신호락 무시

        # ── /status 글리치 가드
        self.status_max_abs_xy_m     = float(rospy.get_param("~status_max_abs_xy_m", 100000.0))
        self.status_glitch_speed_mps = float(rospy.get_param("~status_glitch_speed_mps", 80.0))
        self._prev_good_xy = None
        self._prev_t = None

        # ── DWELL(정지 후 재출발)
        dpxy = rospy.get_param("~dwell_point_xy", [606.9505, -889.418])
        self.dwell_point_xy = (float(dpxy[0]), float(dpxy[1]))
        self.dwell_assoc_thr_m   = float(rospy.get_param("~dwell_assoc_threshold_m", 3.0))
        self.dwell_trigger_radius_m = float(rospy.get_param("~dwell_trigger_radius_m", 1.0))
        self.dwell_hold_seconds  = float(rospy.get_param("~dwell_hold_seconds", 10.0))
        self.dwell_idx = None
        self.dwell_idx_xy = None
        self.dwell_engaged = False
        self.dwell_done = False
        self.dwell_deadline = None

        # ── (신규) HH “고정 조향각” 설정 (Pure Pursuit 제거)
        # 진입 시점에 한 번 읽어 래치하고, HH가 끝날 때까지 유지합니다.
        self.hh_use_fixed_steer = bool(rospy.get_param("~hh_use_fixed_steer", True))
        self.hh_fixed_steer_deg = float(rospy.get_param("~hh_fixed_steer_deg", -0.1))
        self.hh_fixed_steer_latch = None  # rad, HH 진입 시 파라미터를 읽어 래치

        # (신규) Follow(선행차 추종) 파라미터
        self.follow_delta_kmh   = float(rospy.get_param("~follow_delta_kmh", 5.0))   # v_lead - 5 km/h
        self.follow_T_gap       = float(rospy.get_param("~follow_T_gap", 1.2))       # [s]
        self.follow_min_gap_m   = float(rospy.get_param("~follow_min_gap", 5.0))     # [m]
        self.follow_v_min_kmh   = float(rospy.get_param("~follow_v_min_kmh", 5.0))
        self.follow_enable      = bool(rospy.get_param("~follow_enable", True))
        self.lead_gap_m = None
        self.lead_v_obj = None
        self.lead_is_dynamic = False

        # Subscribers
        rospy.Subscriber("/traffic", String, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/stop", String, self.stop_line_callback, queue_size=1)
        rospy.Subscriber("/warn", Bool, self.warn_callback, queue_size=1)
        rospy.Subscriber("/status", status, self.status_callback, queue_size=20)
        rospy.Subscriber("/global_waypoints", waypointarray, self._waypoints_callback, queue_size=1)
        rospy.loginfo(">> /global_waypoints 구독 시작 (매 수신 갱신)")

        # (신규) 선행차 정보
        rospy.Subscriber("/lead_gap_m",   Float32, lambda m: setattr(self, "lead_gap_m", float(m.data)), queue_size=1)
        rospy.Subscriber("/lead_v_obj",   Float32, lambda m: setattr(self, "lead_v_obj", float(m.data)), queue_size=1)
        rospy.Subscriber("/lead_is_dynamic", Bool, lambda m: setattr(self, "lead_is_dynamic", bool(m.data)), queue_size=1)

    # ────────────────────────────────────────
    # 헬퍼
    # ────────────────────────────────────────
    def _is_stop_signal(self, sig: str) -> bool:
        s = (sig or "").upper()
        if s == "RED": return True
        if s == "YELLOW": return self.treat_yellow_as_red
        if s == "UNKNOWN": return self.treat_unknown_as_red
        return False

    # ────────────────────────────────────────
    # 콜백들
    # ────────────────────────────────────────
    def traffic_callback(self, msg: String):
        self.traffic_state = (msg.data or "").strip().upper()

    def stop_line_callback(self, msg: String):
        s = (msg.data or "").strip().upper()
        self.stop_line_state = s in ("TRUE", "1", "YES", "ON")

    def warn_callback(self, msg: Bool):
        self.warn_state = bool(msg.data)

    def status_callback(self, msg: status):
        # 시간 (header 없으면 now)
        try:
            t = msg.now_position.header.stamp
            if t is None or t.to_sec() == 0.0:
                t = rospy.Time.now()
        except Exception:
            t = rospy.Time.now()

        # 좌표
        try:
            x = msg.now_position.pose.position.x
            y = msg.now_position.pose.position.y
        except Exception:
            return

        # 글리치 가드: 값 범위
        if abs(x) > self.status_max_abs_xy_m or abs(y) > self.status_max_abs_xy_m:
            rospy.logwarn_throttle(0.5, "[stanley][STATUS] glitch ignored: abs(x/y) too large x=%.2f y=%.2f", x, y)
            return

        # 글리치 가드: 점프 속도
        if self._prev_good_xy is not None and self._prev_t is not None:
            dt = max((t - self._prev_t).to_sec(), 1e-3)
            vx = (x - self._prev_good_xy[0]) / dt
            vy = (y - self._prev_good_xy[1]) / dt
            spd_est = math.hypot(vx, vy)
            if spd_est > self.status_glitch_speed_mps:
                rospy.logwarn_throttle(0.5, "[stanley][STATUS] glitch ignored: jump speed=%.1f m/s", spd_est)
                return

        # 정상 반영
        self._prev_good_xy = (x, y)
        self._prev_t = t

        self.cur_pos = msg.now_position.pose.position
        self.yaw = math.radians(msg.now_heading)
        self.kph = msg.now_speed
        self.status_received = True

    # 경로 갱신
    def _waypoints_callback(self, msg: waypointarray):
        if len(msg.waypoints) < 2:
            rospy.logwarn_throttle(2.0, "[stanley] 받은 경로가 2개 미만 -> 무시")
            return

        with self._path_lock:
            self.path = msg
            self.xs = np.array([w.x for w in self.path.waypoints], dtype=float)
            self.ys = np.array([w.y for w in self.path.waypoints], dtype=float)
            self.vs = np.array([float(w.speed) for w in self.path.waypoints], dtype=float)

            ds = np.hypot(np.diff(self.xs), np.diff(self.ys))
            self.s = np.concatenate([[0.0], np.cumsum(ds)])
            self.s_end = float(self.s[-1])
            self.last_xy = (self.xs[-1], self.ys[-1])
            self.N = len(self.xs)
            self.path_received = True

            # 현재 위치 근처로 재시드
            if self.cur_pos is not None:
                cx, cy = self.cur_pos.x, self.cur_pos.y
                d2 = (self.xs - cx) ** 2 + (self.ys - cy) ** 2
                self.j_idx = int(np.argmin(d2))
            else:
                self.j_idx = 0

            # STOP_POINTS 매핑
            pts = getattr(self, "stop_points_xy", STOP_POINTS)
            self.stop_indices = self._find_stop_indices(pts)
            self.stop_idx_to_xy = {idx: (self.xs[idx], self.ys[idx]) for idx in self.stop_indices}

            # DWELL 매핑
            self._update_dwell_index_locked()

        rospy.loginfo_throttle(
            1.0,
            "[stanley] 경로 갱신: N=%d, j_idx=%d, stops=%s, dwell_idx=%s",
            self.N, self.j_idx, self.stop_indices if self.stop_indices else "[]",
            (self.dwell_idx if self.dwell_idx is not None else "None")
        )

    def _update_dwell_index_locked(self):
        if self.xs is None or self.ys is None or self.N == 0 or self.dwell_point_xy is None:
            self.dwell_idx = None
            self.dwell_idx_xy = None
            return
        sx, sy = self.dwell_point_xy
        d2 = (self.xs - sx) ** 2 + (self.ys - sy) ** 2
        idx = int(np.argmin(d2))
        dist = float(math.hypot(self.xs[idx] - sx, self.ys[idx] - sy))
        if dist <= self.dwell_assoc_thr_m:
            self.dwell_idx = idx
            self.dwell_idx_xy = (self.xs[idx], self.ys[idx])
            rospy.loginfo("[stanley] dwell @%d matched (%.2fm) → hold %.1fs",
                          idx, dist, self.dwell_hold_seconds)
        else:
            self.dwell_idx = None
            self.dwell_idx_xy = None
            rospy.logwarn("[stanley] dwell candidate (%.2f,%.2f) no match (min %.2fm > thr %.2fm)",
                          sx, sy, dist, self.dwell_assoc_thr_m)

    def _find_stop_indices(self, pts_xy):
        if not pts_xy or self.xs is None:
            return []
        out = []
        for p in pts_xy:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                rospy.logwarn("[stanley] stop_points 항목 무시: %s", p)
                continue
            sx, sy = float(p[0]), float(p[1])
            d2 = (self.xs - sx) ** 2 + (self.ys - sy) ** 2
            idx = int(np.argmin(d2))
            dist = float(math.hypot(self.xs[idx] - sx, self.ys[idx] - sy))
            if dist <= self.stop_assoc_thr:
                out.append(idx)
                rospy.loginfo("[stanley] stop @%d matched (%.2fm)", idx, dist)
            else:
                rospy.logwarn("[stanley] stop candidate (%.2f,%.2f) no match (min %.2fm > thr %.2fm)",
                              sx, sy, dist, self.stop_assoc_thr)
        return sorted(set(out))

    # PI-D
    def speed_pid_step(self, tgt_kph: float, cur_kph: float, now: rospy.Time):
        if self.prev_t is None:
            dt = 1.0 / self.loop_hz
        else:
            dt = max((now - self.prev_t).to_sec(), 1e-3)

        e = tgt_kph - cur_kph
        d_meas = 0.0 if self.prev_kph is None else (cur_kph - self.prev_kph) / dt

        u_p = self.v_kp * e

        will_inc_sat = self.last_sat and ((self.last_u > 0 and e > 0) or (self.last_u < 0 and e < 0))
        if not will_inc_sat:
            self.int_v += e * dt
            self.int_v = float(np.clip(self.int_v, -self.i_limit, self.i_limit))
        u_i = self.v_ki * self.int_v
        u_d = -self.v_kd * d_meas

        u = u_p + u_i + u_d
        u_sat = float(np.clip(u, -1.0, 1.0))
        self.last_sat = (abs(u - u_sat) > 1e-6)

        max_du = self.u_slew_rate * dt
        u_cmd = float(np.clip(u_sat, self.last_u - max_du, self.last_u + max_du))

        if abs(u_cmd) < self.u_dead:
            u_cmd = 0.0

        self.prev_kph = cur_kph
        self.prev_t = now
        self.last_u = u_cmd

        if u_cmd >= 0.0:
            accel, brake = u_cmd, 0.0
        else:
            accel, brake = 0.0, -u_cmd

        return accel, brake, u_cmd, e, d_meas, u_p, u_i, u_d

    # ────────────────────────────────────────
    # 메인 루프
    # ────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(self.loop_hz)

        while not rospy.is_shutdown():
            if not self.status_received:
                rospy.logwarn_throttle(5.0, "/status 수신 대기 …")
                rate.sleep()
                continue

            if not self.path_received:
                rospy.logwarn_throttle(5.0, "/global_waypoints 수신 대기 … (갱신 모드)")
                rate.sleep()
                continue

            # 경로 스냅샷
            with self._path_lock:
                xs = self.xs.copy()
                ys = self.ys.copy()
                vs = self.vs.copy()
                s  = self.s.copy()
                last_x, last_y = self.last_xy
                N = self.N
                j_idx = self.j_idx
                dwell_idx = self.dwell_idx
                dwell_xy  = self.dwell_idx_xy

            # 현재 위치
            x, y, yaw = self.cur_pos.x, self.cur_pos.y, self.yaw
            c, sy = cos(yaw), sin(yaw)
            T_inv = np.array([[c,  sy,  -(c * x + sy * y)],
                              [-sy, c,   (sy * x - c * y)],
                              [0.0, 0.0, 1.0]])

            # ── (0) 전역 영구 정지 래치 ─────────────────────────────
            if self.hh_stop_engaged and self.hh_stop_permanent:
                self.cmd.steering = 0.0
                self.cmd.accel = 0.0
                self.cmd.brake = float(max(self.tl_hard_brake_cmd, self.brake_hold_cmd, 1.0))
                self.pub.publish(self.cmd)

                print("\033c", end="")
                print("── GLOBAL PERMANENT STOP LATCH ─────────────────────────")
                print(f" kph {self.kph:5.2f} | brake {self.cmd.brake:.2f} (accel 0.0)")
                rate.sleep()
                continue

            # ── (1) DWELL(정지) 최우선 ──────────────────────────────
            dwell_info = "None"
            in_dwell_trigger = False
            now = rospy.Time.now()

            if (not self.dwell_done) and (dwell_idx is not None):
                sx_d, sy_d = dwell_xy if dwell_xy is not None else (xs[dwell_idx], ys[dwell_idx])
                dist_euclid_to_dwell = sqrt((x - sx_d)**2 + (y - sy_d)**2)
                dwell_info = f"idx {dwell_idx} | d_e {dist_euclid_to_dwell:5.2f} m"

                # 트리거
                if (not self.dwell_engaged) and (dist_euclid_to_dwell <= self.dwell_trigger_radius_m):
                    self.dwell_engaged = True
                    self.dwell_deadline = now + rospy.Duration.from_sec(self.dwell_hold_seconds)
                    # PI-D 리셋
                    self.int_v = 0.0; self.last_u = 0.0; self.prev_kph = None; self.prev_t = None
                    rospy.loginfo("[stanley][DWELL] ENTER: hold %.1fs at (%.4f, %.4f)",
                                  self.dwell_hold_seconds, sx_d, sy_d)

                # 진행 중
                if self.dwell_engaged:
                    in_dwell_trigger = True
                    # 만료 → 재출발 허용
                    if now >= self.dwell_deadline:
                        self.dwell_engaged = False
                        self.dwell_done = True
                        self.dwell_deadline = None
                        # 복귀 시 PI-D 리셋
                        self.int_v = 0.0; self.last_u = 0.0; self.prev_kph = None; self.prev_t = None
                        rospy.loginfo("[stanley][DWELL] EXIT: resume driving")

            # DWELL 진행 중이면 무조건 정지
            if in_dwell_trigger:
                self.cmd.steering = 0.0
                accel, brake, *_ = self.speed_pid_step(0.0, self.kph, now)
                accel = 0.0
                brake = max(brake, self.tl_hard_brake_cmd)
                if self.kph <= self.brake_hold_kph:
                    brake = max(brake, self.brake_hold_cmd)
                self.cmd.accel = accel
                self.cmd.brake = brake
                self.pub.publish(self.cmd)

                print("\033c", end="")
                print("── DWELL (STOP) ────────────────────────────────────────")
                left_s = max(0.0, (self.dwell_deadline - now).to_sec()) if self.dwell_deadline else 0.0
                print(f" time_left {left_s:4.1f}s | kph {self.kph:5.2f} | brake {brake:.2f}")
                print(f" dwell_info: {dwell_info}")
                rate.sleep()
                continue

            # ── (2) 헤딩 고정(HH) ──────────────────────────────────
            dd = math.hypot(x - self.heading_hold_xy[0], y - self.heading_hold_xy[1])
            if self.heading_hold_active:
                # 과대거리/이탈 자동 해제
                if (self.heading_hold_disable_if_distance_huge and dd > self.heading_hold_huge_distance_m) \
                   or (dd > self.heading_hold_exit_radius_m):
                    rospy.logwarn("[stanley][HH] auto-exit (dist=%.2fm)", dd)
                    self.heading_hold_active = False
                    # 복귀 시 PI-D 리셋
                    self.int_v = 0.0; self.last_u = 0.0; self.prev_kph = None; self.prev_t = None
                    # 고정각 래치 해제
                    self.hh_fixed_steer_latch = None

            # 진입 판정 (DWELL 끝난 후)
            if (not self.heading_hold_active) and self.heading_hold_enabled and self.dwell_done:
                if dd <= self.heading_hold_radius_m:
                    self.heading_hold_active = True
                    self.hh_enter_time = rospy.Time.now()

                    # 타깃 속도 결정
                    if self.heading_hold_target_kph_param > 0.0:
                        self.heading_hold_target_kph = float(min(self.heading_hold_target_kph_param, self.abs_kph_cap))
                    else:
                        v_nominal_now = None
                        start_tmp = j_idx
                        end_tmp = min(j_idx + self.search_ahead, N - 1)
                        best_j_tmp, best_d2_tmp = None, float("inf")
                        for i in range(start_tmp, end_tmp + 1):
                            pl = T_inv.dot([xs[i], ys[i], 1.0])
                            if pl[0] <= 0.0:
                                continue
                            d2 = pl[0]*pl[0] + pl[1]*pl[1]
                            if d2 < best_d2_tmp:
                                best_j_tmp, best_d2_tmp = i, d2
                        if best_j_tmp is not None:
                            v_nominal_now = float(min(vs[best_j_tmp], self.abs_kph_cap))
                        if v_nominal_now is None or v_nominal_now <= 0.0:
                            v_nominal_now = float(min(max(self.kph, 0.0), self.abs_kph_cap))
                        self.heading_hold_target_kph = v_nominal_now

                    # PI-D 리셋
                    self.int_v = 0.0; self.last_u = 0.0; self.prev_kph = None; self.prev_t = None
                    # 신호락/정지 래치 초기화
                    self.tl_lock_active = False
                    self.tl_lock_idx = None
                    self.hh_stop_armed = False
                    self.hh_stop_deadline = None

                    # ── HH “고정 조향각” 래치: 파라미터 값(deg)을 라디안으로 변환해 한 번만 저장
                    self.hh_fixed_steer_latch = None
                    if self.hh_use_fixed_steer:
                        delta = radians(self.hh_fixed_steer_deg)
                        if self.invert_steer:
                            delta = -delta
                        self.hh_fixed_steer_latch = float(np.clip(delta, -self.max_rad, self.max_rad))

                    rospy.loginfo("[stanley] Heading-Hold ENTER: target_deg=%.1f, speed_kph=%.2f, fixed_steer_latched=%s (%.3f rad)",
                                  self.heading_hold_deg, self.heading_hold_target_kph,
                                  bool(self.hh_fixed_steer_latch is not None),
                                  (self.hh_fixed_steer_latch if self.hh_fixed_steer_latch is not None else 0.0))

            # 헤딩 고정 분기
            if self.heading_hold_active:
                # 조향: “고정 조향각 래치” 우선, 없으면 전통 HH(헤딩오차*k)
                if self.hh_use_fixed_steer and (self.hh_fixed_steer_latch is not None):
                    steer = self.hh_fixed_steer_latch
                    theta_e = 0.0  # 표시용
                else:
                    target_yaw = radians(self.heading_hold_deg)
                    theta_e = wrap_angle(target_yaw - yaw)
                    steer = self.heading_hold_k_gain * theta_e
                    if self.invert_steer:
                        steer = -steer
                    steer = float(np.clip(steer, -self.max_rad, self.max_rad))

                now = rospy.Time.now()

                # HH 중 신호 무시 옵션
                if self.tl_disable_in_hh:
                    self.tl_lock_active = False
                    self.tl_lock_idx = None

                # HH 중 /stop 처리 (요청: 그대로 처리 → 기본 무시하지 않음)
                if not self.hh_ignore_stop_in_hh:
                    if bool(self.stop_line_state) and (not self.hh_stop_engaged):
                        if self.hh_stop_delay_s > 0.0:
                            if not self.hh_stop_armed:
                                self.hh_stop_armed = True
                                self.hh_stop_deadline = now + rospy.Duration.from_sec(self.hh_stop_delay_s)
                                rospy.loginfo("[stanley][HH] /stop=TRUE → %.1fs 후 정지 예약", self.hh_stop_delay_s)
                            elif now >= self.hh_stop_deadline:
                                self.hh_stop_engaged = True
                                self.hh_stop_armed = False
                                self.hh_stop_deadline = None
                                rospy.loginfo("[stanley][HH] 정지 타이머 만료 → 영구 정지 래치")
                        else:
                            # 즉시 래치
                            self.hh_stop_engaged = True
                            self.hh_stop_armed = False
                            self.hh_stop_deadline = None
                            rospy.loginfo("[stanley][HH] /stop=TRUE → 즉시 정지 래치")

                # 목표 속도
                tgt_kph = 0.0 if self.hh_stop_engaged else float(min(max(self.heading_hold_target_kph, 0.0), self.abs_kph_cap))

                # 속도 제어
                accel, brake, u_cmd, e_kph, d_meas, u_p, u_i, u_d = self.speed_pid_step(tgt_kph, self.kph, now)

                # 래치 중 하드 브레이크/홀드
                if self.hh_stop_engaged:
                    accel = 0.0
                    brake = max(brake, self.tl_hard_brake_cmd)
                    if self.kph <= self.brake_hold_kph:
                        brake = max(brake, self.brake_hold_cmd)

                # 퍼블리시
                self.cmd.steering = steer
                self.cmd.accel = accel
                self.cmd.brake = brake
                self.pub.publish(self.cmd)

                # 디버그
                print("\033c", end="")
                print("── Heading-Hold MODE (fixed steer latched:", self.hh_fixed_steer_latch is not None, ") ───────────────")
                if self.hh_fixed_steer_latch is not None:
                    print(f" Fixed steer (rad) {self.hh_fixed_steer_latch:+.4f} | deg {degrees(self.hh_fixed_steer_latch):+.2f}")
                else:
                    print(f" target_yaw {self.heading_hold_deg:6.2f}° | cur_yaw {degrees(yaw):6.2f}° | θ_e {degrees(theta_e):6.2f}°")
                print(f" kph cur/tgt {self.kph:5.2f}/{tgt_kph:5.2f}  accel {accel:.2f}  brake {brake:.2f}")
                print(f" HH switch_xy ({self.heading_hold_xy[0]:.2f},{self.heading_hold_xy[1]:.2f}) | dist {dd:.2f} m "
                      f"(enter_thr {self.heading_hold_radius_m:.2f} / exit_thr {self.heading_hold_exit_radius_m:.2f})")
                print(f" stop_line_state {self.stop_line_state} | stop_armed {self.hh_stop_armed} "
                      f"| engaged {self.hh_stop_engaged} | tl_disabled {self.tl_disable_in_hh} | stop_ignored {self.hh_ignore_stop_in_hh}")
                rate.sleep()
                continue  # 일반 로직 스킵

            # ── (3) 일반 Stanley 경로 추종 ────────────────────────
            start = j_idx
            end = min(j_idx + self.search_ahead, N - 1)
            best_j = None; best_d2 = float("inf"); best_pl = None

            for i in range(start, end + 1):
                pl = T_inv.dot([xs[i], ys[i], 1.0])
                if pl[0] <= 0.0:
                    continue
                d2 = pl[0]*pl[0] + pl[1]*pl[1]
                if d2 < best_d2:
                    best_j, best_d2, best_pl = i, d2, pl

            if best_j is None:
                for i in range(j_idx, N):
                    pl = T_inv.dot([xs[i], ys[i], 1.0])
                    if pl[0] <= 0.0:
                        continue
                    d2 = pl[0]*pl[0] + pl[1]*pl[1]
                    if d2 < best_d2:
                        best_j, best_d2, best_pl = i, d2, pl

            if best_j is not None and best_j >= j_idx:
                with self._path_lock:
                    self.j_idx = best_j
                j = best_j
                plocal = best_pl
            else:
                j = best_j
                plocal = best_pl

            # 종점/정지점 관리
            dist_to_last = sqrt((x - last_x)**2 + (y - last_y)**2)
            if j is not None:
                s_now = float(s[j])
                s_remain = max(0.0, self.s_end - s_now)
            else:
                s_now = float("inf"); s_remain = float("inf")

            at_terminal = (
                (s_remain <= self.stop_window_m) or
                (dist_to_last <= self.stop_radius_m)
            )

            if self.stop_indices and j is not None:
                with self._path_lock:
                    self.stop_indices = [idx for idx in self.stop_indices if idx >= j]

            next_stop_idx = None
            dist_euclid_to_stop = float("inf")
            if j is not None and self.stop_indices:
                for idx in self.stop_indices:
                    if idx >= j:
                        next_stop_idx = idx; break
                if next_stop_idx is not None:
                    sx, sy = self.stop_idx_to_xy.get(next_stop_idx, (xs[next_stop_idx], ys[next_stop_idx]))
                    dist_euclid_to_stop = sqrt((x - sx)**2 + (y - sy)**2)

            # 신호 상태
            sig = (self.traffic_state or "").upper()
            in_outer = (next_stop_idx is not None and dist_euclid_to_stop <= self.tl_active_radius_m)
            in_inner = (in_outer and dist_euclid_to_stop <= self.tl_hard_zone_m)
            stop_like = self._is_stop_signal(sig)

            v_nominal = 0.0 if j is None else float(min(vs[j], self.abs_kph_cap))
            tgt_kph = v_nominal

            if in_outer and not self.tl_lock_active and not in_inner and stop_like:
                tgt_kph = self._ramp_tgt_kph(
                    d=dist_euclid_to_stop,
                    v_nominal_kph=v_nominal,
                    start=self.tl_decel_start_m,
                    stop=self.tl_hard_zone_m
                )

            if in_inner and stop_like:
                self.tl_lock_active = True
                self.tl_lock_idx = next_stop_idx
            elif self.tl_lock_active and self.tl_require_green and sig == "GREEN":
                self.tl_lock_active = False
                self.tl_lock_idx = None
                self.int_v = 0.0; self.last_u = 0.0; self.prev_kph = None; self.prev_t = None

            if self.tl_lock_active or (in_inner and stop_like):
                tgt_kph = 0.0

            if at_terminal:
                if dist_to_last <= self.stop_radius_m:
                    tgt_kph = 0.0
                elif dist_to_last <= self.stop_window_m:
                    tgt_kph = self._ramp_tgt_kph(
                        d=dist_to_last,
                        v_nominal_kph=v_nominal,
                        start=self.stop_window_m,
                        stop=self.stop_radius_m
                    )
                else:
                    tgt_kph = 0.0

            # 조향
            if j is not None:
                if j < N - 1:
                    path_yaw = atan2(ys[j+1]-ys[j], xs[j+1]-xs[j])
                else:
                    path_yaw = atan2(ys[j]-ys[j-1], xs[j]-xs[j-1])
                theta_e = wrap_angle(path_yaw - yaw)
                e_y = plocal[1] if plocal is not None else 0.0
                v_ms = max(self.kph / 3.6, 0.0)
                steer = theta_e + atan2(self.k * e_y, v_ms + self.v_eps)
                if self.invert_steer:
                    steer = -steer
                steer = float(np.clip(steer, -self.max_rad, self.max_rad))
            else:
                steer = 0.0

            # (신규) Follow 모드: 동적 선행차일 때 목표속도 캡
            if self.follow_enable and self.lead_is_dynamic and (self.lead_v_obj is not None):
                v_lead_kph = max(0.0, self.lead_v_obj * 3.6)
                cap1 = max(self.follow_v_min_kmh, v_lead_kph - self.follow_delta_kmh)
                cap = cap1
                if self.lead_gap_m is not None:
                    # 시간간격 기반 캡: v = (gap - min_gap)/T_gap (km/h 환산)
                    cap2 = 3.6 * max(0.0, (self.lead_gap_m - self.follow_min_gap_m) / max(self.follow_T_gap, 1e-3))
                    cap = min(cap, max(self.follow_v_min_kmh, cap2))
                tgt_kph = min(tgt_kph, cap)

            # 속도 제어
            now = rospy.Time.now()
            accel, brake, u_cmd, e_kph, d_meas, u_p, u_i, u_d = self.speed_pid_step(tgt_kph, self.kph, now)

            # 하드 브레이크/홀드
            if (stop_like and in_inner) or self.tl_lock_active:
                accel = 0.0
                brake = max(brake, self.tl_hard_brake_cmd)
            if (self.tl_lock_active or at_terminal) and self.kph <= self.brake_hold_kph:
                accel = 0.0
                brake = max(brake, self.brake_hold_cmd, brake)

            # 퍼블리시
            self.cmd.steering = steer
            self.cmd.accel = accel
            self.cmd.brake = brake
            self.pub.publish(self.cmd)

            # 디버그
            print("\033c", end="")
            print("── Stanley + PI-D (Dynamic Global Waypoints) | LOCK:", self.tl_lock_active)
            print(f" idx {j if j is not None else -1}/{N-1}  yaw {degrees(yaw):6.2f}°")
            print(f" kph cur/tgt {self.kph:5.2f}/{tgt_kph:5.2f}  accel {accel:.2f}  brake {brake:.2f}")
            if next_stop_idx is not None:
                print(f" stop_idx {next_stop_idx} | d_e {dist_euclid_to_stop:5.2f} m "
                      f"| OUTER({self.tl_active_radius_m}m)={in_outer}  INNER({self.tl_hard_zone_m}m)={in_inner}")
            else:
                print(" stop_idx None")
            print(f" DWELL engaged={self.dwell_engaged} done={self.dwell_done} | {dwell_info} | trig_r {self.dwell_trigger_radius_m:.2f} m")
            dd = math.hypot(x - self.heading_hold_xy[0], y - self.heading_hold_xy[1])
            print(f" heading-hold active: {self.heading_hold_active} | dist {dd:.2f} m "
                  f"(enter {self.heading_hold_radius_m:.2f} / exit {self.heading_hold_exit_radius_m:.2f})")
            print(f" terminal {at_terminal} | traffic {self.traffic_state!r} | tl_disable_in_hh {self.tl_disable_in_hh} | stop_ignore_in_hh {self.hh_ignore_stop_in_hh}")

            rate.sleep()

    # 선형 램프다운
    def _ramp_tgt_kph(self, d: float, v_nominal_kph: float, start: float, stop: float) -> float:
        if d <= stop:
            return 0.0
        if d >= start:
            return max(0.0, v_nominal_kph)
        t = (d - stop) / max(1e-6, (start - stop))
        return max(0.0, v_nominal_kph * t)


if __name__ == "__main__":
    try:
        StanleyController().run()
    except rospy.ROSInterruptException:
        pass
