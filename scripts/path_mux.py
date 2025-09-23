#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np
from std_msgs.msg import Bool
from std_msgs.msg import Bool as BoolMsg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import waypoint, waypointarray, status as status_msg


def _as_xy_list(param_val):
    """[[x,y], [x,y], ...] 또는 [x,y]를 [(x,y), ...]로 정규화."""
    try:
        if isinstance(param_val, (list, tuple)) and len(param_val) == 2 and \
           all(isinstance(v, (int, float)) for v in param_val):
            return [(float(param_val[0]), float(param_val[1]))]
        out = []
        for p in param_val:
            if isinstance(p, (list, tuple)) and len(p) == 2 and \
               all(isinstance(v, (int, float)) for v in p):
                out.append((float(p[0]), float(p[1])))
        return out
    except Exception:
        return []


def _min_dist_to_list(x, y, xy_list):
    """(x,y)에서 리스트 중 가장 가까운 점까지의 거리. 리스트 비면 inf."""
    if not xy_list:
        return float("inf")
    best = float("inf")
    for (px, py) in xy_list:
        dd = math.hypot(x - px, y - py)
        if dd < best:
            best = dd
    return best


class PathMux:
    def __init__(self):
        rospy.init_node("path_mux", anonymous=True)

        # ===== 토픽/파라미터 =====
        self.topic_global_src = rospy.get_param("~global_src_topic", "/global_waypoints_src")
        self.topic_mux_out    = rospy.get_param("~mux_out_topic",    "/global_waypoints")
        self.lattice_kph      = rospy.get_param("~lattice_kph", 25.0)  # 라티스 구간 속도(kph)

        # ===== Heading-Hold (단일 좌표 제거, 3좌표 리스트만 사용) =====
        self.heading_hold_enabled = bool(rospy.get_param("~heading_hold_enabled", True))

        # 기본값: 요청한 3개 좌표로 대체
        hh_list_default = [
            [618.2910, -907.3485],
            [618.2962, -903.3407],
            [618.2165, -898.8530],
        ]
        hh_xy_list_raw = rospy.get_param("~heading_hold_xy_list", hh_list_default)
        self.heading_hold_xy_list = _as_xy_list(hh_xy_list_raw)

        self.heading_hold_radius_m = float(rospy.get_param("~heading_hold_radius_m", 1.0))
        self.heading_hold_exit_radius_m = float(rospy.get_param("~heading_hold_exit_radius_m", 10.0))
        self.heading_hold_disable_if_distance_huge = bool(
            rospy.get_param("~heading_hold_disable_if_distance_huge", True)
        )
        self.heading_hold_huge_distance_m = float(
            rospy.get_param("~heading_hold_huge_distance_m", 100000.0)  # 100 km
        )

        # (신규) 동적 선행차 감지 시 글로벌 우선 스위치
        self.prefer_global_when_dynamic = bool(rospy.get_param("~prefer_global_when_dynamic", True))
        self.dynamic_hold_secs = float(rospy.get_param("~dynamic_hold_secs", 2.0))
        self._dynamic_deadline = rospy.Time(0)
        self._lead_is_dynamic = False
        rospy.Subscriber("/lead_is_dynamic", BoolMsg, self._lead_dyn_cb, queue_size=1)

        # 과대거리 auto-exit 연속 카운트 (기본 5번 연속일 때만 해제)
        self.hh_auto_exit_consecutive = int(rospy.get_param("~hh_auto_exit_consecutive", 5))

        # “구간 지나면 회피모드 금지” 래치
        self.block_lattice_after_hh = bool(rospy.get_param("~block_lattice_after_hh", True))

        self.pub_heading_hold_active = bool(rospy.get_param("~pub_heading_hold_active", True))

        # ------ 상태 글리치 가드 파라미터 ------
        self.status_max_abs_xy_m = float(rospy.get_param("~status_max_abs_xy_m", 100000.0))  # 100 km
        self.status_glitch_speed_mps = float(rospy.get_param("~status_glitch_speed_mps", 80.0))  # 288 km/h
        self._glitch_warn_period = float(rospy.get_param("~status_glitch_warn_period", 1.0))

        # ------ (기존) 위치 기반 회피금지 구간 파라미터 ------
        # 시작점 반경에 들어오면 회피금지 모드 ON, 종료점 반경에 들어오면 OFF
        self.nolattice_zone_enabled = bool(rospy.get_param("~nolattice_zone_enabled", False))
        nl_start_xy = rospy.get_param("~nolattice_start_xy", [780.5000, -817.6741])
        nl_end_xy   = rospy.get_param("~nolattice_end_xy",   [761.8843, -836.7208])
        self.nolattice_start_xy = (float(nl_start_xy[0]), float(nl_start_xy[1]))
        self.nolattice_end_xy   = (float(nl_end_xy[0]),   float(nl_end_xy[1]))
        self.nolattice_start_radius_m = float(rospy.get_param("~nolattice_start_radius_m", 2.5))
        self.nolattice_end_radius_m   = float(rospy.get_param("~nolattice_end_radius_m",   1.0))
        self.pub_nolattice_active     = bool(rospy.get_param("~pub_nolattice_active", True))

        # 내부 상태
        self.global_wp_src = None
        self.lattice_path = None
        self.use_lattice  = False

        self.heading_hold_active = False
        self._hh_ever_entered = False              # HH를 한 번이라도 들어갔는가
        self._lattice_blocked_after_hh = False     # 구간 통과 후 라티스 영구 금지 래치

        # (신규) 위치 기반 회피금지 활성 상태
        self._nolattice_zone_active = False

        self.have_status = False
        self.cur_x = None
        self.cur_y = None

        # 상태 글리치 가드용 이전 정상 샘플
        self._prev_good_x = None
        self._prev_good_y = None
        self._prev_t = None
        self._last_glitch_warn_time = rospy.Time(0)

        # 과대거리 auto-exit 연속 카운터 (HH용)
        self._huge_dist_count = 0

        # 구독/퍼블리시
        rospy.Subscriber(self.topic_global_src, waypointarray, self.cb_global_src, queue_size=1)
        rospy.Subscriber("/lattice_path", Path, self.cb_lattice, queue_size=1)
        rospy.Subscriber("/speed_control", Bool, self.cb_flag, queue_size=1)
        rospy.Subscriber("/status", status_msg, self.cb_status, queue_size=10)

        self.pub = rospy.Publisher(self.topic_mux_out, waypointarray, queue_size=1)
        self.pub_hh = rospy.Publisher("/heading_hold_active", Bool, queue_size=1) if self.pub_heading_hold_active else None
        self.pub_nlz = rospy.Publisher("/no_lattice_zone_active", Bool, queue_size=1) if self.pub_nolattice_active else None

        # HH 좌표 유효성 체크
        if self.heading_hold_enabled and not self.heading_hold_xy_list:
            rospy.logwarn("[PathMux][HH] no valid heading_hold_xy_list; disabling HH")
            self.heading_hold_enabled = False

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # ===== 상태 업데이트 =====
            self._update_heading_hold_state()
            self._update_no_lattice_zone_state()

            # 헤딩 고정 중이거나, HH 구간 통과 래치가 서있거나,
            # (신규) 위치 기반 회피금지 구간이 활성일 때는 라티스 무시
            # 이미 회피(use_lattice=True)면 글로벌 강제 차단을 적용하지 않음
            dyn_block = (self.prefer_global_when_dynamic and (rospy.Time.now() < self._dynamic_deadline) 
                        and (not self.use_lattice))
            effective_use_lattice = (
                self.use_lattice
                and (not self.heading_hold_active)
                and (not self._lattice_blocked_after_hh)
                and (not self._nolattice_zone_active)
                and (not dyn_block)
            )

            # 상태 퍼블리시(옵션)
            if self.pub_hh:
                self.pub_hh.publish(Bool(data=self.heading_hold_active))
            if self.pub_nlz:
                self.pub_nlz.publish(Bool(data=self._nolattice_zone_active))

            # ===== 경로 MUX =====
            if effective_use_lattice and self.lattice_path and self.global_wp_src:
                rospy.loginfo_throttle(1.0, "[PathMux] Using Lattice+GlobalTail (HH inactive & NLZ inactive)")
                muxed = self.splice_lattice_with_global(self.lattice_path, self.global_wp_src, self.lattice_kph)
                self.pub.publish(muxed)
            elif self.global_wp_src:
                # 라티스를 쓰지 않는 이유를 친절 로그
                if self.heading_hold_active:
                    rospy.loginfo_throttle(1.0, "[PathMux] Heading-Hold ACTIVE → Using GlobalPath only")
                elif self._lattice_blocked_after_hh:
                    rospy.loginfo_throttle(1.0, "[PathMux] LATTICE BLOCKED after HH → Using GlobalPath only")
                elif self._nolattice_zone_active:
                    rospy.loginfo_throttle(1.0, "[PathMux] No-Lattice ZONE ACTIVE → Using GlobalPath only")
                else:
                    rospy.loginfo_throttle(2.0, "[PathMux] Using GlobalPath")
                self.pub.publish(self.global_wp_src)

            rate.sleep()

    #-----------new-----------------------
    def _lead_dyn_cb(self, msg: BoolMsg):
        self._lead_is_dynamic = bool(msg.data)
        if self._lead_is_dynamic:
            self._dynamic_deadline = rospy.Time.now() + rospy.Duration.from_sec(self.dynamic_hold_secs)

    # ---------------------- 콜백들 ----------------------
    def cb_global_src(self, msg: waypointarray):
        self.global_wp_src = msg

    def cb_lattice(self, msg: Path):
        self.lattice_path = msg

    def cb_flag(self, msg: Bool):
        self.use_lattice = bool(msg.data)

    def cb_status(self, msg: status_msg):
        try:
            x = msg.now_position.pose.position.x
            y = msg.now_position.pose.position.y
        except Exception:
            return

        # 타임스탬프 (없으면 now)
        try:
            t = msg.now_position.header.stamp
            if t is None or t.to_sec() == 0.0:
                t = rospy.Time.now()
        except Exception:
            t = rospy.Time.now()

        # --- 글리치 가드: 값 범위 체크 ---
        if abs(x) > self.status_max_abs_xy_m or abs(y) > self.status_max_abs_xy_m:
            self._warn_glitch_once("abs(x/y) too large", x, y)
            return

        # --- 글리치 가드: 비현실 점프(속도) 체크---
        if self._prev_good_x is not None and self._prev_t is not None:
            dt = max((t - self._prev_t).to_sec(), 1e-3)
            vx = (x - self._prev_good_x) / dt
            vy = (y - self._prev_good_y) / dt
            spd = math.hypot(vx, vy)
            if spd > self.status_glitch_speed_mps:
                self._warn_glitch_once(f"jump speed {spd:.2f} m/s", x, y)
                return

        # 정상값 반영
        self.cur_x = x
        self.cur_y = y
        self.have_status = True
        self._prev_good_x = x
        self._prev_good_y = y
        self._prev_t = t

    def _warn_glitch_once(self, reason: str, x: float, y: float):
        now = rospy.Time.now()
        if (now - self._last_glitch_warn_time).to_sec() >= self._glitch_warn_period:
            rospy.logwarn("[PathMux][STATUS] glitch ignored (%s) x=%.3f y=%.3f", reason, x, y)
            self._last_glitch_warn_time = now

    # ---------------------- 로직 ----------------------
    def _update_heading_hold_state(self):
        """헤딩 고정 진입/해제 + 글리치 가드 + '구간 통과 후 라티스 금지' 래치.
        (세 점 중 가장 가까운 점 기준)"""
        if not self.heading_hold_enabled or not self.have_status or self.cur_x is None or self.cur_y is None:
            if self.heading_hold_active:
                rospy.logwarn_throttle(1.0, "[PathMux][HH] disabled/no status → deactivated")
            self.heading_hold_active = False
            self._huge_dist_count = 0
            return

        # 세 좌표 중 가장 가까운 점까지의 거리
        dd = _min_dist_to_list(self.cur_x, self.cur_y, self.heading_hold_xy_list)

        # 해제 가드(거리 이상치/이탈)
        if self.heading_hold_active:
            if dd > self.heading_hold_exit_radius_m:
                # 과대거리 → 연속 카운트로만 해제 (래치 X)
                if self.heading_hold_disable_if_distance_huge and dd > self.heading_hold_huge_distance_m:
                    self._huge_dist_count += 1
                    rospy.logwarn_throttle(0.5, "[PathMux][HH] huge dist sample (dd=%.2fm) cnt=%d/%d",
                                           dd, self._huge_dist_count, self.hh_auto_exit_consecutive)
                    if self._huge_dist_count >= self.hh_auto_exit_consecutive:
                        rospy.logwarn("[PathMux][HH] auto-exit (huge dist persisted)")
                        self.heading_hold_active = False
                        self._huge_dist_count = 0
                        # 글리치 기반 해제는 '구간 통과'로 보지 않음 → lattice 금지 래치 X
                        return
                else:
                    # 정상적인 exit 반경 이탈 → 구간 통과로 간주
                    self._huge_dist_count = 0
                    self.heading_hold_active = False
                    rospy.logwarn_throttle(1.0, "[PathMux][HH] exit by radius (dist=%.2fm > %.2fm)",
                                           dd, self.heading_hold_exit_radius_m)
                    if self.block_lattice_after_hh and self._hh_ever_entered and not self._lattice_blocked_after_hh:
                        self._lattice_blocked_after_hh = True
                        rospy.logwarn("[PathMux][HH] Section passed → LATTICE BLOCKED for remainder of run")
                    return
            else:
                self._huge_dist_count = 0

        # 진입: 세 점 중 하나라도 진입 반경 이내면 활성화
        if (not self.heading_hold_active) and dd <= self.heading_hold_radius_m:
            self.heading_hold_active = True
            self._huge_dist_count = 0
            self._hh_ever_entered = True
            rospy.loginfo("[PathMux][HH] ENTER (min dist=%.2fm <= %.2fm)", dd, self.heading_hold_radius_m)

    def _update_no_lattice_zone_state(self):
        """(기존) 시작점 반경 진입 시 회피금지 ON, 종료점 반경 진입 시 OFF."""
        if not self.nolattice_zone_enabled or not self.have_status or self.cur_x is None or self.cur_y is None:
            # 상태가 없거나 비활성화면 NLZ도 비활성
            if self._nolattice_zone_active:
                rospy.logwarn_throttle(1.0, "[PathMux][NLZ] disabled/no status → deactivated")
            self._nolattice_zone_active = False
            return

        # 현재 위치와 시작/종료점 거리
        dxs = self.cur_x - self.nolattice_start_xy[0]
        dys = self.cur_y - self.nolattice_start_xy[1]
        dds = math.hypot(dxs, dys)

        dxe = self.cur_x - self.nolattice_end_xy[0]
        dye = self.cur_y - self.nolattice_end_xy[1]
        dde = math.hypot(dxe, dye)

        if not self._nolattice_zone_active:
            # 시작 반경 안으로 들어오면 활성화
            if dds <= self.nolattice_start_radius_m:
                self._nolattice_zone_active = True
                rospy.loginfo("[PathMux][NLZ] ENTER (start dist=%.2fm <= %.2fm)",
                              dds, self.nolattice_start_radius_m)
        else:
            # 활성 상태에서 종료 반경 안에 들어오면 해제
            if dde <= self.nolattice_end_radius_m:
                self._nolattice_zone_active = False
                rospy.loginfo("[PathMux][NLZ] EXIT (end dist=%.2fm <= %.2fm)",
                              dde, self.nolattice_end_radius_m)

    # --------- 라티스 + 전역 꼬리 붙이기 ----------
    def splice_lattice_with_global(self, lattice_path: Path, global_wp: waypointarray, lattice_kph: float) -> waypointarray:
        wa = waypointarray()

        # 1) 라티스 Path(map) -> waypointarray (속도는 일정값)
        if lattice_path and lattice_path.poses:
            for ps in lattice_path.poses:
                w = waypoint()
                w.x = ps.pose.position.x
                w.y = ps.pose.position.y
                w.speed = lattice_kph
                wa.waypoints.append(w)

        # 2) 전역 웨이포인트 꼬리 찾기: 라티스 마지막 점과 가장 가까운 전역 인덱스부터 끝까지 붙임
        if global_wp and global_wp.waypoints:
            if wa.waypoints:
                lx = wa.waypoints[-1].x
                ly = wa.waypoints[-1].y
            else:
                return global_wp

            xs = np.array([w.x for w in global_wp.waypoints], dtype=float)
            ys = np.array([w.y for w in global_wp.waypoints], dtype=float)
            d2 = (xs - lx) * (xs - lx) + (ys - ly) * (ys - ly)
            idx_tail = int(np.argmin(d2))

            # 3) 꼬리 붙이기 (전역 속도 그대로)
            for i in range(idx_tail, len(global_wp.waypoints)):
                wa.waypoints.append(global_wp.waypoints[i])

        return wa


if __name__ == "__main__":
    try:
        PathMux()
    except rospy.ROSInterruptException:
        pass
