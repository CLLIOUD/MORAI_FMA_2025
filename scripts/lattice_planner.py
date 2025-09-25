#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, math, numpy as np, random
from math import cos, sin, atan2
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Bool, Float32
import tf2_ros, tf2_geometry_msgs

# /status 메시지 (파일명이 소문자 status.msg 이므로 클래스명도 status일 수 있어 alias 권장)
from custom_msgs.msg import status as Status


# ───────────────────────────────────────────────────────────────
# LeadTracker : 가장 가까운 선행체 추적(경량 최근접-유지)
# ───────────────────────────────────────────────────────────────
class LeadTracker:
    def __init__(self, planner):
        self.p = planner
        self.last_xy = None          # (x,y) 월드
        self.last_s  = None          # 선행체 경로 누적거리
        self.last_t  = None          # rospy.Time as float seconds
        self.missed  = 0

    def update(self, clusters_map: PoseArray, stamp: rospy.Time):
        p = self.p
        path = p.local_path
        if clusters_map is None or path is None or p.vx is None:
            self.missed += 1
            return False, None, None

        xs = np.array([ps.pose.position.x for ps in path.poses], dtype=float)
        ys = np.array([ps.pose.position.y for ps in path.poses], dtype=float)
        ds = np.hypot(np.diff(xs), np.diff(ys)); s = np.concatenate([[0.0], np.cumsum(ds)])

        j_ego = int(np.argmin((xs - p.vx)**2 + (ys - p.vy)**2))
        j_ego = min(max(0, j_ego), len(xs)-2)
        s_ego = float(s[j_ego])

        yaw = [math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)]; yaw.append(yaw[-1])

        margin   = float(p.follow_lane_margin)
        lat_lim  = max(0.2, p.road_half_w - margin)
        ahead    = float(min(p.corridor_ahead_m, 60.0))
        kmax     = min(len(xs)-1, j_ego + 300)

        cand = []
        for o in clusters_map.poses:
            ox, oy = o.position.x, o.position.y
            k = min(range(j_ego, kmax), key=lambda i:(xs[i]-ox)**2+(ys[i]-oy)**2)
            th = yaw[k]
            dx, dy = (ox - xs[k]), (oy - ys[k])
            longi  = math.cos(th)*dx + math.sin(th)*dy
            lat    = -math.sin(th)*dx + math.cos(th)*dy
            if (0.0 < longi <= ahead) and (abs(lat) <= lat_lim):
                cand.append((ox, oy, float(s[k] + longi)))

        if not cand:
            self.missed += 1
            return False, None, None

        if self.last_xy is not None:
            ox0, oy0 = self.last_xy
            c = min(cand, key=lambda c_: (c_[0]-ox0)**2 + (c_[1]-oy0)**2)
            if math.hypot(c[0]-ox0, c[1]-oy0) > p.follow_nn_gate:
                c = min(cand, key=lambda c_: c_[2])
        else:
            c = min(cand, key=lambda c_: c_[2])

        s_lead = c[2]
        t_now  = float(stamp.to_sec()) if stamp and stamp != rospy.Time() else rospy.get_time()
        v_rel  = None
        if (self.last_s is not None) and (self.last_t is not None):
            ds_dt = (s_lead - self.last_s) / max(1e-3, (t_now - self.last_t))
            v_rel = ds_dt - float(p.v_mps)

        x_lead = float(s_lead - s_ego)
        self.last_xy = (c[0], c[1]); self.last_s = s_lead; self.last_t = t_now
        self.missed  = 0
        return True, x_lead, v_rel


class latticePlanner:
    def __init__(self):
        rospy.init_node('lattice_planner', anonymous=True)

        # Subs
        rospy.Subscriber("/global_path", Path, self.path_cb)
        rospy.Subscriber("/right_path",  Path, self.right_cb)
        rospy.Subscriber("/clusters",    PoseArray, self.clusters_cb, queue_size=1)
        rospy.Subscriber("/cluster_distances", Float32MultiArray, self.dist_cb, queue_size=1)
        rospy.Subscriber("/cluster_features", Float32MultiArray, self.features_cb, queue_size=1)
        rospy.Subscriber("/status", Status, self.status_cb_status, queue_size=10)

        # Pubs
        self.pub_path  = rospy.Publisher("/lattice_path", Path, queue_size=1)
        self.pub_speed = rospy.Publisher("/speed_control", Bool, queue_size=1)

        # States
        self.is_path   = False
        self.is_status = False
        self.local_path = None; self.right_path = None
        self.vx = self.vy = None
        self.v_mps = 0.0
        self.v_kph = 0.0
        self.clusters_map = None; self.min_dist_m = None
        self.min_edge_m = None; self.min_center_m = None
        self.occ_pct = None; self.ttc_s = None
        self._clusters_stamp = rospy.Time(0)
        self.cluster_features = []
        self._features_stamp = rospy.Time(0)

        # TF
        self.tfbuf = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # Road / corridor
        g = rospy.get_param
        self.road_half_w       = g("~road_half_width", 2.6)
        self.corridor_ahead_m  = g("~corridor_ahead_m", 25.0)
        self.path_inflation_m  = g("~path_inflation_m", 0.4)

        # Avoid hysteresis
        self.avoid_on_frames   = g("~avoid_on_frames", 5)
        self.avoid_off_frames  = g("~avoid_off_frames", 10)
        self._vote = 0; self.avoid_active = False

        # Free-space bands
        self.band_dx      = g("~band_dx", 0.5)
        self.car_width    = g("~car_width", 1.90)
        self.lat_margin   = g("~lat_margin", 0.45)
        self.required_clear = 0.5*self.car_width + self.lat_margin
        self.side_guard   = g("~side_guard", 0.8)

        # Edge penetration / ramp
        self.edge_penetration_min = g("~edge_penetration_min", 0.20)
        self.edge_ramp_band       = g("~edge_ramp_band", 0.30)
        self.edge_ramp_gamma      = g("~edge_ramp_gamma", 1.5)

        # Trigger
        self.trigger_min_clusters = g("~trigger_min_clusters", 1)

        # Candidates / scoring
        self.clearance_stride  = g("~clearance_stride", 2)
        self.clearance_soft_w  = g("~clearance_soft_weight", 10.0)

        # Commit logic
        self.avoid_commit_m = g("~avoid_commit_m", 19.0)
        self._hold_until_s  = None
        self._sticky_side   = 0  # -1 L, +1 R

        # Candidate generation (Bezier + guards)
        self.edge_margin   = g("~edge_margin", 0.45)
        self.avoid_x_scale = g("~avoid_x_end_scale", 0.9)
        self.bez_alpha     = g("~bezier_alpha", 0.18)
        self.bez_beta      = g("~bezier_beta", 0.18)
        self.bez_push      = g("~bezier_push", 0.8)
        self.bez_relax     = g("~bezier_relax", 0.35)
        self.min_x_end_m   = g("~min_x_end_m", 20.0)
        self.min_points    = g("~min_points", 40)

        # REJOIN
        self.rejoin_clear_m      = g("~rejoin_clear_m", 3.0)
        self.rejoin_clear_frames = g("~rejoin_clear_frames", 1)
        self.rejoin_x_m          = g("~rejoin_x_m", 24.0)
        self.rejoin_lat_eps      = g("~rejoin_lat_eps", 0.35)
        self._rejoin_cnt = 0
        self._rejoin_mode = False
        self.rejoin_hard   = g("~rejoin_hard", True)
        self.rejoin_slice_m= g("~rejoin_slice_m", 26.0)

        # Safety guards
        self.dynamic_lat_k     = g("~dynamic_lat_k", 0.05)
        self.band_dilate_m     = g("~band_dilate_m", 0.40)
        self.min_free_prefix_m = g("~min_free_prefix_m", 22.0)
        self.clearance_window_m= g("~clearance_window_m", 24.0)
        self.clearance_percentile_q = g("~clearance_percentile_q", 60.0)

        # Edge-only 억제
        self.occ_inner_th  = g("~occ_inner_th", 6.0)
        self.occ_trigger_pct = g("~occ_trigger_pct", 12.0)
        self.edge_soft_band= g("~edge_soft_band", 0.30)

        # 안티-채터
        self.score_margin       = g("~score_margin", 0.12)
        self.stick_frames       = int(g("~stick_frames", 4))
        self.min_switch_hold_s  = g("~min_switch_hold_s", 1.0)
        self.main_hz            = 15
        self.occ_inner_pct = None
        self.occ_edge_pct  = None
        self._last_idx     = None
        our_last_score     = None
        self._stick_cnt    = 0
        self._hold_until   = 0.0
        self.idx_step_cap  = g("~idx_step_cap", 2)
        self.default_avoid_side = g("~default_avoid_side", -1)

        # Follow-Latch
        self.follow_enable       = bool(g("~follow_enable", True))
        self.follow_x_min        = g("~follow_x_min", 6.0)
        self.follow_x_max        = g("~follow_x_max", 35.0)
        self.follow_vrel_abs_max = g("~follow_vrel_abs_max", 1.5)
        self.follow_on_frames    = int(g("~follow_on_frames", 6))
        self.follow_off_frames   = int(g("~follow_off_frames", 10))
        self.follow_min_hold_s   = g("~follow_min_hold_s", 2.0)
        self.follow_lane_margin  = g("~follow_lane_margin", 0.35)
        self.follow_nn_gate      = g("~follow_nn_gate", 3.0)
        self.stick_margin        = g("~stick_margin", 0.15)
        self.switch_on_frames    = int(g("~switch_on_frames", 5))
        self.lead = LeadTracker(self)
        self.follow_vote   = 0
        self.follow_active = False
        self._follow_hold_until = 0.0

        # 긴 장애물 + 속도연동 진입각
        self.long_block_threshold_m = g("~long_block_threshold_m", 8.0)
        self.long_alpha = g("~long_alpha", 0.12)
        self.long_push  = g("~long_push", 1.15)
        self.long_relax = g("~long_relax", 0.70)
        self.long_ofs_s1= g("~long_offset_scale1", 1.60)
        self.long_ofs_s2= g("~long_offset_scale2", 2.00)
        self.long_xe_scale = g("~long_xe_scale", 1.40)
        self.long_entry_boost_k   = g("~long_entry_boost_k", 0.01)
        self.long_entry_boost_cap = g("~long_entry_boost_cap", 0.25)
        # (이전 추가) 버스 첫 후보 오프셋 확대 계수
        self.long_first_ofs_gain  = g("~long_first_ofs_gain", 1.30)

        # 작은 박스(스페클) 대응
        self.speckle_gap_close_m   = g("~speckle_gap_close_m", 1.0)
        self.speckle_window_m      = g("~speckle_window_m", 18.0)
        self.speckle_density_pct   = g("~speckle_density_pct", 30.0)
        self.speckle_commit_m      = g("~speckle_commit_m", 12.0)
        self.speckle_ofs_scale     = g("~speckle_ofs_scale", 1.10)
        self.speckle_xe_scale      = g("~speckle_xe_scale", 0.98)

        # ────────────── 회피 종료 직전 직선 주행(post-hold) ──────────────
        self.post_hold_min_s  = g("~post_hold_min_s", 1.0)   # 1.5 ~ 2.5 s
        self.post_hold_max_s  = g("~post_hold_max_s", 1.8)
        self.post_hold_min_m  = g("~post_hold_min_m", 2.5)   # 너무 느릴 때 최소 길이
        self.post_hold_step_m = g("~post_hold_step_m", 0.5)  # 직선 경로 샘플 간격
        self.post_hold_extra_s = g("~post_hold_extra_s", 0.2)

        # 회피 조기 발동 스케일
        self.trigger_early_scale = g("~trigger_early_scale", 1.08)

        # 복귀 시작 후 장애물 무시 시간(초)
        self.rejoin_ignore_secs = g("~rejoin_ignore_secs", 3.0)
        self._rejoin_ignore_until_t = 0.0

        # >>> NEW: 회피 폭 강화 (후보 오프셋/안전여유)
        self.avoid_width_gain   = g("~avoid_width_gain", 1.08)   # 후보 오프셋 8% 확대
        self.avoid_extra_clear_m= g("~avoid_extra_clear_m", 0.15)
        self.avoid_low_speed_block_mps = float(g("~avoid_low_speed_block_mps", 1.5))

        # 장애물 평행 회피(미니멀 래터럴) 설정
        self.hug_enable            = bool(g("~hug_enable", True))
        self.hug_entry_buffer      = g("~hug_entry_buffer", 1.2)
        self.hug_exit_buffer       = g("~hug_exit_buffer", 1.0)
        self.hug_secondary_margin  = g("~hug_secondary_margin", 0.35)
        self.hug_entry_shape       = g("~hug_entry_shape", 0.65)
        self.hug_exit_relax        = g("~hug_exit_relax", 0.30)
        self.hug_parallel_scale    = g("~hug_parallel_scale", 0.45)
        self.hug_tangent_clip      = g("~hug_tangent_clip", 0.60)

        self._post_hold_active   = False
        self._post_hold_until_t  = 0.0
        self._post_hold_path     = None
        self._post_hold_done     = False  # 이번 회피 싸이클에서 post-hold를 이미 수행했는가?

        #-------------------------------------------------------------------------------------
        # (신규) 선행차 추정값 퍼블
        self.pub_lead_gap = rospy.Publisher("/lead_gap_m", Float32, queue_size=1)
        self.pub_lead_vobj = rospy.Publisher("/lead_v_obj", Float32, queue_size=1)
        self.pub_lead_vrel = rospy.Publisher("/lead_v_rel", Float32, queue_size=1)
        self.pub_lead_dyn  = rospy.Publisher("/lead_is_dynamic", Bool, queue_size=1)

        # (신규) Cut-in Guard 파라미터/상태
        self.cutin_enable      = bool(rospy.get_param("~cutin_guard_right/enable", True))
        self.cutin_lat_range   = rospy.get_param("~cutin_guard_right/lat_range", [0.5, 3.0])    # [m] (우측=음수 가정 시 내부에서 abs로 사용)
        self.cutin_ahead_range = rospy.get_param("~cutin_guard_right/ahead_range", [4.0, 25.0]) # [m] 전방 범위
        self.cutin_vrel_thr    = float(rospy.get_param("~cutin_guard_right/dyn_thresh_vrel", 0.3)) # [m/s]
        self.cutin_hold_secs   = float(rospy.get_param("~cutin_guard_right/hold_secs", 2.0))
        self.cutin_lat_allow   = float(rospy.get_param("~cutin_guard_right/max_lat_dev_allow", 0.7))
        self.cutin_cost_boost  = float(rospy.get_param("~cutin_guard_right/cost_boost", 1e6))
        self._cutin_guard_deadline = rospy.Time(0)


        rospy.loginfo("lattice_planner: start")
        self.spin()

    # --------- callbacks ---------
    def path_cb(self, msg): self.is_path=True; self.local_path=msg
    def right_cb(self, msg): self.right_path=msg

    def status_cb_status(self, msg: Status):
        try:
            self.vx = float(msg.now_position.pose.position.x)
            self.vy = float(msg.now_position.pose.position.y)
            self.v_kph = float(msg.now_speed)
            self.v_mps = self.v_kph / 3.6
            self.is_status = True
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[lattice] /status parse fail: {e}")

    def dist_cb(self, msg):
        try:
            data = list(msg.data)
            self.min_edge_m   = float(data[0]) if len(data) >= 1 else None
            self.min_center_m = float(data[1]) if len(data) >= 2 else None
            self.occ_pct      = float(data[2]) if len(data) >= 3 else None
            self.ttc_s        = float(data[3]) if len(data) >= 4 else None
            self.min_dist_m   = self.min_center_m
        except Exception:
            self.min_edge_m = self.min_center_m = self.min_dist_m = None
            self.occ_pct = self.ttc_s = None

    def features_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        feats = []
        step = 5
        for i in range(0, len(data), step):
            if i + 4 >= len(data):
                break
            cx, cy, yaw, half_len, half_width = data[i:i+5]
            feats.append({
                'x': float(cx),
                'y': float(cy),
                'yaw': float(yaw),
                'half_len': max(0.0, float(half_len)),
                'half_width': max(0.0, float(half_width))
            })
        self.cluster_features = feats
        self._features_stamp = rospy.Time.now()

    # --------- occupancy helpers ----------
    def _compute_inner_edge_occ(self):
        if self.clusters_map is None or not getattr(self.clusters_map, "poses", None):
            self.occ_inner_pct, self.occ_edge_pct = 0.0, 0.0
            return self.occ_inner_pct, self.occ_edge_pct
        inner   = max(0.0, float(self.road_half_w - self.lat_margin))
        edge_lim= inner + float(self.edge_soft_band)
        n_roi = n_inner = n_edge = 0
        for p in self.clusters_map.poses:
            x, y = float(p.position.x), float(p.position.y)
            if not (0.0 < x <= float(self.corridor_ahead_m)): continue
            ay = abs(y); n_roi += 1
            if ay <= inner: n_inner += 1
            elif ay <= edge_lim: n_edge += 1
        if n_roi == 0:
            self.occ_inner_pct, self.occ_edge_pct = 0.0, 0.0
        else:
            self.occ_inner_pct = 100.0 * n_inner / n_roi
            self.occ_edge_pct  = 100.0 * n_edge  / n_roi
        return self.occ_inner_pct, self.occ_edge_pct

    def _apply_stickiness(self, best_idx, best_score):
        now = rospy.get_time()

        if self._last_idx is None:
            self._last_idx, self._last_score = best_idx, best_score
            self._stick_cnt, self._hold_until = 0, now + self.min_switch_hold_s
            return self._last_idx, self._last_score
        if now < self._hold_until:
            return self._last_idx, self._last_score
        better_enough = (best_score <= (self._last_score - self.score_margin))
        if better_enough and best_idx != self._last_idx:
            self._stick_cnt += 1
        else:
            self._stick_cnt = 0
        if self._stick_cnt >= self.stick_frames:
            self._last_idx, self._last_score = best_idx, best_score
            self._stick_cnt, self._hold_until = 0, now + self.min_switch_hold_s
        return self._last_idx, self._last_score

    def clusters_cb(self, pa: PoseArray):
        try:
            T = self.tfbuf.lookup_transform('map', pa.header.frame_id, pa.header.stamp, rospy.Duration(0.2))
        except Exception:
            try:
                T = self.tfbuf.lookup_transform('map', pa.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[lattice] cluster tf fail: {e}")
                self.clusters_map=None; return
        pa_map = PoseArray(); pa_map.header.frame_id='map'; pa_map.header.stamp=pa.header.stamp
        pa_map.poses=[tf2_geometry_msgs.do_transform_pose(PoseStamped(header=pa.header,pose=p),T).pose for p in pa.poses]
        self.clusters_map=pa_map
        self._clusters_stamp = pa.header.stamp

    # --------- main loop ----------
    def spin(self):
        rate=rospy.Rate(self.main_hz)
        while not rospy.is_shutdown():
            if not (self.is_path and self.is_status):
                rate.sleep(); continue

            self._compute_inner_edge_occ()

            # 회피 발동 조기화: d_on에 스케일 적용
            d_on_base = 5.0 + 0.6*self.v_mps + (self.v_mps*self.v_mps)/(2.0*3.0)
            d_on = d_on_base * float(self.trigger_early_scale)

            n_on, min_longi = self._count_on_path(self.local_path, infl=self.path_inflation_m)
            is_close   = (self.min_dist_m is not None) and (self.min_dist_m < d_on)
            on_path    = (n_on >= self.trigger_min_clusters)
            near_prefix= (min_longi is not None) and (min_longi < self.min_free_prefix_m)
            inner_ok   = (self.occ_inner_pct is not None) and (self.occ_inner_pct >= self.occ_inner_th)
            want_static = is_close or ((inner_ok) and near_prefix) or (on_path and near_prefix)
            low_speed_block = (self.avoid_low_speed_block_mps > 1e-3) and (float(self.v_mps) < float(self.avoid_low_speed_block_mps))
            want = want_static and (not low_speed_block)
            if low_speed_block and want_static:
                rospy.loginfo_throttle(1.0, f"[lattice] avoid suppressed (v={self.v_mps:.2f} m/s < {self.avoid_low_speed_block_mps:.1f} m/s)")

            now = rospy.get_time()
            strong_static = bool(want)   # 정적 회피를 강하게 원할 조건이 이미 성립했는가?

            # ── (신규) 선행차 추정: gap/v_obj/v_rel/dynamic 판정
            found, x_lead, v_rel = self.lead.update(self.clusters_map, getattr(self, "_clusters_stamp", rospy.Time(0)))
            lead_gap = x_lead
            v_obj = (v_rel + float(self.v_mps)) if (v_rel is not None) else None
            is_dyn = bool((v_rel is not None) and (abs(v_rel) >= float(rospy.get_param("~dyn_thresh_vrel", 0.3))))

            # 퍼블리시 (NaN 처리 대신 값 없는 경우는 생략할 수도 있음)
            if lead_gap is not None: self.pub_lead_gap.publish(Float32(data=float(lead_gap)))
            if v_obj   is not None: self.pub_lead_vobj.publish(Float32(data=float(v_obj)))
            if v_rel   is not None: self.pub_lead_vrel.publish(Float32(data=float(v_rel)))
            self.pub_lead_dyn.publish(Bool(data=bool(is_dyn)))

            # (정책) 동적 선행차면 차로유지 우선 → 라티스 비활성(= 글로벌 우선)
            if is_dyn and (not strong_static):
                want = False   # 회피 투표 자체를 막음

            # ── (신규) 우측 Cut-in 감지 → Guard 래치
            if self.cutin_enable:
                if self._detect_right_cutin(self.local_path, self.clusters_map, v_rel_est=v_rel):
                    self._cutin_guard_deadline = rospy.Time.now() + rospy.Duration.from_sec(self.cutin_hold_secs)
            cutin_guard_active = rospy.Time.now() < self._cutin_guard_deadline
            self._cutin_guard_active = bool(cutin_guard_active)  # 아래 후보 선택 바이어스에서 사용


            # Follow-Latch
            if self.follow_enable:
                found, x_lead, v_rel = self.lead.update(self.clusters_map, getattr(self, "_clusters_stamp", rospy.Time(0)))
                lead_ok = (found and (x_lead is not None) and (self.follow_x_min <= x_lead <= self.follow_x_max)
                           and (v_rel is not None) and (abs(v_rel) <= self.follow_vrel_abs_max))
                occ_ok  = (self.occ_pct is None) or (float(self.occ_pct) >= float(self.occ_trigger_pct))
                near_ok = bool(near_prefix)
                follow_want = bool(lead_ok and occ_ok and near_ok and (not strong_static))
                if follow_want: self.follow_vote = min(self.follow_vote + 1, self.follow_on_frames)
                else:           self.follow_vote = max(self.follow_vote - 1, -self.follow_off_frames)

                if (not self.follow_active) and (self.follow_vote >= self.follow_on_frames):
                    self.follow_active = True
                    self._follow_hold_until = now + float(self.follow_min_hold_s)
                if self.follow_active:
                    miss = self.lead.missed
                    cond_dist = (x_lead is not None) and (x_lead > 45.0)
                    cond_close= (v_rel is not None) and (v_rel < -2.0)
                    cond_hold = (now > self._follow_hold_until)
                    cond_miss = (miss >= self.follow_off_frames)
                    if (cond_hold and (cond_dist or cond_close or cond_miss or (self.follow_vote <= -self.follow_off_frames))):
                        self.follow_active = False
                if self.follow_active:
                    want = False
                    rospy.loginfo_throttle(0.5, f"[follow] active={self.follow_active} x_lead={x_lead if found else 'None'} "
                                                f"v_rel={v_rel if v_rel is not None else 'None'} vote={self.follow_vote:+d} "
                                                f"occ={self.occ_pct} near_prefix={near_prefix}")

            # 복귀 중 N초 장애물 무시
            if (self._rejoin_mode or self._post_hold_active) and (now < self._rejoin_ignore_until_t):
                if want:
                    rospy.loginfo_throttle(0.5, "[rejoin-ignore] ignoring obstacles for %.1fs",
                                           self._rejoin_ignore_until_t - now)
                want = False

            # 회피 투표
            self._vote = min(self._vote+1, self.avoid_on_frames) if want else max(self._vote-1, -self.avoid_off_frames)
            s_now = self._cur_s(self.local_path, self.vx, self.vy)

            # 회피 시작 (post-hold 상태 초기화)
            if (not self.avoid_active) and (self._vote >= self.avoid_on_frames):
                self.avoid_active=True
                self._hold_until_s=None
                self._sticky_side=0
                self._post_hold_active  = False
                self._post_hold_path    = None
                self._post_hold_done    = False
                self._post_hold_until_t = 0.0
                self._rejoin_ignore_until_t = 0.0

            if self.avoid_active:
                prev_rejoin = self._rejoin_mode  # rising-edge 감지용

                # 커밋 거리 계산(스페클/체인 고려)
                if self._hold_until_s is None and s_now is not None:
                    commit = float(self.avoid_commit_m)
                    bands = self._build_bands(self.local_path)
                    if bands:
                        chain_len = self._estimate_run_with_gaps(bands, lat0=0.0, gap_tol_m=self.speckle_gap_close_m)
                        density   = self._speckle_density(bands, window_m=self.speckle_window_m)
                        if (density >= float(self.speckle_density_pct)) and (chain_len < float(self.long_block_threshold_m)):
                            commit = float(self.speckle_commit_m)
                    self._hold_until_s = s_now + commit

                clear_front = (n_on == 0) and (self.min_dist_m is None or self.min_dist_m > self.rejoin_clear_m)
                self._rejoin_cnt = self._rejoin_cnt + 1 if clear_front else 0

                # 기본 rejoin 판정
                base_rejoin_ready = ((self._hold_until_s is not None) and (s_now is not None) and (s_now >= self._hold_until_s)
                                     and (self._rejoin_cnt >= self.rejoin_clear_frames))

                # rejoin/post-hold 상태 전이
                if base_rejoin_ready and (not self._post_hold_done) and (not self._post_hold_active):
                    self._start_post_hold()
                    self._rejoin_mode = False
                elif self._post_hold_active:
                    self._rejoin_mode = False
                    if now >= self._post_hold_until_t:
                        self._post_hold_active = False
                        self._post_hold_done   = True
                        self._rejoin_mode      = True
                else:
                    self._rejoin_mode = base_rejoin_ready

                # rejoin 모드 진입 순간에 장애물 무시 타이머
                if (not prev_rejoin) and self._rejoin_mode:
                    self._rejoin_ignore_until_t = now + float(self.rejoin_ignore_secs)
                    rospy.loginfo("[rejoin-ignore] started for %.1fs", float(self.rejoin_ignore_secs))

                # 회피 종료 조건
                if (self._hold_until_s is not None) and (s_now is not None) and (s_now >= self._hold_until_s):
                    if self._vote <= -self.avoid_off_frames:
                        clear_front = (n_on == 0) and (self.min_dist_m is None or self.min_dist_m > self.rejoin_clear_m)
                        lat_err = self._lat_err_to_global()
                        if clear_front and (lat_err is not None) and (lat_err < self.rejoin_lat_eps):
                            self.avoid_active=False
                            self._sticky_side=0
                            self._rejoin_mode=False
                            self._last_idx=None
                            self._post_hold_active=False
                            self._post_hold_done=False
                            self._post_hold_path=None
                            self._rejoin_ignore_until_t = 0.0

            rospy.loginfo_throttle(0.5, f"[lattice] vote {self._vote:+d} | avoid {self.avoid_active} | "
                                        f"rejoin {self._rejoin_mode} ignore_t {max(0.0, self._rejoin_ignore_until_t - now):.1f}s | "
                                        f"min_longi {min_longi if min_longi is not None else 'None'} "
                                        f"| d_on {d_on:.2f} min_d {self.min_dist_m} "
                                        f"| occ_inner {self.occ_inner_pct:.1f}% occ_edge {self.occ_edge_pct:.1f}% "
                                        f"| occ_all {self.occ_pct} ttc {self.ttc_s} | follow {self.follow_active} "
                                        f"| post_hold {self._post_hold_active}")

            if self.follow_active:
                self._publish(False)
            else:
                self._publish(self.avoid_active)

            rate.sleep()

    # 신규: 우측 cut-in 감지 (간단 휴리스틱)
    def _detect_right_cutin(self, ref_path:Path, clusters_map:PoseArray, v_rel_est:float=None)->bool:
        if clusters_map is None or len(clusters_map.poses) == 0:
            return False
        # ref_path/ego pose 기반으로 프레네 근사
        xs = [ps.pose.position.x for ps in ref_path.poses]
        ys = [ps.pose.position.y for ps in ref_path.poses]
        # ego 인덱스 근사
        ex, ey = xs[0], ys[0]  # 이미 local_path면 0이 차량 근처
        d2 = [(x-ex)**2+(y-ey)**2 for x,y in zip(xs,ys)]
        j0 = int(np.argmin(d2))
        yaw = [math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)] + [0.0]
        th0 = yaw[j0]
        found = False
        for p in clusters_map.poses:
            dx, dy = p.position.x - xs[j0], p.position.y - ys[j0]
            lon =  math.cos(th0)*dx + math.sin(th0)*dy
            lat = -math.sin(th0)*dx + math.cos(th0)*dy
            if (abs(lat) >= self.cutin_lat_range[0] and abs(lat) <= self.cutin_lat_range[1] and
                lon >= self.cutin_ahead_range[0] and lon <= self.cutin_ahead_range[1]):
                found = True
                break
        # v_rel_est(접근 중) 조건은 있으면 추가
        if found and (v_rel_est is not None) and (v_rel_est < -self.cutin_vrel_thr):
            return True
        return found

    # --------- helpers ----------
    def _lat_err_to_global(self):
        path=self.local_path
        if path is None or len(path.poses)<2 or self.vx is None: return None
        xs=[p.pose.position.x for p in path.poses]; ys=[p.pose.position.y for p in path.poses]
        j=min(range(len(xs)), key=lambda i:(xs[i]-self.vx)**2+(ys[i]-self.vy)**2)
        if j>=len(xs)-1: j=max(0, j-1)
        th=math.atan2(ys[j+1]-ys[j], xs[j+1]-xs[j])
        dx,dy=self.vx-xs[j], self.vy-ys[j]
        lat = -math.sin(th)*dx + math.cos(th)*dy
        return abs(float(lat))

    def _req_clear(self):
        # >>> 여유폭 + avoid_extra_clear_m
        return 0.5*self.car_width + self.lat_margin + self.dynamic_lat_k * float(self.v_mps) + float(self.avoid_extra_clear_m)

    def _cur_s(self, path:Path, x, y):
        if path is None or len(path.poses)<2: return None
        xs=np.array([p.pose.position.x for p in path.poses]); ys=np.array([p.pose.position.y for p in path.poses])
        j=int(np.argmin((xs-x)**2+(ys-y)**2))
        ds=np.hypot(np.diff(xs),np.diff(ys)); s=np.concatenate([[0.0],np.cumsum(ds)])
        return float(s[j])

    def _count_on_path(self, path:Path, infl:float=0.4):
        cm=self.clusters_map
        if (cm is None) or (path is None) or (len(path.poses)<2): return 0, 999.0
        xs=[p.pose.position.x for p in path.poses]; ys=[p.pose.position.y for p in path.poses]
        vx,vy=self.vx,self.vy
        j=min(range(len(xs)), key=lambda i:(xs[i]-vx)**2+(ys[i]-vy)**2)
        j2=min(len(xs)-1, j+250)
        lane_w=self.road_half_w + infl
        ahead=min(self.corridor_ahead_m, 6.0+0.6*self.v_mps+(self.v_mps**2)/(2*3.0)+16.0)
        cnt=0; min_longi=999.0
        for o in cm.poses:
            ox,oy=o.position.x,o.position.y
            k=min(range(j,j2), key=lambda i:(xs[i]-ox)**2+(ys[i]-oy)**2)
            if k>=len(xs)-1: continue
            yaw=math.atan2(ys[k+1]-ys[k], xs[k+1]-xs[k])
            dx,dy=ox-xs[k],oy-ys[k]
            longi = math.cos(yaw)*dx + math.sin(yaw)*dy
            lat   =-math.sin(yaw)*dx + math.cos(yaw)*dy
            if 0.0<=longi<=ahead and abs(lat)<=lane_w:
                cnt+=1; min_longi=min(min_longi,longi)
        return cnt, min_longi

    def _build_bands(self, path:Path):
        cm=self.clusters_map
        if (cm is None) or (path is None) or (len(path.poses)<2): return []
        xs=np.array([p.pose.position.x for p in path.poses]); ys=np.array([p.pose.position.y for p in path.poses])
        vx,vy=self.vx,self.vy
        j0=int(np.argmin((xs-vx)**2+(ys-vy)**2))
        yaw=[math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)]; yaw.append(yaw[-1])
        N=int(self.corridor_ahead_m/self.band_dx)+1
        bands=[[] for _ in range(N)]
        for o in cm.poses:
            ox,oy=o.position.x,o.position.y
            k=min(range(j0, min(len(xs)-1,j0+250)), key=lambda i:(xs[i]-ox)**2+(ys[i]-oy)**2)
            th=yaw[k]
            dx,dy=ox-xs[k],oy-ys[k]
            longi= math.cos(th)*dx + math.sin(th)*dy
            lat  =-math.sin(th)*dx + math.cos(th)*dy
            if not (0.0<=longi<=self.corridor_ahead_m): continue
            if abs(lat) > (self.road_half_w + self.side_guard): continue
            b = int(longi // self.band_dx)
            rad = self._req_clear()
            inner = max(0.0, float(self.road_half_w - self.lat_margin))
            a0 = float(lat - rad); c0 = float(lat + rad)
            overlap = max(0.0, min(c0, inner) - max(a0, -inner))
            if overlap < float(self.edge_penetration_min): continue
            ay = abs(float(lat))
            if ay > inner and self.edge_ramp_band > 1e-3:
                edge_max = float(self.road_half_w + self.side_guard)
                dist_in_edge = max(0.0, min(self.edge_ramp_band, edge_max - ay))
                frac = max(0.0, min(1.0, dist_in_edge / float(self.edge_ramp_band)))
                scale = max(0.4, float(frac) ** float(self.edge_ramp_gamma))
                a0 = float(lat) - rad * scale
                c0 = float(lat) + rad * scale
            bands[b].append((a0, c0))

        # merge & dilate
        for i in range(N):
            iv=bands[i]
            if not iv: continue
            iv=sorted(iv, key=lambda x:x[0])
            m=[iv[0]]
            for a,c in iv[1:]:
                if a<=m[-1][1]: m[-1]=(m[-1][0], max(m[-1][1], c))
                else:           m.append((a,c))
            if self.band_dilate_m > 0.0:
                d = float(self.band_dilate_m)
                lim = self.road_half_w + self.side_guard + 1.0
                m = [ (max(-lim, a - d), min(lim, c + d)) for (a,c) in m ]
            bands[i]=m
        return bands

    # (기존) 중앙선 연속 점유 길이
    def _estimate_blocked_length_ahead(self, bands, lat0: float = 0.0) -> float:
        if not bands: return 0.0
        run = 0.0
        for row in bands:
            hit = any((a - 0.05) <= lat0 <= (c + 0.05) for (a, c) in row)
            if hit: run += float(self.band_dx)
            elif run > 0.0: break
        return run

    # (신규) 짧은 틈(gap) 허용 연속 길이
    def _estimate_run_with_gaps(self, bands, lat0: float = 0.0, gap_tol_m: float = 1.0) -> float:
        if not bands: return 0.0
        run = 0.0; gap = 0.0; started=False
        for row in bands:
            hit = any((a - 0.05) <= lat0 <= (c + 0.05) for (a,c) in row)
            if hit:
                if gap>0.0: run += gap; gap=0.0
                run += float(self.band_dx); started=True
            else:
                if started:
                    gap += float(self.band_dx)
                    if gap > float(gap_tol_m):
                        break
        return run

    def _row_hits_inner(self, row, inner):
        for a,c in row:
            if not (c < -inner or a > inner):
                return True
        return False

    def _speckle_density(self, bands, window_m=None) -> float:
        if not bands: return 0.0
        window_m = float(window_m if window_m is not None else self.speckle_window_m)
        n = min(len(bands), int(window_m/float(self.band_dx)))
        if n <= 0: return 0.0
        inner = float(self.road_half_w - self.lat_margin)
        rows  = bands[:n]
        total = len(rows); occupied=0
        for row in rows:
            if self._row_hits_inner(row, inner): occupied += 1
        return 100.0 * occupied / max(1, total)

    def _path_is_free(self, cand:Path, bands) -> bool:
        if (cand is None) or (not cand.poses) or (not bands) or (self.local_path is None): return True
        xs=np.array([p.pose.position.x for p in self.local_path.poses])
        ys=np.array([p.pose.position.y for p in self.local_path.poses])
        yaw=[math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)]; yaw.append(yaw[-1])
        j0=int(np.argmin((xs-self.vx)**2+(ys-self.vy)**2))
        for ps in cand.poses[::max(1,self.clearance_stride)]:
            k=min(range(j0, min(len(xs)-1, j0+250)),
                  key=lambda i:(xs[i]-ps.pose.position.x)**2+(ys[i]-ps.pose.position.y)**2)
            th=yaw[k]
            dx,dy=ps.pose.position.x-xs[k], ps.pose.position.y-ys[k]
            longi= math.cos(th)*dx + math.sin(th)*dy
            lat  =-math.sin(th)*dx + math.cos(th)*dy
            if longi<0.0 or longi>self.corridor_ahead_m: continue
            b=int(longi//self.band_dx)
            if b>=len(bands): break
            for a,c in bands[b]:
                if (a-0.05)<=lat<=(c+0.05): return False
        return True

    def _first_collision_m(self, cand:Path, bands) -> float:
        if cand is None or not cand.poses or not bands or self.local_path is None: return float('inf')
        xs=np.array([p.pose.position.x for p in self.local_path.poses])
        ys=np.array([p.pose.position.y for p in self.local_path.poses])
        yaw=[math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)]; yaw.append(yaw[-1])
        j0=int(np.argmin((xs-self.vx)**2+(ys-self.vy)**2))
        step=max(1, self.clearance_stride)
        for ps in cand.poses[::step]:
            k=min(range(j0, min(len(xs)-1, j0+250)),
                  key=lambda i:(xs[i]-ps.pose.position.x)**2+(ys[i]-ps.pose.position.y)**2)
            th=yaw[k]
            dx,dy=ps.pose.position.x-xs[k], ps.pose.position.y-ys[k]
            longi= math.cos(th)*dx + math.sin(th)*dy
            lat  =-math.sin(th)*dx + math.cos(th)*dy
            if longi<0.0: continue
            b=int(longi//self.band_dx)
            if 0<=b<len(bands):
                for a,c in bands[b]:
                    if (a-0.05)<=lat<=(c+0.05): return float(longi)
            if longi>self.corridor_ahead_m: break
        return float('inf')

    def _nearest_obstacle_side_ahead(self, ahead_m: float = 30.0) -> int:
        cm = self.clusters_map; path = self.local_path
        if cm is None or path is None or len(path.poses) < 2 or self.vx is None: return 0
        xs = np.array([p.pose.position.x for p in path.poses])
        ys = np.array([p.pose.position.y for p in path.poses])
        j0 = int(np.argmin((xs - self.vx) ** 2 + (ys - self.vy) ** 2))
        if j0 >= len(xs) - 1: j0 = max(0, len(xs) - 2)
        yaw = [math.atan2(ys[i+1]-ys[i], xs[i+1]-xs[i]) for i in range(len(xs)-1)]; yaw.append(yaw[-1])
        best_longi = float('inf'); best_lat = 0.0
        lane_guard = self.road_half_w + self.side_guard
        kmax = min(len(xs) - 1, j0 + 250)
        for o in cm.poses:
            ox, oy = o.position.x, o.position.y
            k = min(range(j0, kmax), key=lambda i: (xs[i]-ox)**2 + (ys[i]-oy)**2)
            th = yaw[k]
            dx, dy = ox - xs[k], oy - ys[k]
            longi = math.cos(th)*dx + math.sin(th)*dy
            lat   = -math.sin(th)*dx + math.cos(th)*dy
            if 0.0 <= longi <= ahead_m and abs(lat) <= lane_guard:
                if longi < best_longi:
                    best_longi = longi; best_lat = lat
        if math.isinf(best_longi): return 0
        return 1 if best_lat > 0.0 else -1

    def _plan_hugging(self, Ti: np.ndarray, xe: float, lane_w: float, heading: float):
        if not self.hug_enable or not self.cluster_features:
            return None
        best = None
        max_ahead = float(min(self.corridor_ahead_m, xe))
        for feat in self.cluster_features:
            pt = np.array([[feat['x']], [feat['y']], [1.0]], dtype=float)
            local = Ti.dot(pt)
            lx = float(local[0]); ly = float(local[1])
            if lx <= 1.0 or lx >= max_ahead:
                continue
            if abs(ly) > lane_w + float(self.side_guard):
                continue
            half_w = max(0.05, float(feat['half_width']))
            half_l = max(0.05, float(feat['half_len']))
            if best is None or lx < best['lx']:
                best = {
                    'lx': lx,
                    'ly': ly,
                    'half_w': half_w,
                    'half_l': half_l,
                    'yaw': float(feat['yaw'])
                }
        if best is None:
            return None

        side = -1.0 if best['ly'] >= 0.0 else 1.0
        clearance = float(self.required_clear + self.avoid_extra_clear_m)
        base_lat = best['ly'] + side * (best['half_w'] + clearance)
        base_lat = float(np.clip(base_lat, -lane_w, lane_w))
        secondary = base_lat + side * float(self.hug_secondary_margin)
        secondary = float(np.clip(secondary, -lane_w, lane_w))

        entry_x = max(2.5, best['lx'] - best['half_l'] - float(self.hug_entry_buffer))
        exit_x = min(xe - 1.0, best['lx'] + best['half_l'] + float(self.hug_exit_buffer))
        if exit_x <= entry_x + 0.5:
            exit_x = min(xe - 0.5, entry_x + 1.5)
        entry_ratio = max(0.05, min(0.8, entry_x / max(1e-3, xe)))
        exit_ratio = max(entry_ratio + 0.05, min(0.95, exit_x / max(1e-3, xe)))

        yaw_local = math.atan2(math.sin(best['yaw'] - heading), math.cos(best['yaw'] - heading))

        return {
            'lat_targets': [base_lat, secondary],
            'entry_ratio': entry_ratio,
            'exit_ratio': exit_ratio,
            'yaw_local': yaw_local,
            'side': side
        }

    def _slice_path_ahead(self, path: Path, x: float, y: float, length_m: float) -> Path:
        if path is None or len(path.poses) < 2 or x is None or y is None: return None
        xs = np.array([p.pose.position.x for p in path.poses], dtype=float)
        ys = np.array([p.pose.position.y for p in path.poses], dtype=float)
        j0 = int(np.argmin((xs - x)**2 + (ys - y)**2))
        ds = np.hypot(np.diff(xs), np.diff(ys)); s = np.concatenate([[0.0], np.cumsum(ds)])
        s0 = s[j0]; target = s0 + float(length_m)
        j1 = j0
        while j1 < len(s)-1 and s[j1] < target: j1 += 1
        if j1 <= j0+1: return None
        out = Path(); out.header.frame_id = path.header.frame_id if path.header.frame_id else 'map'
        out.header.stamp = rospy.Time.now()
        for k in range(j0, j1+1): out.poses.append(path.poses[k])
        return out

    # ────── 직선 post-hold 시작/생성 ──────
    def _start_post_hold(self):
        base = random.uniform(self.post_hold_min_s, self.post_hold_max_s)
        hold_s = float(max(self.post_hold_min_s, min(self.post_hold_max_s, base)))
        hold_s += float(self.post_hold_extra_s)  # 복귀 지연 추가
        length = max(float(self.post_hold_min_m), float(self.v_mps) * hold_s)
        self._post_hold_path = self._make_straight_path_ahead(length, step=self.post_hold_step_m)
        self._post_hold_active  = True
        self._post_hold_until_t = rospy.get_time() + hold_s
        rospy.loginfo("[lattice] POST-HOLD start: %.2fs(+%.2fs), len=%.1fm (speed_control=True)",
                      hold_s, float(self.post_hold_extra_s), length)

    def _make_straight_path_ahead(self, length_m: float, step: float = 0.5) -> Path:
        path = self.local_path
        if path is None or len(path.poses) < 2 or self.vx is None or self.vy is None:
            return None
        xs = [p.pose.position.x for p in path.poses]; ys = [p.pose.position.y for p in path.poses]
        j = min(range(len(xs)), key=lambda i:(xs[i]-self.vx)**2+(ys[i]-self.vy)**2)
        if j >= len(xs)-1: j = max(0, j-1)
        th = math.atan2(ys[j+1]-ys[j], xs[j+1]-xs[j])  # 현재 진행 방향
        n = max(2, int(length_m / max(0.1, float(step))) + 1)
        out = Path(); out.header.frame_id = 'map'; out.header.stamp = rospy.Time.now()
        for i in range(n):
            d = float(i) * float(step)
            px = float(self.vx) + d * math.cos(th)
            py = float(self.vy) + d * math.sin(th)
            ps = PoseStamped()
            ps.pose.position.x = px; ps.pose.position.y = py; ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            out.poses.append(ps)
        return out

    # --------- publish ----------
    def _publish(self, avoid:bool):
        if avoid:
            if self._post_hold_active and (self._post_hold_path is not None) and (len(self._post_hold_path.poses) >= 2):
                self.pub_path.publish(self._post_hold_path)
                self.pub_speed.publish(Bool(True))
                rospy.loginfo_throttle(0.5, "[lattice] POST-HOLD publishing straight path (speed_control=True)")
                return

            if self._rejoin_mode and self.rejoin_hard:
                p = self._slice_path_ahead(self.local_path, self.vx, self.vy, self.rejoin_slice_m)
                if p is not None and len(p.poses) >= 2:
                    self.pub_path.publish(p)
                    self.pub_speed.publish(Bool(False))
                    self._last_idx = None
                    rospy.loginfo("[lattice] REJOIN-HARD -> publish global slice %.1fm, speed_control=False", self.rejoin_slice_m)
                    return
                else:
                    rospy.logwarn("[lattice] rejoin_hard slice failed -> fallback to candidates")

            cands=self._make_candidates(self.local_path)
            if not cands or max(len(p.poses) for p in cands) < self.min_points:
                if self.local_path:
                    self.pub_path.publish(self.local_path)
                self.pub_speed.publish(Bool(True))
                rospy.logwarn("[lattice] poor candidates -> fallback to local_path")
                return

            bands=self._build_bands(self.local_path)
            free=[i for i,p in enumerate(cands) if self._path_is_free(p,bands)]
            center=len(cands)//2
            safe = [i for i in free if self._first_collision_m(cands[i], bands) >= self.min_free_prefix_m]
            pool = safe if safe else free

            side_hint = self._nearest_obstacle_side_ahead(ahead_m=min(self.corridor_ahead_m, 30.0))
            # 컷인 가드 래치 중이면 왼쪽(큰 편차) 후보를 피한다
            if getattr(self, "_cutin_guard_active", False):
                side_hint = -1  # 오른쪽/차로유지 쪽을 선호
                pool = [i for i in pool if i >= center] or pool  # 왼쪽 후보 대폭 억제

            if side_hint > 0:
                pool = [i for i in pool if i < center] or pool
            elif side_hint == 0 and self.default_avoid_side < 0:
                prefer = [i for i in pool if i < center]
                pool = prefer or pool

            if self._rejoin_mode and pool and not self.rejoin_hard:
                idx = min(pool, key=lambda i: abs(i-center))
            else:
                if pool:
                    if self._sticky_side==0:
                        left_w  = sum(sum(abs(a) for a,_ in row if a<0) for row in bands)
                        right_w = sum(sum(c for a,c in row if c>0) for row in bands)
                        self._sticky_side = -1 if left_w < right_w else +1
                    side = [i for i in pool if (i<center if self._sticky_side<0 else i>center)]
                    choose_from = side or pool
                    idx = max(choose_from, key=lambda i: self._clearance_percentile(
                        cands[i], self.clearance_window_m, self.clearance_percentile_q))
                else:
                    backup = free if free else list(range(len(cands)))
                    if side_hint > 0:
                        backup_pref = [i for i in backup if i < center] or backup
                    elif side_hint < 0:
                        backup_pref = [i for i in backup if i > center] or backup
                    elif self.default_avoid_side < 0:
                        backup_pref = [i for i in backup if i < center] or backup
                    else:
                        backup_pref = backup
                    if backup_pref:
                        idx = max(backup_pref, key=lambda i: self._clearance_percentile(
                            cands[i], self.clearance_window_m, self.clearance_percentile_q))
                    else:
                        idx = center

            if self._last_idx is not None and avoid and not self._rejoin_mode:
                cap = int(max(1, self.idx_step_cap))
                if idx > self._last_idx + cap: idx = self._last_idx + cap
                elif idx < self._last_idx - cap: idx = self._last_idx - cap
            self._last_idx = idx
            idx=max(0, min(idx, len(cands)-1))

            self.pub_path.publish(cands[idx])
            self.pub_speed.publish(Bool(True))
            rospy.loginfo("[lattice] avoid ON, choose #%d (rejoin=%s, side_hint=%d, safe_pool=%d/%d)",
                          idx, self._rejoin_mode, side_hint, len(safe), len(free))
        else:
            if self.local_path:
                self.pub_path.publish(self.local_path)
                self.pub_speed.publish(Bool(False))
                self._last_idx = None
            rospy.loginfo("[lattice] avoid OFF -> follow global")

    # --------- scoring fallback ----------
    def _choose_by_clearance(self, cands):
        need=self._req_clear()
        weights=[]
        for p in cands:
            cl=self._min_clear(p, self.clusters_map, self.clearance_stride)
            w = 1_000_000 if cl<need else self.clearance_soft_w/max(0.1, cl)
            weights.append(w)
        return int(np.argmin(weights))

    def _min_clear(self, path:Path, cm:PoseArray, stride:int):
        if (path is None) or (not path.poses) or (cm is None) or (not cm.poses): return float('inf')
        pts=path.poses[::max(1,stride)]
        m=float('inf')
        for ps in pts:
            px,py=ps.pose.position.x, ps.pose.position.y
            for o in cm.poses:
                d=math.hypot(px-o.position.x, py-o.position.y)
                if d<m: m=d
        return m

    def _clearance_percentile(self, cand: Path, window_m: float, q: float) -> float:
        cm = self.clusters_map
        if cand is None or not cand.poses or cm is None or not cm.poses: return float('inf')
        xs = np.array([ps.pose.position.x for ps in cand.poses], dtype=float)
        ys = np.array([ps.pose.position.y for ps in cand.poses], dtype=float)
        if len(xs) >= 2:
            ds = np.hypot(np.diff(xs), np.diff(ys))
            s = np.concatenate([[0.0], np.cumsum(ds)])
            mask = s <= float(window_m)
            if not np.any(mask): mask[:] = True
            pts = np.stack([xs[mask], ys[mask]], axis=1)
        else:
            pts = np.stack([xs, ys], axis=1)
        obs = np.array([[o.position.x, o.position.y] for o in cm.poses], dtype=float)
        if obs.size == 0: return float('inf')
        mins = []
        step = max(1, int(self.clearance_stride))
        for px, py in pts[::step]:
            d = np.hypot(obs[:, 0] - px, obs[:, 1] - py)
            mins.append(float(np.min(d)))
        if not mins: return float('inf')
        return float(np.percentile(mins, float(q)))

    # --------- candidates (Bezier) ----------
    def _make_candidates(self, ref:Path):
        out=[]
        if ref is None or self.vx is None: return out

        xs=[p.pose.position.x for p in ref.poses]; ys=[p.pose.position.y for p in ref.poses]
        if len(xs)<3: return out
        j=int(np.argmin((np.array(xs)-self.vx)**2 + (np.array(ys)-self.vy)**2))
        v_kph=self.v_kph; look=int(max(20, v_kph*0.4))
        s=max(0, j-1); e=min(len(xs)-1, s+look*2)
        sx,sy=xs[s],ys[s]; nx,ny=xs[s+1],ys[s+1]; ex,ey=xs[e],ys[e]
        th=atan2(ny-sy, nx-sx)
        T =np.array([[cos(th),-sin(th),sx],[sin(th),cos(th),sy],[0,0,1]], dtype=float)
        Ti=np.array([[ T[0,0], T[1,0], -(T[0,0]*sx + T[1,0]*sy)],
                     [ T[0,1], T[1,1], -(T[0,1]*sx + T[1,1]*sy)],
                     [0,0,1]], dtype=float)
        le = Ti.dot(np.array([[ex],[ey],[1]], dtype=float)); xe=float(le[0]); y_end=float(le[1])
        le0= Ti.dot(np.array([[self.vx],[self.vy],[1]], dtype=float)); y0=float(le0[1])
        if xe<=1.0:
            e=min(len(xs)-1, s+look*3); ex,ey=xs[e],ys[e]; le=Ti.dot(np.array([[ex],[ey],[1]], dtype=float)); xe=float(le[0])
        if xe<=1.0: return out

        bands = self._build_bands(ref)
        blocked_len = self._estimate_blocked_length_ahead(bands, lat0=0.0)
        chain_len   = self._estimate_run_with_gaps(bands, lat0=0.0, gap_tol_m=self.speckle_gap_close_m)
        density     = self._speckle_density(bands, window_m=self.speckle_window_m)

        is_long   = (blocked_len >= float(self.long_block_threshold_m))
        is_chain  = (chain_len  >= float(self.long_block_threshold_m)*0.75)
        is_speckle= (density    >= float(self.speckle_density_pct)) and (not is_long)

        if self.avoid_active:
            xe *= float(np.clip(self.avoid_x_scale, 0.8, 1.0))
        if is_long or is_chain:
            xe *= float(self.long_xe_scale)
        elif is_speckle:
            xe *= float(self.speckle_xe_scale)

        xe = min(max(xe, max(self.min_x_end_m, 4.0+2.0*self.v_mps)), 80.0)
        W  = self.road_half_w - self.edge_margin
        gw = float(self.avoid_width_gain)  # <<< 회피 폭 확대 게인

        if is_long or is_chain:
            # 첫 번째 후보는 long_first_ofs_gain과 gw를 모두 반영
            offsets   = [ 1.42*W*gw, 1.8*W*gw]
            bez_alpha = float(self.long_alpha)
            bez_push  = float(self.long_push)
            bez_beta  = float(self.bez_beta)
            bez_relax = float(self.long_relax)
            entry_gain= 1.0 + min(float(self.long_entry_boost_cap), float(self.long_entry_boost_k)*float(self.v_mps))
            alpha_eff = max(0.04, bez_alpha * max(0.6, 1.0 - 0.03*float(self.v_mps)))
        elif is_speckle:
            offsets   = [ 1.42*W*gw, 1.8*W*gw]
            bez_alpha = float(self.bez_alpha)
            bez_push  = float(self.bez_push)
            bez_beta  = float(self.bez_beta)
            bez_relax = float(self.bez_relax)
            entry_gain= 1.0
            alpha_eff = bez_alpha
        else:
            offsets   = [ 1.42*W*gw, 1.8*W*gw]
            bez_alpha = float(self.bez_alpha)
            bez_push  = float(self.bez_push)
            bez_beta  = float(self.bez_beta)
            bez_relax = float(self.bez_relax)
            entry_gain= 1.0
            alpha_eff = bez_alpha

        lat_targets = [y_end + ofs for ofs in offsets]
        hug_plan = self._plan_hugging(Ti, xe, W, th)
        hugging_active = False
        if hug_plan:
            hugging_active = True
            lat_targets = hug_plan['lat_targets']
            offsets = [lat - y_end for lat in lat_targets]
            alpha_eff = hug_plan['entry_ratio']
            bez_beta = 1.0 - hug_plan['exit_ratio']
            bez_push = float(self.hug_entry_shape)
            bez_relax = float(self.hug_exit_relax)
            entry_gain = 1.0

        N=max(self.min_points, int(xe/0.5)+1)
        for target_lat, ofs in zip(lat_targets, offsets):
            P0=np.array([0.0, y0], float)
            P3=np.array([xe, target_lat], float)
            if hugging_active:
                entry_ratio = hug_plan['entry_ratio']
                exit_ratio  = hug_plan['exit_ratio']
                entry_lat = y0 + (target_lat - y0) * float(self.hug_entry_shape)
                span = max(0.0, (exit_ratio - entry_ratio) * xe)
                slope = math.tan(hug_plan['yaw_local']) if span > 1e-3 else 0.0
                slope = float(np.clip(slope, -float(self.hug_tangent_clip), float(self.hug_tangent_clip)))
                hold_lat = target_lat + slope * span * float(self.hug_parallel_scale)
                hold_lat = float(np.clip(hold_lat, -W, W))
                P1=np.array([xe*entry_ratio, entry_lat], float)
                P2=np.array([xe*exit_ratio, hold_lat], float)
            else:
                P1=np.array([xe*alpha_eff, y0 + ofs*bez_push*entry_gain], float)
                P2=np.array([xe*(1.0-bez_beta), P3[1]-ofs*bez_relax], float)
            lp=Path(); lp.header.frame_id='map'
            for i in range(N):
                t=i/float(N-1)
                B=((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3
                gp=T.dot(np.array([[B[0]],[B[1]],[1.0]], dtype=float))
                ps=PoseStamped(); ps.pose.position.x=float(gp[0,0]); ps.pose.position.y=float(gp[1,0])
                ps.pose.position.z=0.0; ps.pose.orientation.w=1.0
                lp.poses.append(ps)
            out.append(lp)
        return out


if __name__ == '__main__':
    try:
        latticePlanner()
    except rospy.ROSInterruptException:
        pass
