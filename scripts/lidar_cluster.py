#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy, math, numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sensor_msgs.msg import PointCloud2, Imu
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32MultiArray, Bool


# ─────────────────────────────────────────────────────────────────────────────
# 프로파일 정의 (모드별 임계/ROI/클러스터 파라미터)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Profile:
    # ROI/FOV
    corridor_y: float
    fov_deg:    float
    ahead_m:    float
    max_r:      float
    z_max:      float

    # ground/height thresholds
    gnd_clear: float
    dynamic_clear_r0: float
    far_clear_gain: float
    low_slope_thresh: float
    low_slope_extra: float
    hi_slope_thresh: float
    hi_slope_extra: float
    downhill_pitch_deg: float
    downhill_h_extra: float
    uphill_pitch_deg: float
    uphill_h_extra: float
    uphill_convex_extra: float

    # edge ramp / curb suppression
    edge_ramp_enable: bool
    edge_guard_y: float
    obs_shrink_y: float
    obs_edge_h_min: float
    curb_h_cap: float
    edge_ramp_gain: float
    edge_ramp_h_cap: float
    edge_preserve_h: float

    # curb shape gate
    curb_shape_gate_enable: bool
    curb_edge_guard_y: float
    curb_shape_w_max: float
    curb_shape_l_min: float

    # clustering gates
    db_eps: float
    db_min_samples: int
    cluster_min_pts: int
    cluster_max_pts: int
    box_w_min: float
    box_w_max: float
    box_l_min: float
    box_l_max: float


# ─────────────────────────────────────────────────────────────────────────────
# 메인 노드: 전처리 공유 + 듀얼 프로파일 스위칭
# ─────────────────────────────────────────────────────────────────────────────
class LidarClusterDual:
    def __init__(self):
        rospy.init_node('velodyne_clustering', anonymous=True)

        # Subs / Pubs
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback, queue_size=1)
        self.cluster_pub        = rospy.Publisher("/clusters", PoseArray, queue_size=1)
        self.distance_pub       = rospy.Publisher("/cluster_distances", Float32MultiArray, queue_size=1)
        self.filtered_pub       = rospy.Publisher("/lidar_filtered", PointCloud2, queue_size=1)
        # 디버깅용(프로파일별 결과도 발행)
        self.cluster_pub_cruise  = rospy.Publisher("/clusters_cruise",  PoseArray, queue_size=1)
        self.cluster_pub_evasive = rospy.Publisher("/clusters_evasive", PoseArray, queue_size=1)
        self.filtered_pub_cruise = rospy.Publisher("/lidar_filtered_cruise", PointCloud2, queue_size=1)
        self.filtered_pub_evasive= rospy.Publisher("/lidar_filtered_evasive", PointCloud2, queue_size=1)

        # 파라미터 헬퍼
        gp = rospy.get_param

        # 공통/공유 전처리 파라미터 (기본은 넉넉하게)
        self.min_r      = float(gp("~min_r", 1.65))
        self.voxel_leaf = float(gp("~voxel_leaf", 0.05))

        # 지면 모델링 공통
        self.use_adaptive_ground = bool(gp("~use_adaptive_ground", True))
        self.ground_bins   = int(gp("~ground_bins", 180))
        self.gnd_near_m    = float(gp("~gnd_near_m", 14.0))
        self.use_quad_ground = bool(gp("~use_quad_ground", True))
        self.gnd_mid_m     = float(gp("~gnd_mid_m", 44.0))
        self.gnd_slope_cap = float(gp("~gnd_slope_cap", 0.20))
        self.gnd_curv_cap  = float(gp("~gnd_curv_cap", 0.030))
        self.ground_smooth_span = int(gp("~ground_smooth_span", 8))

        # 노면 잔존/볼록성 공통
        self.convex_dr    = float(gp("~convex_dr", 0.8))
        self.convex_clear = float(gp("~convex_clear", 0.10))

        # IMU leveling
        self.use_imu_leveling = bool(gp("~use_imu_leveling", True))
        self.imu_topic        = gp("~imu_topic", "/imu/data")
        self.pitch_deg = 0.0; self.roll_deg  = 0.0
        if self.use_imu_leveling:
            rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=10)

        # 내리막/오르막 보정 + 스무딩 다이내믹
        self.dynamic_smooth_gain = float(gp("~dynamic_smooth_gain", 0.40))
        self.ground_smooth_max   = int(gp("~ground_smooth_max", 4))

        # ── 주행/회피 프로파일 구성
        # 네가 준 "예전 파라미터 세트"를 cruise 기본값으로 넣음
        def p(ns: str, key: str, default):
            return gp(f"~{ns}/{key}", default)

        self.pro_cruise = Profile(
            corridor_y = p("cruise", "corridor_y", 2.60),
            fov_deg    = p("cruise", "fov_deg",    80.0),
            ahead_m    = p("cruise", "ahead_m",    80.0),
            max_r      = p("cruise", "max_r",      45.0),
            z_max      = p("cruise", "z_max",       1.6),

            gnd_clear  = p("cruise", "gnd_clear",  0.30),
            dynamic_clear_r0 = p("cruise", "dynamic_clear_r0", 8.5),
            far_clear_gain   = p("cruise", "far_clear_gain",   0.0180),
            low_slope_thresh = p("cruise", "low_slope_thresh", 0.045),
            low_slope_extra  = p("cruise", "low_slope_extra",  0.10),
            hi_slope_thresh  = p("cruise", "hi_slope_thresh",  0.065),
            hi_slope_extra   = p("cruise", "hi_slope_extra",   0.10),
            downhill_pitch_deg = p("cruise", "downhill_pitch_deg", 2.0),
            downhill_h_extra   = p("cruise", "downhill_h_extra",   0.05),
            uphill_pitch_deg   = p("cruise", "uphill_pitch_deg",   0.7),
            uphill_h_extra     = p("cruise", "uphill_h_extra",     0.12),
            uphill_convex_extra  = p("cruise", "uphill_convex_extra",0.10),

            edge_ramp_enable = p("cruise", "edge_ramp_enable", True),
            edge_guard_y     = p("cruise", "edge_guard_y",     0.70),
            obs_shrink_y     = p("cruise", "obs_shrink_y",     0.82),
            obs_edge_h_min   = p("cruise", "obs_edge_h_min",   0.40),
            curb_h_cap       = p("cruise", "curb_h_cap",       0.40),
            edge_ramp_gain   = p("cruise", "edge_ramp_gain",   1.05),
            edge_ramp_h_cap  = p("cruise", "edge_ramp_h_cap",  0.90),
            edge_preserve_h  = p("cruise", "edge_preserve_h",  0.62),

            curb_shape_gate_enable = p("cruise", "curb_shape_gate_enable", True),
            curb_edge_guard_y      = p("cruise", "curb_edge_guard_y",      0.80),
            curb_shape_w_max       = p("cruise", "curb_shape_w_max",       0.16),
            curb_shape_l_min       = p("cruise", "curb_shape_l_min",       3.00),

            db_eps         = p("cruise", "db_eps",         0.37),
            db_min_samples = p("cruise", "db_min_samples", 7),
            cluster_min_pts= p("cruise", "cluster_min_pts",3),
            cluster_max_pts= p("cruise", "cluster_max_pts",600),
            box_w_min      = p("cruise", "box_w_min",      0.15),
            box_w_max      = p("cruise", "box_w_max",      3.2),
            box_l_min      = p("cruise", "box_l_min",      0.15),
            box_l_max      = p("cruise", "box_l_max",      7.5),
        )

        # 회피 프로파일: 넓은 시야/차로폭 기본값(원래 네 현행 코드 성향) + 약간 민감한 DBSCAN
        self.pro_evasive = Profile(
            corridor_y = p("evasive", "corridor_y", 4.00),
            fov_deg    = p("evasive", "fov_deg",    90.0),
            ahead_m    = p("evasive", "ahead_m",    80.0),
            max_r      = p("evasive", "max_r",      45.0),
            z_max      = p("evasive", "z_max",       1.6),

            gnd_clear  = p("evasive", "gnd_clear",  self.pro_cruise.gnd_clear),
            dynamic_clear_r0 = p("evasive", "dynamic_clear_r0", self.pro_cruise.dynamic_clear_r0),
            far_clear_gain   = p("evasive", "far_clear_gain",   self.pro_cruise.far_clear_gain),
            low_slope_thresh = p("evasive", "low_slope_thresh", self.pro_cruise.low_slope_thresh),
            low_slope_extra  = p("evasive", "low_slope_extra",  self.pro_cruise.low_slope_extra),
            hi_slope_thresh  = p("evasive", "hi_slope_thresh",  self.pro_cruise.hi_slope_thresh),
            hi_slope_extra   = p("evasive", "hi_slope_extra",   self.pro_cruise.hi_slope_extra),
            downhill_pitch_deg = p("evasive", "downhill_pitch_deg", self.pro_cruise.downhill_pitch_deg),
            downhill_h_extra   = p("evasive", "downhill_h_extra",   self.pro_cruise.downhill_h_extra),
            uphill_pitch_deg   = p("evasive", "uphill_pitch_deg",   self.pro_cruise.uphill_pitch_deg),
            uphill_h_extra     = p("evasive", "uphill_h_extra",     self.pro_cruise.uphill_h_extra),
            uphill_convex_extra  = p("evasive", "uphill_convex_extra",self.pro_cruise.uphill_convex_extra),

            edge_ramp_enable = p("evasive", "edge_ramp_enable", True),
            edge_guard_y     = p("evasive", "edge_guard_y",     0.70),
            obs_shrink_y     = p("evasive", "obs_shrink_y",     0.82),
            obs_edge_h_min   = p("evasive", "obs_edge_h_min",   0.40),
            curb_h_cap       = p("evasive", "curb_h_cap",       0.40),
            edge_ramp_gain   = p("evasive", "edge_ramp_gain",   1.05),
            edge_ramp_h_cap  = p("evasive", "edge_ramp_h_cap",  0.90),
            edge_preserve_h  = p("evasive", "edge_preserve_h",  0.62),

            curb_shape_gate_enable = p("evasive", "curb_shape_gate_enable", True),
            curb_edge_guard_y      = p("evasive", "curb_edge_guard_y",      0.80),
            curb_shape_w_max       = p("evasive", "curb_shape_w_max",       0.16),
            curb_shape_l_min       = p("evasive", "curb_shape_l_min",       3.00),

            db_eps         = p("evasive", "db_eps",         0.29),
            db_min_samples = p("evasive", "db_min_samples", 7),
            cluster_min_pts= p("evasive", "cluster_min_pts",3),
            cluster_max_pts= p("evasive", "cluster_max_pts",600),
            box_w_min      = p("evasive", "box_w_min",      0.15),
            box_w_max      = p("evasive", "box_w_max",      3.2),
            box_l_min      = p("evasive", "box_l_min",      0.15),
            box_l_max      = p("evasive", "box_l_max",      7.5),
        )

        # 공유 전처리 ROI = 두 프로파일의 합집합(더 넓은 값)
        self.shared_corridor_y = max(self.pro_cruise.corridor_y, self.pro_evasive.corridor_y)
        self.shared_fov_deg    = max(self.pro_cruise.fov_deg,    self.pro_evasive.fov_deg)
        self.shared_ahead_m    = max(self.pro_cruise.ahead_m,    self.pro_evasive.ahead_m)
        self.shared_max_r      = max(self.pro_cruise.max_r,      self.pro_evasive.max_r)
        self.shared_z_max      = max(self.pro_cruise.z_max,      self.pro_evasive.z_max)

        # 모드 스위치 (히스테리시스)
        self._desired_avoid = False
        self._avoid = False
        self._hcount = 0
        self.on_frames  = int(gp("~avoid_on_frames",  6))
        self.off_frames = int(gp("~avoid_off_frames", 12))
        rospy.Subscriber("/avoid_mode", Bool, self._avoid_cb, queue_size=1)

    # ─────────────────────────────────────────────────────────────────────
    # IMU 콜백: quaternion → roll/pitch
    # ─────────────────────────────────────────────────────────────────────
    def imu_cb(self, msg: Imu):
        q = msg.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        # roll
        sinr_cosp = 2.0*(w*x + y*z); cosr_cosp = 1.0 - 2.0*(x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = 2.0*(w*y - z*x)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        self.roll_deg  = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)

    # ─────────────────────────────────────────────────────────────────────
    # 모드 히스테리시스
    # ─────────────────────────────────────────────────────────────────────
    def _avoid_cb(self, msg: Bool):
        self._desired_avoid = bool(msg.data)

    def _decide_mode(self) -> bool:
        if self._desired_avoid:
            self._hcount = min(self._hcount + 1, self.on_frames)
        else:
            self._hcount = max(self._hcount - 1, -self.off_frames)
        if self._hcount >= self.on_frames:   self._avoid = True
        if self._hcount <= -self.off_frames: self._avoid = False
        return self._avoid

    # ─────────────────────────────────────────────────────────────────────
    # 공용 전처리: IMU 레벨링 → 공유 ROI → 지면 모델 → 포인트별 메트릭 산출
    # 반환: (hdr, metrics, vis_all_xyz)
    #  metrics: list of (x,y,z,r,bi,h,slope,h_cv,yabs)
    # ─────────────────────────────────────────────────────────────────────
    def preprocess_metrics(self, cloud_msg: PointCloud2):
        hdr = cloud_msg.header
        bins = self.ground_bins

        # bin별 최소 z (근/중/원)
        znear = np.full(bins, np.inf, dtype=np.float32); rnear = np.full(bins, np.inf, dtype=np.float32)
        zmid  = np.full(bins, np.inf, dtype=np.float32); rmid  = np.full(bins, np.inf, dtype=np.float32)
        zfar  = np.full(bins, np.inf, dtype=np.float32); rfar  = np.full(bins, -np.inf, dtype=np.float32)
        near_max = float(self.gnd_near_m); mid_max = float(self.gnd_mid_m)

        # IMU leveling 회전
        use_lvl = self.use_imu_leveling
        pr = math.radians(self.pitch_deg); rr = math.radians(self.roll_deg)
        if use_lvl:
            cp, sp = math.cos(-pr), math.sin(-pr)  # Ry(-pitch)
            cr, sr = math.cos(-rr), math.sin(-rr)  # Rx(-roll)

        cand = []
        vis_all_xyz = []

        # 1) IMU leveling + 공유 ROI 컷
        for x0, y0, z0 in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            if use_lvl:
                x1 =  cp*x0 + sp*z0; y1 =  y0; z1 = -sp*x0 + cp*z0
                x  =  x1; y  =  cr*y1 - sr*z1; z  =  sr*y1 + cr*z1
            else:
                x, y, z = x0, y0, z0

            r = math.hypot(x, y)
            if not (self.min_r < r < self.shared_max_r): continue
            if not (0.0 < x <= self.shared_ahead_m):     continue
            if abs(y) > self.shared_corridor_y:          continue
            if z > self.shared_z_max:                    continue
            if abs(math.degrees(math.atan2(y, x))) > self.shared_fov_deg: continue

            # bin index (각도 기반)
            bi = int(((math.atan2(y, x) + math.pi) / (2.0 * math.pi)) * bins) % bins

            # ground 후보 통계
            if r <= near_max:
                if z < znear[bi]: znear[bi], rnear[bi] = z, r
            elif r <= mid_max:
                if z < zmid[bi]:  zmid[bi],  rmid[bi]  = z, r
            else:
                if z < zfar[bi]:  zfar[bi],  rfar[bi]  = z, r

            cand.append((x, y, z, bi, r))
            vis_all_xyz.append((x, y, z))

        if not cand:
            return hdr, [], []

        # 2) bin별 지면 모델 (a r^2 + b r + c)
        acoef = np.zeros(bins, dtype=np.float32)
        bcoef = np.zeros(bins, dtype=np.float32)
        ccoef = np.zeros(bins, dtype=np.float32)

        if self.use_quad_ground:
            for bi in range(bins):
                pts_r, pts_z = [], []
                if np.isfinite(znear[bi]): pts_r.append(float(rnear[bi])); pts_z.append(float(znear[bi]))
                if np.isfinite(zmid[bi]):  pts_r.append(float(rmid[bi]));  pts_z.append(float(zmid[bi]))
                if np.isfinite(zfar[bi]):  pts_r.append(float(rfar[bi]));  pts_z.append(float(zfar[bi]))
                n = len(pts_r)
                if n >= 3:
                    a, b, c = np.polyfit(np.array(pts_r), np.array(pts_z), 2)
                    a = float(max(-self.gnd_curv_cap, min(self.gnd_curv_cap, a)))
                    acoef[bi], bcoef[bi], ccoef[bi] = a, float(b), float(c)
                elif n == 2:
                    b, c = np.polyfit(np.array(pts_r), np.array(pts_z), 1)
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, float(b), float(c)
                elif n == 1:
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, 0.0, float(pts_z[0])
                else:
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, 0.0, 0.0
        else:
            # 선형 모델만 사용할 경우
            for bi in range(bins):
                pts_r, pts_z = [], []
                if np.isfinite(znear[bi]): pts_r.append(float(rnear[bi])); pts_z.append(float(znear[bi]))
                if np.isfinite(zmid[bi]):  pts_r.append(float(rmid[bi]));  pts_z.append(float(zmid[bi]))
                if np.isfinite(zfar[bi]):  pts_r.append(float(rfar[bi]));  pts_z.append(float(zfar[bi]))
                n = len(pts_r)
                if n >= 2:
                    b, c = np.polyfit(np.array(pts_r), np.array(pts_z), 1)
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, float(b), float(c)
                elif n == 1:
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, 0.0, float(pts_z[0])
                else:
                    acoef[bi], bcoef[bi], ccoef[bi] = 0.0, 0.0, 0.0

        # 3) cross-bin smoothing (+ uphill expand)
        K = int(max(0, self.ground_smooth_span))
        if self.use_imu_leveling and self.pitch_deg > self.pro_cruise.uphill_pitch_deg:
            addK = int(self.dynamic_smooth_gain * (self.pitch_deg - self.pro_cruise.uphill_pitch_deg))
            K = min(self.ground_smooth_max, K + max(0, addK))

        if K > 0:
            W = 2*K + 1
            def smooth(arr):
                out = np.zeros_like(arr, dtype=np.float32)
                for o in range(-K, K+1): out += np.roll(arr, o)
                return (out / float(W)).astype(np.float32)
            acoef = smooth(acoef); bcoef = smooth(bcoef); ccoef = smooth(ccoef)

        def zg_eval(bi: int, r: float) -> float:
            return float(acoef[bi]*r*r + bcoef[bi]*r + ccoef[bi])

        def slope_eval(bi: int, r: float) -> float:
            return float(2.0*acoef[bi]*r + bcoef[bi])

        # 4) 포인트별 메트릭 산출
        metrics = []
        for x, y, z, bi, r in cand:
            zg = zg_eval(bi, r)
            h  = z - zg
            slope = abs(slope_eval(bi, r))
            # convex 검사 대비값
            if self.convex_dr > 1e-3:
                rm = max(self.min_r + 0.1, r - self.convex_dr)
                rp = min(self.shared_max_r - 0.1, r + self.convex_dr)
                h_cv = z - 0.5*(zg_eval(bi, rm) + zg_eval(bi, rp))
            else:
                h_cv = h
            metrics.append((x, y, z, r, bi, h, slope, h_cv, abs(y)))

        return hdr, metrics, vis_all_xyz

    # ─────────────────────────────────────────────────────────────────────
    # 포인트 keep 여부 (프로파일별 임계 적용)
    # met: (x,y,z,r,bi,h,slope,h_cv,yabs)
    # ─────────────────────────────────────────────────────────────────────
    def keep_point(self, met, pro: Profile) -> bool:
        x,y,z,r,bi,h,slope,h_cv,yabs = met

        # 기본 임계 + 거리 가중
        h_th = pro.gnd_clear
        if r > pro.dynamic_clear_r0:
            h_th += pro.far_clear_gain * (r - pro.dynamic_clear_r0)

        # 경사 보정
        if slope < pro.low_slope_thresh:
            h_th += pro.low_slope_extra
        if slope > pro.hi_slope_thresh:
            h_th += pro.hi_slope_extra

        # 내리막/오르막 보정
        if self.use_imu_leveling and self.pitch_deg < -pro.downhill_pitch_deg:
            h_th += pro.downhill_h_extra
        if self.use_imu_leveling and self.pitch_deg >  pro.uphill_pitch_deg:
            h_th += pro.uphill_h_extra

        # 볼록성 검사
        if (h < h_th) or (h_cv < self.convex_clear + pro.uphill_convex_extra):
            return False

        # 엣지 램프 + 높은 물체 보존
        edge_start     = pro.corridor_y - pro.edge_guard_y
        obs_edge_start = pro.corridor_y - pro.obs_shrink_y
        if pro.edge_ramp_enable and yabs >= edge_start:
            edge_pen = yabs - edge_start  # 0~edge_guard_y
            h_ramped = pro.curb_h_cap + pro.edge_ramp_gain * edge_pen
            if pro.edge_ramp_h_cap > 0.0:
                h_ramped = min(h_ramped, pro.edge_ramp_h_cap)
            if (h < h_ramped) and (h < pro.edge_preserve_h):
                return False

        # 가장자리 보조 억제(높은 물체는 통과)
        if yabs >= edge_start and h < min(pro.curb_h_cap, pro.edge_preserve_h):
            return False
        if yabs >= obs_edge_start and h < min(pro.obs_edge_h_min, pro.edge_preserve_h):
            return False

        # 프로파일별 최종 ROI
        if not (0.0 < x <= pro.ahead_m):  return False
        if r > pro.max_r:                 return False
        if yabs > pro.corridor_y:         return False
        if z > pro.z_max:                 return False
        if abs(math.degrees(math.atan2(y, x))) > pro.fov_deg: return False

        return True

    # ─────────────────────────────────────────────────────────────────────
    # voxel downsample (xy)
    # ─────────────────────────────────────────────────────────────────────
    def _voxelize(self, xy: List[Tuple[float,float]], leaf: float=0.10):
        if not xy: return []
        inv = 1.0 / max(1e-6, leaf)
        vox = {}
        for x, y in xy:
            kx = math.floor(x * inv); ky = math.floor(y * inv)
            sx, sy, c = vox.get((kx, ky), (0.0, 0.0, 0))
            vox[(kx, ky)] = (sx + x, sy + y, c + 1)
        return [(sx / c, sy / c) for (sx, sy, c) in vox.values()]

    # ─────────────────────────────────────────────────────────────────────
    # 클러스터링 + 박스 게이트 (프로파일별)
    # ─────────────────────────────────────────────────────────────────────
    def _cluster_with_profile(self, xy, header, pro: Profile):
        pa = PoseArray(); pa.header = header
        if xy is None or len(xy) == 0:
            return pa, 999.0, 999.0

        db = DBSCAN(eps=pro.db_eps, min_samples=pro.db_min_samples)
        try:
            labels = db.fit_predict(xy)
        except ValueError:
            return pa, 999.0, 999.0

        uniq = [k for k in np.unique(labels) if k != -1]
        min_edge = float('inf'); min_center = float('inf')
        for k in uniq:
            pts = xy[labels == k]; n = pts.shape[0]
            if n < pro.cluster_min_pts or n > pro.cluster_max_pts: continue

            xs = pts[:,0]; ys = pts[:,1]
            w = float(ys.max() - ys.min()); l = float(xs.max() - xs.min())
            cx = float(xs.mean());         cy = float(ys.mean())

            if not (pro.box_w_min <= w <= pro.box_w_max and pro.box_l_min <= l <= pro.box_l_max):
                continue
            if not (0.0 < cx <= pro.ahead_m):
                continue

            # 차로 안쪽 띠와 겹치기 요구 (가장자리에 붙은 얇은 것 컷)
            inner = pro.corridor_y - pro.obs_shrink_y
            if abs(cy) - 0.5*w > inner:
                continue

            if pro.curb_shape_gate_enable:
                edge_start = (pro.corridor_y - pro.obs_shrink_y) - pro.curb_edge_guard_y
                if abs(cy) >= edge_start and (w <= pro.curb_shape_w_max) and (l >= pro.curb_shape_l_min):
                    continue

            p = Pose(); p.position.x = cx; p.position.y = cy; p.position.z = 0.0
            pa.poses.append(p)

            d_edge_k   = float(np.min(np.hypot(xs, ys)))
            d_center_k = math.hypot(cx, cy)
            if d_edge_k   < min_edge:   min_edge   = d_edge_k
            if d_center_k < min_center: min_center = d_center_k

        if not pa.poses:
            return pa, 999.0, 999.0
        return pa, (min_edge if np.isfinite(min_edge) else 999.0), (min_center if np.isfinite(min_center) else 999.0)

    # ─────────────────────────────────────────────────────────────────────
    # 유틸 퍼블리시
    # ─────────────────────────────────────────────────────────────────────
    def publish_empty(self, header):
        msg = PoseArray(); msg.header = header
        self.cluster_pub.publish(msg)
        self.cluster_pub_cruise.publish(msg)
        self.cluster_pub_evasive.publish(msg)
        d = Float32MultiArray(); d.data = [999.0, 999.0]
        self.distance_pub.publish(d)

    def publish_cloud(self, pub, header, xyz_list):
        if not xyz_list: return
        try:
            from sensor_msgs.point_cloud2 import create_cloud_xyz32
            pub.publish(create_cloud_xyz32(header, xyz_list))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # 메인 콜백
    # ─────────────────────────────────────────────────────────────────────
    def callback(self, cloud_msg: PointCloud2):
        hdr, metrics, vis_all = self.preprocess_metrics(cloud_msg)
        if not metrics:
            self.publish_empty(hdr)
            return

        # 프로파일별 1차 게이트
        raw_xy_c, raw_xy_e = [], []
        vis_c, vis_e = [], []
        for met in metrics:
            if self.keep_point(met, self.pro_cruise):
                raw_xy_c.append((met[0], met[1]))
                vis_c.append((met[0], met[1], met[2]))
            if self.keep_point(met, self.pro_evasive):
                raw_xy_e.append((met[0], met[1]))
                vis_e.append((met[0], met[1], met[2]))

        # voxel
        xy_c = np.array(self._voxelize(raw_xy_c, self.voxel_leaf), dtype=float) if raw_xy_c else None
        xy_e = np.array(self._voxelize(raw_xy_e, self.voxel_leaf), dtype=float) if raw_xy_e else None

        # 클러스터링
        pa_c, min_edge_c, min_center_c = self._cluster_with_profile(xy_c, hdr, self.pro_cruise)
        pa_e, min_edge_e, min_center_e = self._cluster_with_profile(xy_e, hdr, self.pro_evasive)

        # 디버깅 토픽
        self.cluster_pub_cruise.publish(pa_c)
        self.cluster_pub_evasive.publish(pa_e)
        self.publish_cloud(self.filtered_pub_cruise, hdr, vis_c)
        self.publish_cloud(self.filtered_pub_evasive, hdr, vis_e)

        # 활성 모드 결정 및 발행
        if self._decide_mode():
            # 회피 활성
            self.cluster_pub.publish(pa_e)
            self.publish_cloud(self.filtered_pub, hdr, vis_e)
            out = Float32MultiArray(); out.data = [float(min_edge_e), float(min_center_e)]
        else:
            # 주행 활성
            self.cluster_pub.publish(pa_c)
            self.publish_cloud(self.filtered_pub, hdr, vis_c)
            out = Float32MultiArray(); out.data = [float(min_edge_c), float(min_center_c)]
        self.distance_pub.publish(out)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    LidarClusterDual()
    rospy.spin()
