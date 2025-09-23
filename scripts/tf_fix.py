#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped

def q_from_rpy(roll, pitch, yaw):
    """roll=x, pitch=y, yaw=z (rad) -> quaternion (x,y,z,w)"""
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * sy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw

def load_transforms_from_param():
    raw = rospy.get_param("~transforms", [])
    # 허용 형태: list(dict), dict(단일), str(YAML)
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, str):
        try:
            items = yaml.safe_load(raw)
            if isinstance(items, dict):
                items = [items]
            elif not isinstance(items, list):
                rospy.logerr("~transforms YAML 파싱 결과가 list/dict가 아님: %r", type(items))
                return []
        except Exception as e:
            rospy.logerr("~transforms 문자열 YAML 파싱 실패: %s", e)
            return []
    else:
        rospy.logerr("~transforms 타입이 지원되지 않음: %r", type(raw))
        return []

    out = []
    for i, t in enumerate(items):
        if not isinstance(t, dict):
            rospy.logwarn("transforms[%d] 건너뜀: dict 아님 (%r)", i, type(t))
            continue

        parent = str(t.get("parent", "base_link"))
        child  = str(t.get("child",  f"child_{i}"))
        xyz    = t.get("xyz",    [0.0, 0.0, 0.0])
        rpy_deg = t.get("rpy_deg", None)
        rpy_rad = t.get("rpy_rad", None)

        try:
            tx, ty, tz = float(xyz[0]), float(xyz[1]), float(xyz[2])
            if rpy_rad is not None:
                rr, pr, yr = float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])
            else:
                if rpy_deg is None:
                    rpy_deg = [0.0, 0.0, 0.0]
                rr = math.radians(float(rpy_deg[0]))
                pr = math.radians(float(rpy_deg[1]))
                yr = math.radians(float(rpy_deg[2]))
        except Exception as e:
            rospy.logwarn("transforms[%d] 파싱 실패 → 건너뜀: %s", i, e)
            continue

        out.append((parent, child, tx, ty, tz, rr, pr, yr))
    return out

def main():
    rospy.init_node("multi_static_tf", anonymous=False)

    tf_list = load_transforms_from_param()
    if not tf_list:
        rospy.logwarn("유효한 변환이 없습니다(~transforms 비었거나 파싱 실패).")
    else:
        # 중복 child 경고
        seen = set()
        for _, child, *_ in tf_list:
            if child in seen:
                rospy.logwarn("중복 child_frame_id '%s' 발견. 마지막 값으로 덮어씁니다.", child)
            seen.add(child)

        br = tf2_ros.StaticTransformBroadcaster()
        msgs = []
        stamp = rospy.Time.now()
        for parent, child, tx, ty, tz, rr, pr, yr in tf_list:
            qx, qy, qz, qw = q_from_rpy(rr, pr, yr)
            m = TransformStamped()
            m.header.stamp = stamp
            m.header.frame_id = parent
            m.child_frame_id  = child
            m.transform.translation.x = tx
            m.transform.translation.y = ty
            m.transform.translation.z = tz
            m.transform.rotation.x = qx
            m.transform.rotation.y = qy
            m.transform.rotation.z = qz
            m.transform.rotation.w = qw
            msgs.append(m)

        br.sendTransform(msgs)
        rospy.loginfo("Published %d static transforms to /tf_static.", len(msgs))

    rospy.spin()

if __name__ == "__main__":
    main()

