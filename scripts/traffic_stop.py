#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO  # YOLOv8 전용

class CombinedDetectorNode:
    def __init__(self):
        # 노드 초기화
        rospy.init_node("yolo_node")

        # --- [1] 신호등 인식 설정 (YOLOv8) ---  # ← 그대로 유지
        self.weights = "/home/choi/catkin_ws/src/morai_script/scripts/best.pt"
        self.conf_thres = 0.7
        self.imgsz = 512
        self.max_det = 3

        # YOLOv8 모델 로드
        self.model = YOLO(self.weights)
        self.names = self.model.names
        rospy.loginfo("[YOLOv8] weights=%s", self.weights)

        # --- [정지선 HSV 파라미터] ---
        self.roi_top_ratio = rospy.get_param("~stop_roi_top_ratio", 0.60)  # 하단 40% 사용
        self.hsv_s_max     = rospy.get_param("~stop_hsv_s_max", 50)        # S ≤ 60 (저채도)
        self.hsv_v_min     = rospy.get_param("~stop_hsv_v_min", 200)       # V ≥ 200 (고밝기)

        # --- Publisher ---
        self.pub_traffic = rospy.Publisher("/traffic", String, queue_size=1)
        self.pub_stop    = rospy.Publisher("/stop", String, queue_size=1)

        # --- Subscriber ---
        self.sub_image = rospy.Subscriber(
            "/image_jpeg/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**22
        )
        rospy.loginfo("🚦 Combined Detector (Traffic Light + Stop Line) started.")

    def cls_to_status(self, label: str) -> str:
        if label == "red_light":
            return "RED"
        if label == "green_light":
            return "GREEN"
        if label == "yellow_light":
            return "YELLOW"
        return "UNKNOWN"

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn("Image decode error: %s", e)
            return

        # --- [1] 신호등 인식 로직 (그대로) ---
        self.process_traffic_lights(img)
        # --- [2] 정지선 인식 로직 (HSV만) ---
        self.process_stop_line(img)

    def process_traffic_lights(self, img):
        img_center_x = img.shape[1] // 2
        status = "UNKNOWN"

        results = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=0.45,
            max_det=self.max_det,
            verbose=False
        )
        if len(results):
            res = results[0]
            closest, min_d = None, 1e9
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    c = int(cls[i])
                    label = self.names[c]
                    if label not in ("red_light", "yellow_light", "green_light"):
                        continue
                    center_x = (x1 + x2) // 2
                    d = abs(center_x - img_center_x)
                    if d < min_d and conf[i] >= self.conf_thres:
                        min_d = d
                        closest = label
            if closest:
                status = self.cls_to_status(closest)

        self.pub_traffic.publish(String(data=status))

    def process_stop_line(self, img):
        """HSV(저채도+고밝기)만 이용한 간단 정지선 검출."""
        h, w = img.shape[:2]
        y0 = int(h * self.roi_top_ratio)
        roi = img[y0:, :]  # 하단 ROI

        # 1) HSV로 저채도+고밝기 게이팅 (흰색 계열 통과)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, self.hsv_v_min), (179, self.hsv_s_max, 255))  # S<=, V>=

        # 2) (선택) 간단한 형태학으로 잡노이즈 정리 — HSV만 요구여서 최소한만 사용
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask_clean = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3) 에지 + HoughLinesP
        edges = cv2.Canny(mask_clean, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=80,
            minLineLength=80, maxLineGap=10
        )

        msg_out = String()
        msg_out.data = "TRUE" if lines is not None else "FALSE"
        self.pub_stop.publish(msg_out)


if __name__ == "__main__":
    try:
        CombinedDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

