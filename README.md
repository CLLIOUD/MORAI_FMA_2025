# MORAI_FMA_2025 코드베이스 개요

## 1. 패키지 전체 구조
* `morai_script`는 ROS 1 Noetic용 catkin 패키지로, `rospy`, 기본 메시지 패키지, MORAI 전용 메시지(`morai_msgs`), 커스텀 메시지(`custom_msgs`) 등에 의존하며, `scripts/` 이하의 파이썬 노드를 설치 대상으로 지정합니다.
* 제공되는 `.launch` 파일들은 센서 상태 취합과 전역 경로 배포(`first.launch`), LiDAR·카메라 기반 인지(`second.launch`), 정적 TF 브로드캐스트(`tf_fix.launch`) 등 주요 노드를 조합해 실행할 수 있도록 구성되어 있습니다.

## 2. 센싱 및 상태 추정
* `scripts/status.py`는 GPS, IMU, 차량 상태를 수신해 `/status`(커스텀 메시지)와 RViz 표시에 쓰일 `/vehicle_heading` 마커를 퍼블리시합니다. 파라미터로 입력 토픽, yaw 오프셋, 화살표 길이 등을 조정할 수 있으며, UTM 변환과 IMU 기반 yaw 계산, 글리치 가드를 모두 포함합니다.

## 3. 전역 경로 생성과 관리
* `scripts/path.py`는 텍스트 웨이포인트 파일을 찾아 불러온 뒤 거리 기반 재샘플링과 곡률/가감속 제약이 있는 속도 계획을 수행합니다. 결과는 `global_waypoints_src`(커스텀 WaypointArray), `global_path`(nav_msgs/Path), `path_speed_markers`(MarkerArray)로 발행되어 RViz 시각화까지 지원합니다.
* `scripts/path_mux.py`는 전역 경로와 라티스 경로를 상황에 맞게 선택·합성합니다. 헤딩 고정(Heading Hold) 구간, 회피 금지 존, 선행차 동적 감지, 상태 글리치 가드 등 다양한 파라미터 기반 조건을 적용해 `global_waypoints`를 최종 출력합니다.

## 4. 차량 제어
* `scripts/stanley.py`는 Stanley 조향과 PI-D 속도 제어를 결합한 핵심 주행 노드입니다. 최신 전역 웨이포인트를 지속적으로 반영하고, 신호등·정지선·경고 토픽을 통해 2단계 감속 및 정지를 처리하며, 헤딩 고정과 DWELL(정지 후 재출발) 로직, 선행차 추종 제한, 다양한 안전 가드를 포함합니다.
* 속도 제어기 내부에서는 포화·slew-rate·적분 제한을 고려한 PI-D 계산으로 가감속 명령을 생성해 `/ctrl_cmd_0`를 발행합니다.

## 5. 로컬 플래너와 장애물 인지
* `scripts/lattice_planner.py`는 전역 경로와 센서 데이터를 기반으로 회피 경로 후보를 생성·평가하는 라티스 플래너입니다. 선행차 추적 보조(`LeadTracker`), 안전 여유·재진입·채터링 억제 등 다수의 파라미터가 준비되어 있으며 `/lattice_path`와 `/speed_control`을 출력합니다.
* `scripts/lidar_cluster.py`는 LiDAR 포인트 클라우드를 처리해 이중 프로파일(주행/회피) 기반의 DBSCAN 클러스터링을 수행하고, 결과를 `/clusters`, `/cluster_distances` 등으로 퍼블리시합니다. 공통 전처리, 지면 추정, IMU 레벨링, 차로별 파라미터가 세분화되어 있습니다.

## 6. 카메라 기반 신호/정지선 감지
* `scripts/traffic_stop.py`는 YOLOv8 모델(`best.pt`)을 활용한 신호등 분류와 HSV 기반 정지선 검출을 결합합니다. 영상은 `/image_jpeg/compressed`에서 받아 `/traffic`(신호등 상태), `/stop`(정지선 감지 여부)를 발행하며, 파라미터로 ROI 비율과 HSV 임계값을 제공합니다.

## 7. 좌표계 정합
* `scripts/tf_fix.py`는 파라미터로 전달된 여러 정적 변환을 파싱해 `/tf_static`으로 브로드캐스트합니다. 리스트·딕셔너리·YAML 문자열 입력을 모두 지원하며, 중복 child 프레임 경고와 RPY→쿼터니언 변환 유틸리티를 포함합니다.

## 8. 다음 학습 권장 사항
* ROS 토픽·메시지 플로우를 실제 실행 환경에서 `rqt_graph` 등으로 시각화하면서 각 노드가 어떤 데이터에 의존하는지 익히기.
* `custom_msgs` 정의(외부 패키지)에 포함된 `waypoint`, `waypointarray`, `status` 구조를 확인해 메시지 필드의 의미와 단위를 이해하기.
* 라티스 플래너·Stanley 노드가 공유하는 파라미터 세트(YAML/launch) 작성법을 학습해 시뮬레이션과 실차에서의 튜닝 전략을 마련하기.
* YOLO 가중치 파일 경로, 센서 TF 등 환경 의존 자산을 팀 규칙으로 문서화하고, dev/실차 환경 전환 시 필요한 수정 절차를 체계화하기.
