import cv2
import numpy as np
import time

# --- 1. 환경 설정 ---
CAMERA_ID = 0  # 카메라 ID
# 🚨 [수정 방법] 카메라가 안 열리면 이 값을 1 또는 -1로 바꿔서 시도해보세요.
ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_SIZE = 0.05  # 마커 크기 (미터)

# **[중요] 카메라 캘리브레이션 데이터 (정확도 핵심)**
MTX = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
DST = np.zeros((5, 1), dtype=np.float32)

# 추적할 마커 ID와 부위 이름 정의
TRACKED_MARKERS = {
    1: "Head",  # 머리
    2: "Shoulder_L",  # 왼쪽 어깨
    3: "Shoulder_R",  # 오른쪽 어깨
    4: "Elbow_L",  # 왼쪽 팔꿈치
    5: "Elbow_R",  # 오른쪽 팔꿈치
    6: "Hand_L",  # 왼손
    7: "Hand_R",  # 오른손
}

# 관절 연결 정의 (마커 ID 기준)
SKELETON_CONNECTIONS = [
    (1, 2),  # Head -> Shoulder_L
    (1, 3),  # Head -> Shoulder_R
    (2, 3),  # Shoulder_L -> Shoulder_R
    (2, 4),  # Shoulder_L -> Elbow_L
    (4, 6),  # Elbow_L -> Hand_L
    (3, 5),  # Shoulder_R -> Elbow_R
    (5, 7),  # Elbow_R -> Hand_R
]

# 각도 측정 관절 정의 (세 점이 필요: [P1, Joint, P2])
ANGLE_JOINTS = {
    "Elbow_L_Angle": (2, 4, 6),  # Shoulder_L -> Elbow_L -> Hand_L
    "Elbow_R_Angle": (3, 5, 7),  # Shoulder_R -> Elbow_R -> Hand_R
}


# --- 2. 헬퍼 함수 ---

def get_2d_projection(point_3d, mtx, dist):
    """3D 점을 2D 이미지 좌표로 투영합니다."""
    point_3d = np.array([point_3d], dtype=np.float32)
    rVec = np.zeros((3, 1))
    tVec = np.zeros((3, 1))
    image_points, _ = cv2.projectPoints(point_3d, rVec, tVec, mtx, dist)
    return tuple(image_points[0][0].astype(int))


def calculate_angle(p1, joint, p2):
    """세 3D 점을 이용해 관절 각도를 계산합니다."""
    vector_a = p1 - joint
    vector_b = p2 - joint

    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return None

    cosine_angle = dot_product / (magnitude_a * magnitude_b)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return 180 - angle_degrees  # 굽힘 각도를 위해 180도에서 빼줍니다.


# --- 3. 초기화 ---
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 🚨 [수정]: cv2.CAP_DSHOW 플래그를 제거하고 기본 백엔드를 사용합니다.
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("------------------------------------------------------------------")
    print(f"[CRITICAL ERROR] Camera (ID {CAMERA_ID})를 열 수 없습니다. 권한 및 연결 상태를 확인하세요.")
    print("  -> 다른 프로그램이 카메라를 사용 중이 아닌지 확인하거나, CAMERA_ID를 1 또는 -1로 변경해 시도해 보세요.")
    print("------------------------------------------------------------------")
    exit()

print(f"카메라 {CAMERA_ID} 연결 성공. 3D 골격 시각화 및 각도 측정을 시작합니다.")

# --- 4. 메인 루프 ---
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # 마커 감지
    corners, ids, rejected = detector.detectMarkers(frame)

    # 감지된 마커의 3D 위치(tVec)를 저장할 딕셔너리
    marker_tVecs = {}

    if ids is not None:
        detected_ids = ids.flatten()

        for i in range(len(detected_ids)):
            marker_id = detected_ids[i]

            if marker_id not in TRACKED_MARKERS:
                continue

            corner = corners[i]

            # 자세 추정
            rVecs, tVecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, ARUCO_SIZE, MTX, DST)

            if rVecs is None or tVecs is None or len(rVecs) == 0:
                continue

            rVec = rVecs[0][0]
            tVec = tVecs[0][0]

            # 3D 위치 저장 (Numpy 배열로 저장)
            marker_tVecs[marker_id] = tVec

            # --- 5. 프레임에 결과 표시 (개별 마커) ---

            # 경고 제거를 위해 좌표축 길이 0.03m로 설정
            cv2.drawFrameAxes(frame, MTX, DST, rVec, tVec, 0.03)

            # 텍스트 표시
            distance_cm = tVec[2] * 100
            part_name = TRACKED_MARKERS[marker_id]
            text = f"[{part_name}] ID:{marker_id} | Dist:{distance_cm:.1f}cm"

            # 마커의 왼쪽 상단 좌표
            pts = corner[0].astype(np.int32)
            text_pos_x = pts[0][0]
            text_pos_y = pts[0][1] - 15

            cv2.putText(frame, text, (text_pos_x, text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ----------------------------------------------------
        # --- 6. 골격 시각화 (Skeleton Drawing) ---
        # ----------------------------------------------------
        projected_points = {}

        # 3D 마커 위치를 2D 화면 좌표로 변환 대신, 이미 감지된 마커의 2D 중심 좌표를 사용합니다.
        for mid in marker_tVecs.keys():
            # 해당 ID의 코너를 찾습니다.
            idx = np.where(detected_ids == mid)[0][0]
            corner = corners[idx]

            # 마커의 중심 좌표 (2D)를 계산합니다.
            center_x = int(np.mean(corner[0, :, 0]))
            center_y = int(np.mean(corner[0, :, 1]))

            projected_points[mid] = (center_x, center_y)

        # 연결
        for id1, id2 in SKELETON_CONNECTIONS:
            if id1 in projected_points and id2 in projected_points:
                pt1 = projected_points[id1]
                pt2 = projected_points[id2]
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # 하늘색 선으로 연결
                cv2.circle(frame, pt1, 5, (0, 255, 255), -1)  # 노란색 점으로 관절 표시
                cv2.circle(frame, pt2, 5, (0, 255, 255), -1)  # 노란색 점으로 관절 표시

        # ----------------------------------------------------
        # --- 7. 각도 측정 및 표시 ---
        # ----------------------------------------------------
        y_offset = 30  # 화면 상단에 각도 정보를 표시할 위치

        for angle_name, (id1, joint_id, id2) in ANGLE_JOINTS.items():
            if id1 in marker_tVecs and joint_id in marker_tVecs and id2 in marker_tVecs:
                p1 = marker_tVecs[id1]
                joint = marker_tVecs[joint_id]
                p2 = marker_tVecs[id2]

                angle = calculate_angle(p1, joint, p2)

                if angle is not None:
                    # 결과 텍스트 생성
                    angle_text = f"{angle_name}: {angle:.1f} degrees"

                    # 화면 좌측 상단에 표시
                    cv2.putText(frame, angle_text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)  # 보라색
                    y_offset += 30

    # 8. 화면 표시
    cv2.imshow("Multi-Marker Aruco Tracking - Press 'q' to quit", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 9. 종료 ---
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")
