import cv2
import numpy as np
import time
from pythonosc import udp_client
from scipy.spatial.transform import Rotation as R

# --- 1. 환경 설정 및 OSC 설정 ---
CAMERA_ID = 0
ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_SIZE = 0.05  # 마커 크기 (미터)

# VRChat OSC 설정
VRC_IP = "127.0.0.1"  # VRChat이 실행 중인 컴퓨터의 IP 주소
VRC_PORT = 9000  # VRChat이 OSC 데이터를 수신하는 기본 포트
client = udp_client.SimpleUDPClient(VRC_IP, VRC_PORT)

# 카메라 캘리브레이션 데이터 (🚨실제 사용 시 정확한 값을 입력해야 합니다)
MTX = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
DST = np.zeros((5, 1), dtype=np.float32)

# 추적할 마커 ID와 VRChat 트래커 이름 정의 (상반신 7점)
# 🚨[중요] 실제로 부착한 마커 ID를 아래 딕셔너리에 매핑해야 합니다.
TRACKED_MARKERS = {
    1: {"name": "Head", "vrc_slot": "head"},
    2: {"name": "Shoulder_L", "vrc_slot": "leftshoulder"},  # 왼쪽 어깨 마커
    3: {"name": "Shoulder_R", "vrc_slot": "rightshoulder"},  # 오른쪽 어깨 마커
    4: {"name": "Chest", "vrc_slot": "chest"},  # 가슴 마커
    5: {"name": "Hip", "vrc_slot": "hip"},  # 골반 마커
    6: {"name": "Hand_L", "vrc_slot": "lefthand"},
    7: {"name": "Hand_R", "vrc_slot": "righthand"},

}


# ----------------------------------------------------
# --- 2. 헬퍼 함수 ---
# ----------------------------------------------------

def rvec_to_quaternion(rvec):
    """
    OpenCV의 회전 벡터(rVec)를 VRChat이 사용하는 쿼터니언 (x, y, z, w)으로 변환합니다.
    """
    try:
        # rVecs는 (1, 3) 형태의 3x1 벡터이므로, Rodriques 함수에 맞게 rvec[0]이 아닌 rvec 자체를 전달합니다.
        R_matrix, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(R_matrix)
        quaternion = r.as_quat()  # (x, y, z, w) 순서로 반환
        return quaternion
    except Exception as e:
        # print(f"쿼터니언 변환 오류: {e}")
        return np.array([0.0, 0.0, 0.0, 1.0])


def send_osc_data(vrc_slot, tvec, rvec, client):
    """
    VRChat OSC 형식으로 위치 및 회전 데이터를 전송합니다.
    """
    # 1. 위치(Position) 전송
    client.send_message(f"/tracking/tracker/{vrc_slot}/position",
                        [float(tvec[0]), float(tvec[1]), float(tvec[2])])

    # 2. 회전(Rotation) 전송
    quaternion = rvec_to_quaternion(rvec)
    client.send_message(f"/tracking/tracker/{vrc_slot}/rotation",
                        [float(q) for q in quaternion])


# --- 3. 초기화 ---
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("------------------------------------------------------------------")
    print(f"[CRITICAL ERROR] Camera (ID {CAMERA_ID})를 열 수 없습니다.")
    print("------------------------------------------------------------------")
    exit()

print(f"카메라 {CAMERA_ID} 연결 성공. VRChat 상반신 7점 OSC 데이터 전송을 시작합니다.")
print(f"VRChat IP: {VRC_IP}, Port: {VRC_PORT}")
print("--- 트래킹 대상: Head(1), Chest(4), Hip(8), L/R Shoulder(2/3), L/R Hand(6/7) ---")
print("------------------------------------------------------------------")

# --- 4. 메인 루프 ---
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # 마커 감지
    corners, ids, rejected = detector.detectMarkers(frame)

    # 감지된 마커의 3D 위치 및 회전을 저장할 딕셔너리
    marker_data = {}

    if ids is not None:
        detected_ids = ids.flatten()
        y_offset = 30  # 디버깅 텍스트 시작 위치

        for i in range(len(detected_ids)):
            marker_id = detected_ids[i]

            # 1. 추적 대상에 포함되는 마커만 처리
            if marker_id in TRACKED_MARKERS:
                corner = corners[i]

                # 자세 추정
                rVecs, tVecs_single, _ = cv2.aruco.estimatePoseSingleMarkers(corner, ARUCO_SIZE, MTX, DST)

                if rVecs is None or tVecs_single is None or len(rVecs) == 0:
                    continue

                rVec = rVecs[0][0]
                tVec = tVecs_single[0][0]

                # --- 2. OSC 전송 ---
                vrc_slot = TRACKED_MARKERS[marker_id]["vrc_slot"]
                send_osc_data(vrc_slot, tVec, rVec, client)

                # --- 3. 프레임에 결과 표시 ---
                cv2.drawFrameAxes(frame, MTX, DST, rVec, tVec, 0.03)
                distance_cm = tVec[2] * 100
                part_name = TRACKED_MARKERS[marker_id]["name"]

                # 마커 자체에 정보 표시
                pts = corner[0].astype(np.int32)
                text = f"ID:{marker_id} [{part_name}] | Dist:{distance_cm:.1f}cm"
                cv2.putText(frame, text, (pts[0][0], pts[0][1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 좌측 상단에 OSC 전송 상태 표시
                info_text = f"OSC SEND: {vrc_slot} | Pos: ({tVec[0]:.2f}, {tVec[1]:.2f}, {tVec[2]:.2f})"
                cv2.putText(frame, info_text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20

        if y_offset == 30:  # 아무 마커도 감지되지 않은 경우
            cv2.putText(frame, "No Aruco Markers detected or tracked.", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 5. 화면 표시
    cv2.imshow("VRChat Upper Body 7-Point Aruco Tracker - OSC Active", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. 종료 ---
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")