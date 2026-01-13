import cv2
import numpy as np
import time

# --- 1. 환경 설정 및 마커 정의 ---

# 카메라 ID (일반적으로 0 또는 -1)
CAMERA_ID = 0
# Aruco 딕셔너리 및 파라미터
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# Aruco 파라미터 최적화 (인식률 향상 목적)
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()

# --- Aruco 인식률 향상을 위한 파라미터 조정 ---
# Adaptive Thresholding 윈도우 크기 조정 (다양한 조명 환경 대응)
ARUCO_PARAMETERS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMETERS.adaptiveThreshWinSizeMax = 23
ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 10
# 적응형 이진화 상수 조정 (약간 낮춤)
ARUCO_PARAMETERS.adaptiveThreshConstant = 7
# 마커 코너 정제 (정확도 향상: Subpixel 정제 사용)
ARUCO_PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# 마커의 실제 크기 (미터 단위)
ARUCO_SIZE = 0.05  # 50mm 마커

# 카메라 캘리브레이션 데이터 (실제 값으로 대체해야 함)
# MTX: 카메라 행렬, DST: 왜곡 계수
MTX = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], dtype=np.float32)
DST = np.zeros((5, 1), dtype=np.float32)

# 추적할 마커 ID와 부위 정의
TRACKED_MARKERS = {
    0: "Head",
    1: "Shoulder_L",
    2: "Shoulder_R",
}


# --- 2. 3D 칼만 필터 클래스 정의 ---
class KalmanFilter3D:
    """
    3차원 위치 (tVec) 또는 3차원 회전 (rVec) 추적을 위한 칼만 필터
    """

    def __init__(self, process_noise=1e-2, measure_noise=1e-1):
        # 6차원 상태 벡터: [x, y, z, vx, vy, vz] (위치와 속도)
        self.kf = cv2.KalmanFilter(6, 3)

        # 상태 전이 행렬 (F): dt를 1로 가정
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        # 측정 행렬 (H): 위치만 측정 가능 [x, y, z]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measure_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

        # 예측 안정화 로직을 위한 카운터 추가: 감지 실패 횟수
        self.missing_count = 0

    def predict(self):
        # 다음 상태 예측
        predicted = self.kf.predict()
        self.missing_count += 1  # 예측이 호출될 때마다 카운트 증가 (감지 실패 의미)
        return predicted[:3].flatten()

    def update(self, measurement):
        # 측정값 업데이트
        measurement_np = np.array(measurement, dtype=np.float32).reshape(3, 1)
        corrected = self.kf.correct(measurement_np)
        self.missing_count = 0  # 감지 성공 시 카운트 리셋
        return corrected[:3].flatten()

    def reset_state(self, measurement):
        # 드리프트 보정 시뮬레이션을 위한 상태 초기화
        self.kf.statePost[:3] = measurement.reshape(3, 1)  # 위치를 측정값으로
        self.kf.statePost[3:] = 0.0  # 속도는 0으로 리셋
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0  # 불확실성 리셋
        self.missing_count = 0  # 리셋 시 카운트 리셋


# --- 3. 트래킹 시스템 초기화 ---
tVec_filters = {}  # 위치(tVec) 보정용
rVec_filters = {}  # 회전(rVec) 보정용 (IMU 안정화 역할 시뮬레이션)

# 예측 중 튀는 현상을 막기 위한 최대 허용 예측 프레임 수
MAX_PREDICTION_FRAMES = 10  # 10프레임 이상 끊기면 예측 중단

# 드리프트 보정을 위한 카운터
drift_correction_counter = 0
DRIFT_CORRECTION_INTERVAL = 150  # 150프레임마다 드리프트 보정 (약 5초 주기)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("FATAL ERROR: 웹캠을 열 수 없습니다. CAMERA_ID를 확인하세요.")
    exit()

# --- 4. 메인 루프 ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()

        # Aruco 마커 감지
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

        detected_ids = ids.flatten().tolist() if ids is not None else []

        # --- 드리프트 보정 카운터 증가 및 실행 ---
        drift_correction_counter += 1
        is_drift_correction_frame = (drift_correction_counter >= DRIFT_CORRECTION_INTERVAL)
        if is_drift_correction_frame:
            cv2.putText(output_frame, "DRIFT CORRECTION (RESET)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        2)
            drift_correction_counter = 0

        # 4.1. 감지된 마커 처리 (Update)
        if ids is not None:
            # 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_SIZE, MTX, DST)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                tvec_measured = tvecs[i][0]
                rvec_measured = rvecs[i][0]

                if marker_id not in TRACKED_MARKERS:
                    continue

                    # 필터 초기화
                if marker_id not in tVec_filters:
                    # tVec (위치): 노이즈 중간, 측정 신뢰도 낮게
                    tVec_filters[marker_id] = KalmanFilter3D(process_noise=1e-2, measure_noise=5e-1)
                    tVec_filters[marker_id].reset_state(tvec_measured)

                    # rVec (회전, IMU 안정화): 노이즈 매우 낮음, 측정 신뢰도 높게
                    rVec_filters[marker_id] = KalmanFilter3D(process_noise=1e-3, measure_noise=1e-1)
                    rVec_filters[marker_id].reset_state(rvec_measured)

                # --- 드리프트 보정 실행 (강제 리셋) ---
                if is_drift_correction_frame:
                    tVec_filters[marker_id].reset_state(tvec_measured)
                    rVec_filters[marker_id].reset_state(rvec_measured)

                # 칼만 필터 업데이트 (Update)
                tvec_corrected = tVec_filters[marker_id].update(tvec_measured)
                rvec_corrected = rVec_filters[marker_id].update(rvec_measured)

                # 시각화
                cv2.drawFrameAxes(output_frame, MTX, DST, rvec_corrected, tvec_corrected, ARUCO_SIZE * 1.5, thickness=2)

                # 마커 그리기
                cv2.aruco.drawDetectedMarkers(output_frame, [corners[i]], ids[i], borderColor=(0, 255, 0))

                pos_text = f"{TRACKED_MARKERS[marker_id]} X:{tvec_corrected[0]:.2f} Y:{tvec_corrected[1]:.2f} Z:{tvec_corrected[2]:.2f}"
                # 텍스트가 잘 안 보일 경우 배경 박스 추가
                (w, h), _ = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                box_pt1 = (corners[i][0][0][0].astype(int) - 5, corners[i][0][0][1].astype(int) - 25)
                box_pt2 = (box_pt1[0] + w + 10, box_pt1[1] + h + 10)

                cv2.rectangle(output_frame, box_pt1, box_pt2, (0, 0, 0), cv2.FILLED)  # 검은색 배경
                cv2.putText(output_frame, pos_text,
                            (corners[i][0][0][0].astype(int), corners[i][0][0][1].astype(int) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # 노란색 텍스트

        # 4.2. 마커 끊김 보정 (Predict)
        for marker_id in list(tVec_filters.keys()):
            # 감지된 마커가 아니면서 추적 대상 마커인 경우
            if marker_id not in detected_ids and marker_id in TRACKED_MARKERS:

                # 예측 안정화 로직: 최대 예측 프레임 제한
                if tVec_filters[marker_id].missing_count < MAX_PREDICTION_FRAMES:
                    # 칼만 필터 예측 (Predict)
                    tvec_predicted = tVec_filters[marker_id].predict()
                    rvec_predicted = rVec_filters[marker_id].predict()

                    # 예측된 위치의 화면 좌표를 얻기 위해 (가상 투영)
                    marker_center_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

                    # tVec의 크기가 너무 커지면 projectPoints가 실패하여 튀는 현상이 발생할 수 있으므로 예외 처리
                    try:
                        img_points, _ = cv2.projectPoints(marker_center_3d, rvec_predicted, tvec_predicted, MTX, DST)

                        center_x = int(img_points[0][0][0])
                        center_y = int(img_points[0][0][1])

                        # 예측된 포즈로 가상의 축 그리기 (빨간색 - 예측 중)
                        cv2.drawFrameAxes(output_frame, MTX, DST, rvec_predicted, tvec_predicted, ARUCO_SIZE * 1.5,
                                          thickness=2)

                        # 예측 중 텍스트 표시
                        cv2.putText(output_frame,
                                    f"{TRACKED_MARKERS[marker_id]} PREDICTING ({tVec_filters[marker_id].missing_count})",
                                    (center_x - 70, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    except cv2.error as e:
                        # projectPoints 오류가 발생하면 (주로 tVec이 극단적으로 튀었을 때) 예측 중지 및 상태 리셋
                        print(f"Prediction error for marker {marker_id}: {e}. State reset.")
                        tVec_filters[marker_id].reset_state(np.zeros(3, dtype=np.float32))
                        rVec_filters[marker_id].reset_state(np.zeros(3, dtype=np.float32))

                else:
                    # MAX_PREDICTION_FRAMES를 초과하면 예측 중지 (튀는 선 방지)
                    cv2.putText(output_frame, f"{TRACKED_MARKERS[marker_id]} LOST", (20, 50 + marker_id * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # 예측 중단 시 칼만 필터의 누적된 오차를 줄이기 위해 속도를 0으로 리셋
                    tVec_filters[marker_id].kf.statePost[3:] = 0.0
                    rVec_filters[marker_id].kf.statePost[3:] = 0.0

        # 결과 출력
        cv2.imshow("Aruco Tracking with Full Stabilization", output_frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # 캡처 해제 및 윈도우 닫기
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")