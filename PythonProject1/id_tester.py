import cv2
import time
import numpy as np


def test_camera_ids():
    """
    0부터 9까지의 ID를 순회하며 카메라 연결을 시도하고,
    성공한 카메라의 ID와 스트리밍 화면을 보여줍니다.
    """

    print("\n[INFO] 웹캠 ID 테스트를 시작합니다. 'q' 키를 눌러 다음 ID로 넘어가세요.")

    found_cameras = []

    # 0번부터 9번까지의 ID를 테스트합니다.
    for i in range(10):
        # 캡처 객체 생성
        cap = cv2.VideoCapture(i)

        # 캡처 객체가 열리지 않으면 다음 ID로 넘어갑니다.
        if not cap.isOpened():
            print(f"[TEST] ID {i}: 연결 실패 (카메라 없음)")
            continue

        # 카메라가 열렸다면, 500ms 동안 스트림을 읽어 카메라를 활성화시킵니다.
        # 일부 시스템에서는 첫 프레임이 검은 화면일 수 있기 때문에 몇 프레임을 버립니다.
        time.sleep(0.5)
        ret, frame = cap.read()

        if ret:
            # 성공적으로 프레임을 읽었다면, 사용자에게 화면을 보여줍니다.
            found_cameras.append(i)
            print(f"[SUCCESS] ID {i}: 연결 성공. 창을 확인하세요.")

            while True:
                # 프레임을 다시 읽고 창에 표시
                ret, frame = cap.read()
                if not ret:
                    break

                # 화면에 현재 ID를 표시
                cv2.putText(frame, f"Camera ID: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera ID {i}", frame)

                # 'q' 키를 누르면 현재 카메라 ID 테스트 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow(f"Camera ID {i}")
                    break

        # 캡처 객체 해제
        cap.release()

    # 테스트 결과 요약 출력
    print("\n[SUMMARY] 테스트 완료된 유효 카메라 ID 목록:")
    if found_cameras:
        for id in found_cameras:
            print(f" - ID: {id}")
        print("\n이 ID들을 사용하여 'stereo_tracker_poc.py' 파일의 CAMERA_ID_LEFT 및 CAMERA_ID_RIGHT 값을 설정하세요.")
    else:
        print(" - 유효한 카메라를 찾을 수 없습니다. 카메라 연결 상태를 확인하세요.")


if __name__ == "__main__":
    test_camera_ids()
