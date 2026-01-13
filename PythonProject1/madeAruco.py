import cv2
import numpy as np
import os

# --- 1. 마커 생성 설정 ---

# 사용할 마커 딕셔너리 (코드에서 사용했던 것과 일치해야 합니다)
# 6x6 격자에 250개의 마커가 있는 딕셔너리
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# 생성할 마커의 ID
# 0부터 249 사이의 숫자를 입력하세요. (코드에서 13번을 사용했습니다.)
MARKER_ID = 7

# 최종 이미지 크기 (픽셀 단위). 클수록 인쇄 시 고화질입니다.
OUTPUT_SIZE_PIXELS = 700

# 출력 파일 경로 및 이름
OUTPUT_FILENAME = f"aruco_marker_id_{MARKER_ID}.png"


# --- 2. 마커 생성 및 저장 함수 ---
def generate_and_save_marker(dictionary_name, marker_id, size_pixels, filename):
    try:
        # ArUco 딕셔너리 로드
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_name)

        # 마커 이미지 생성
        # borderBits=1은 바깥쪽 테두리의 두께를 1비트(검은색 픽셀)로 설정합니다.
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            size_pixels
        )

        # 이미지를 파일로 저장
        cv2.imwrite(filename, marker_image)

        print(f"--- 마커 생성 완료 ---")
        print(f"파일 이름: {os.path.abspath(filename)}")
        print(f"마커 ID: {marker_id}")
        print(f"마커 딕셔너리: {dictionary_name}")
        print(f"이미지 크기: {size_pixels}x{size_pixels} 픽셀")
        print("\n이 파일을 인쇄하여 사용하시면 됩니다.")

    except Exception as e:
        print(f"[ERROR] 마커 생성 중 오류 발생: {e}")


# --- 3. 실행 ---
if __name__ == "__main__":
    generate_and_save_marker(
        ARUCO_DICT,
        MARKER_ID,
        OUTPUT_SIZE_PIXELS,
        OUTPUT_FILENAME
    )
