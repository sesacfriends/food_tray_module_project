import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json

# def get_json():
#     json_path = './sample/01_011_01011001_160273207049000.json'
#     with open(json_path) as fj:
#         data = json.load(fj)




# 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y


# 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 마우스 좌표 초기값 설정
mouse_x, mouse_y = -1, -1

# 관심 영역 (ROI) 좌표 설정 (polygon 형식)
roi_pts = np.array([[200, 120], [285, 120], [285, 215], [200, 215]], np.int32)  # 예시 polygon 좌표
roi_pts = roi_pts.reshape((-1, 1, 2))  # cv2.polylines() 함수에 필요한 형태로 변환

# 시간 측정 변수 초기화
last_print_time = time.time()
print_interval = 2  # 출력 간격 (초)


try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 깊이 및 컬러 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 마우스 위치의 깊이 정보 가져오기 (마우스 좌표가 유효한 경우)
        if mouse_x >= 0 and mouse_y >= 0 and mouse_x < depth_image.shape[1] and mouse_y < depth_image.shape[0]:
            distance_meters = depth_frame.get_distance(mouse_x, mouse_y)
            
            distance_cm = distance_meters * 100
            depth_text = f"Depth at ({mouse_x}, {mouse_y}): {distance_cm:.2f} cm"

             # color_image에 텍스트 표시 (녹색)
            cv2.putText(color_image, depth_text, (10, color_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # depth_image에 텍스트 표시 (흰색 - 255)
            # cv2.putText(depth_image, depth_text, (10, depth_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1) # 흰색
            cv2.putText(depth_image, depth_text, (10, depth_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (65535,), 1) # 16비트 흰색

            # cv2.polylines(depth_image, [roi_pts], True, (255,), 2)

        # 마우스 위치의 깊이 정보 가져오기 (마우스 좌표가 유효한 경우)
        # if mouse_x >= 0 and mouse_y >= 0 and mouse_x < depth_image.shape[1] and mouse_y < depth_image.shape[0]:
        #     distance_cm = distance_meters * 100
        #     depth_text = f"Depth at ({mouse_x}, {mouse_y}): {distance_cm:.2f} cm"



        # 컬러 이미지에 polygon ROI 그리기
        cv2.polylines(color_image, [roi_pts], True, (0, 255, 0), 2)
        cv2.polylines(depth_image, [roi_pts], True, (0, 0, 255), 2)

        # ROI 내부의 픽셀만 마스킹하여 거리 계산
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts], 255)  # ROI 내부를 흰색(255)로 채움

        masked_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        # 마스크된 깊이 이미지에서 거리 계산 (여기서는 평균)
        roi_depth = masked_depth_image[masked_depth_image > 0]  # 0이 아닌 유효한 거리 값만 선택

        # 현재시간 가져오기
        current_time = time.time()

        # 출력 간격이 지났는지 확인
        if current_time - last_print_time >= print_interval:
            if roi_depth.size > 0:  # ROI에 유효한 깊이 값이 있는지 확인
                average_distance_mm = np.mean(roi_depth)
                average_distance_cm = average_distance_mm / 10
                average_distance_m = average_distance_cm / 100
                # print(f"관심 영역 평균 거리: {average_distance_mm:.2f} mm")
                print(f"관심 영역 평균 거리: {average_distance_cm:.2f} cm")
                # print(f"관심 영역 평균 거리: {average_distance_m:.2f} m")

            last_print_time = current_time

        # 컬러 이미지 표시 (ROI 포함)
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image)
        cv2.setMouseCallback('Color Image', mouse_callback)  # 마우스 콜백 함수 설정
        cv2.setMouseCallback('Depth Image', mouse_callback)  # 마우스 콜백 함수 설정

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()