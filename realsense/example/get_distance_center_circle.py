import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정 (깊이 및 컬러 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 깊이 스트림
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 컬러 스트림
pipeline.start(config)


try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() # 컬러 프레임 가져오기

        if not depth_frame or not color_frame: # 둘 중 하나라도 없으면 continue
            continue

        # 깊이 프레임을 NumPy 배열로 변환 (필요한 경우 사용)
        depth_image = np.asanyarray(depth_frame.get_data())

        # 컬러 프레임을 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())


        # 이미지 중앙 픽셀의 거리 출력 (필요한 경우 사용)
        center_x = depth_image.shape[1] // 2
        center_y = depth_image.shape[0] // 2

        point = (center_x, center_y)


        distance_meters = depth_frame.get_distance(center_x, center_y)
        distance_cm = distance_meters * 100
        print(f"중앙 픽셀 거리: {distance_cm:.2f} cm")

        # 컬러 이미지에 원하는 점 표시 (빨간색 원)
        cv2.circle(color_image, point, 3, (0,0,255), -1)


        # 컬러 이미지 표시
        cv2.imshow('Color Image', color_image)


        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()