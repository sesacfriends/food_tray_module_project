import pyrealsense2 as rs
import numpy as np
import cv2

# 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y

# 파이프라인 설정 (깊이 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림 추가
pipeline.start(config)

# 마우스 좌표 초기값 설정
mouse_x, mouse_y = -1, -1

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() # 컬러 프레임 가져오기

        if not depth_frame or not color_frame:
            continue

        # 깊이 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())  # 컬러 이미지로 변환

        # 깊이 이미지를 3채널로 변환
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 마우스 위치의 깊이 정보 가져오기 (마우스 좌표가 유효한 경우)
        if mouse_x >= 0 and mouse_y >= 0 and mouse_x < depth_image.shape[1] and mouse_y < depth_image.shape[0]:
            distance_meters = depth_frame.get_distance(mouse_x, mouse_y)
            distance_cm = distance_meters * 100
            depth_text = f"Depth at ({mouse_x}, {mouse_y}): {distance_cm:.2f} cm"


            cv2.putText(color_image, depth_text, (10, color_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # 컬러 이미지 표시 (마우스 좌표 정보 포함)  
        cv2.imshow('Color Image', color_image) # color_image로 변경
        cv2.setMouseCallback('Color Image', mouse_callback)  # 마우스 콜백 함수 설정


        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()