import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정 (깊이 스트림 활성화)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 깊이 프레임을 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 이미지 중앙 픽셀의 거리 출력 (센티미터 단위)
        center_x = depth_image.shape[1] // 2
        center_y = depth_image.shape[0] // 2
        distance_meters = depth_frame.get_distance(center_x, center_y)
        distance_cm = distance_meters * 100  # 미터를 센티미터로 변환
        print(f"중앙 픽셀 거리: {distance_cm:.2f} cm")

        # 깊이 이미지를 컬러맵으로 변환하여 표시 (선택 사항)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()