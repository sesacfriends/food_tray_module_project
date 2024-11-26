import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
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

        # 미터 단위 거리 값을 센티미터 단위로 변환
        depth_image_cm = depth_image * 100.0


        # matplotlib을 사용하여 깊이 이미지를 표시
        plt.imshow(depth_image_cm, cmap='jet')  # 컬러맵 적용 (viridis, plasma, inferno 등 다양한 컬러맵 사용 가능)
        plt.colorbar(label='Distance (cm)')  # 컬러바 추가
        plt.title('Depth Image (cm)')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.pause(0.01)  # 이미지 업데이트
        plt.clf()  # 이전 프레임 삭제


        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

finally:
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows()
    plt.close()  # matplotlib 창 닫기