import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Detect 모델 로드
model = YOLO('best.pt')  # Detect 모델 경로를 설정

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 파이프라인 생성
config = rs.config()  # RealSense 카메라 설정 객체 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 깊이 스트림 활성화
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림 활성화
pipeline.start(config)  # 파이프라인 시작

try:
    while True:  # 실시간 데이터 처리
        frames = pipeline.wait_for_frames()  # 프레임 가져오기
        depth_frame = frames.get_depth_frame()  # 깊이 프레임 가져오기
        color_frame = frames.get_color_frame()  # 컬러 프레임 가져오기

        # 유효하지 않은 프레임 건너뛰기
        if not depth_frame or not color_frame:
            continue

        # NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO Detect 모델 실행
        results = model(color_image)  # YOLO 모델을 사용하여 객체 감지 수행

        # 객체 탐지 결과 처리
        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()  # Bounding Box 좌표 (x, y, w, h)
            classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

            for i, box in enumerate(boxes):
                x, y, w, h = box  # Bounding Box 정보
                center_x, center_y = int(x), int(y)  # 중심점 좌표
                depth = depth_image[center_y, center_x]  # 중심점의 깊이 값 추출

                # 유효한 깊이 값 확인
                if depth <= 0:
                    continue

                # 결과 출력
                print(f"Object {i+1} - Class: {model.names[int(classes[i])]} - Depth: {depth:.3f} mm")

                # 경계 상자와 깊이 정보 시각화
                cv2.rectangle(color_image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                cv2.putText(color_image, f"Depth: {depth:.2f}mm",
                            (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 시각화
        cv2.imshow("YOLO Detect Results", color_image)  # 컬러 이미지 시각화
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()  # RealSense 파이프라인 중지
    cv2.destroyAllWindows()  # OpenCV 창 닫기
