import pyrealsense2 as rs
import numpy as np  # 효율적인 배열 연산을 위한 numpy 임포트
import cv2  # 시각화를 위한 OpenCV 임포트 (선택적)

try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 깊이 스트림 설정

    pipeline.start(config) # 스트리밍 시작

    while True:
        frames = pipeline.wait_for_frames() # 프레임 받아오기
        depth_frame = frames.get_depth_frame() # 깊이 프레임 가져오기
        if not depth_frame: # 깊이 프레임이 없으면 다음 프레임으로
            continue

        # 깊이 데이터를 numpy 배열로 변환하여 조작 용이하게 함
        depth_image = np.asanyarray(depth_frame.get_data())

        # OpenCV를 사용한 시각화 (선택적)
        cv2.imshow("Depth Image", depth_image) # 깊이 이미지 표시
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

        # 개선된 깊이 표현
        coverage = [0] * 64
        for y in range(0, 480, 20):  # 20 단위로 행 반복
            for x in range(0, 640, 10):  # 10 단위로 열 반복
                dist = depth_frame.get_distance(x, y) # depth_frame에서 직접 거리 가져오기
                if 0 < dist < 1: # 거리가 0과 1 사이일 경우
                    coverage[x // 10] += 1 # coverage 계산

            line = ""
            for c in coverage: # coverage를 문자열로 변환
                line += " .:nhBXWW"[c // 25]
            coverage = [0] * 64
            print(line)


finally:  # 에러 발생 여부와 관계없이 리소스 해제
    pipeline.stop()  # 스트리밍 중지
    cv2.destroyAllWindows() # OpenCV 창 닫기 (사용된 경우)
    print("스트리밍 중지.")