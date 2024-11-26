## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams

# 파이프라인 생성
# 파이프라인 구축
pipeline = rs.pipeline()
config = rs.config()


# Get device product line for setting a supporting resolution
# 카메라 정보 확인
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 스트림 설정
# 깊이와 색상 스트림 활성화, 해상도, 형식, 프레임 속도 설정
# 640*480 해상도 / 30fps설정
# rs.format.z16 : 16비트 깊이 데이터
# rs.format.bgr8 : 8비트 BGR컬러 데이터
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트리밍 시작
profile = pipeline.start(config)


# 깊이 스케일 정보 가져오기 -> 깊이 값(16비트 정수)을 실제 거리(미터)로 변환하기 위한 스케일 factor를 가져옴
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)


# 배경 제거 거리 설정
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.

# 정렬 객체 생성
# align_to : 정렬 기준이 되는 스트림을 지정
align_to = rs.stream.color
# 깊이 프레임을 컬러 프레임에 정렬
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # 깊이 프레임을 컬러 프레임에 정렬
        # 깊이 프레임을 컬러 프레임의 perspective에 맞춰 정렬
        # 깊이 정보와 컬러 정보가 일치
        aligned_frames = align.process(frames)

        # 정렬된 프레임 가져오기
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # 프레임 유효성 검사
        if not aligned_depth_frame or not color_frame:
            continue

        # numpy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 배경제거 - Set pixels further than clipping_distance to grey
        grey_color = 153
        # 깊이 이미지를 3채널로 변환
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 
        
        # 깊이 맵 생성
        # 특정범위 이상과 이하는 배경 날리기
        # 배경 회색으로 처리
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # depth image 배경 안날린 것
        bg_removed_none = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), color_image, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # 깊이 맵 생성
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 이미지 합치기 및 표시
        # 원본 컬러 이미지 + 깊이 맵
        images = np.hstack((bg_removed_none, depth_colormap))
        # images = np.hstack((depth_image_3d, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        # 키 입력 대기
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
