
# 카메라 - 물체(점 1개. 지정된 위치) 간의 거리를 구하는 코드
# 사진을 촬영 후 사진에 표시된 지점의 거리를 구함
    #   

import cv2
import pyrealsense2
from realsense_depth import *

# 카메라 초기화
dc = DepthCamera()
ret, depth_frame, color_frame = dc.get_frame()

# 한 점을 지정하여 그 지점에 보이는 피사체의 거리 재기
point = (300, 300)
cv2.circle(color_frame, point, 4, (0, 0, 255))
distance = depth_frame[point[1], point[0]]

# 거리 단위: mm
print(distance)

cv2.imshow("depth frame", depth_frame)
cv2.imshow("Color frame", color_frame)
cv2.waitKey(0)
