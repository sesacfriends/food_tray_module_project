
# 카메라 - 물체(점 1개. 지정된 위치) 간의 거리를 구하는 코드
# 실시간으로 사진에 표시된 지점의 거리를 구함

import cv2
import pyrealsense2
from realsense_depth import *

# 카메라 초기화
dc = DepthCamera()

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # 한 점을 지정하여 그 지점에 보이는 피사체의 거리 재기
    point = (400, 300)
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]

    # 거리 단위: mm
    print(distance)

    # window화면에 보여주기
    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    cv2.waitKey(1)

    # waitKey(0) > 내가 키를 누를때마다 다음 프레임으로 넘어감. 누르기 전까지는 무한정 대기
    # waitKey(1) > 실시간 화면 업데이트

    # ESC키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 모든 창 닫기
cv2.destroyAllWindows()