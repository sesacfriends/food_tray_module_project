import cv2
import numpy as np

def putTextCenter(img, text, center, fontFace, fontScale, color, thickness):
    """
    이미지의 중앙에 텍스트를 그립니다.

    Args:
        img (numpy.ndarray): 텍스트를 그릴 이미지.
        text (str): 그릴 텍스트.
        center (tuple): 텍스트 중심의 (x, y) 좌표.
        fontFace (int): cv2 폰트 스타일 (예: cv2.FONT_HERSHEY_SIMPLEX).
        fontScale (float): 폰트 크기 배율.
        color (tuple): 텍스트 색상 (BGR 형식).
        thickness (int): 텍스트 두께.
    """
    # 텍스트의 크기 (너비, 높이)와 baseline을 가져옴
        # putText는 텍스트의 왼쪽 아래를 기준으로 그림
        # 중앙에 위치시키려면 아래와 같이 좌표 계산
    text_size, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    text_origin = (int(center[0] - text_size[0] / 2), int(center[1] + text_size[1] / 2))    # 텍스트의 왼쪽 아래 좌표 계산
    cv2.putText(img, text, text_origin, fontFace, fontScale, color, thickness)


# 웹캠 실행 (0번: 기본 웹캠)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret: # 프레임 읽기 실패 > 루프 종료
        break

    # 텍스트, 중심 좌표, 폰트 설정
        # 프레임 크기에 맞춰 계산
    text1 = "Hello, Center!"
    text2 = "Good Morning, I am Top~"
    text3 = "Good Evening, My name is Bottom"
    
    # 글자 위치
    center = (frame.shape[1] // 2, frame.shape[0] // 2)  # 프레임의 중심
    ctop = (frame.shape[1] // 2, frame.shape[0] // 6)  # 윗 부분
    cbottom = (frame.shape[1] // 2, frame.shape[0] * 5 // 6 )  # 아랫 부분
    
    # 글자 속성
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)  # Blue
    thickness = 2

    # 글자 뿌리기
    putTextCenter(frame, text1, center, fontFace, fontScale, color, thickness)
    putTextCenter(frame, text2, ctop, fontFace, fontScale, color, thickness)
    putTextCenter(frame, text3, cbottom, fontFace, fontScale, color, thickness)

    # 결과 표시
    cv2.imshow("Webcam with Centered Text", frame)

    if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()