import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def put_text_korean(img, text, pos, font_path, font_size, color, thickness):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color)
    img = np.array(img_pil)
    return img

# 웹캠 실행 (0번: 기본 웹캠)
cap = cv2.VideoCapture(0)

# 한글 폰트 경로 (시스템에 설치된 폰트 사용)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows (맑은 고딕)
#font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" # Linux (나눔고딕) - 설치 필요

# 텍스트, 위치, 폰트 크기, 색상, 두께 설정
text = "안녕하세요"
pos = (50, 50)
font_size = 30
color = (0, 0, 255)  # BGR (파란색)
#color = (255, 255, 255) # 흰색
thickness = 2

while True:
    ret, frame = cap.read()
    if not ret: # 프레임 읽기 실패 > 루프 종료
        break

    # 한글 텍스트 추가
    frame = put_text_korean(frame, text, pos, font_path, font_size, color, thickness)

    # 프레임 출력
    cv2.imshow("Webcam with Korean Text", frame)

    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()