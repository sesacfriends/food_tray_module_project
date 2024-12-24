#pip install opencv-contrib-python

import cv2
import os

# 얼굴 인식을 위한 Haar Cascade 분류기 로드
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 이미지에서 얼굴 영역을 추출하는 함수
def face_extractor(img):
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 회색조 이미지에서 얼굴 검출
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # scaleFactor=1.3, minNeighbors=5

    # 얼굴이 검출되지 않으면 None 반환
    if len(faces) == 0:
        return None

    return [img[y:y+h, x:x+w] for (x, y, w, h) in faces] # 모든 얼굴 영역을 리스트로 반환


# 훈련 이미지를 저장할 경로 생성
train_path = "C:/Github/50_project/food_tray_module_project/pytapo/face2/train"
if not os.path.exists(train_path):
    os.makedirs(train_path)

# 웹캠 열기
cap = cv2.VideoCapture(0)
# 저장된 얼굴 이미지 카운트
count = 0

# 100개의 얼굴 이미지를 수집할 때까지 반복
while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    # 함수 호출 결과 저장
    faces = face_extractor(frame)
    
    if faces:
        for i, face in enumerate(faces):
            count += 1
            # 추출한 얼굴 크기 조정 및 회색조 변환
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 이미지 파일 저장
            file_name_path = f'train/user{count}.jpg'  # train 폴더에 user1.jpg, user2.jpg, ... 형식으로 저장
            cv2.imwrite(file_name_path, face)

            # 이미지에 카운트 표시
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # 이미지 출력
            cv2.imshow('Face train', face)
    else:
        print("얼굴을 못 찾았습니다.")

    # ESC 키를 누르거나 100개의 이미지를 수집하면 종료
    if cv2.waitKey(1) == 27 or count == 100:
        break

# 웹캠 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
print('-----STEP_1. 인코딩용 데이터를 수집완료하였습니다-----')