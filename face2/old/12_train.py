    #pip install opencv-contrib-python

import cv2
import os

###################################################################################
# PROCESS 01_정의 > 얼굴 인식을 위한 얼굴 데이터, 이름 추출                               #
###################################################################################


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

    # 모든 얼굴 영역을 리스트로 반환
    return [img[y:y+h, x:x+w] for (x, y, w, h) in faces] 

    # 폴더 이름 생성 함수 (sesac + numbering)
def generate_folder(name):
    i = 1
    while True:
        folder_name = f"sesac{i:02d}"   # 2자리 넘버링 (01, 02...)
        folder_path = os.path.join("train", folder_name)
        if not os.path.exists(folder_path):
            return folder_name, folder_path
        i += 1



###################################################################################
# PROCESS 02_정의 > 데이터 학습                                                    #
###################################################################################


import numpy as np
import glob

# 2번 프로세스: 모델 학습
def train_model(train_path):  # train_path를 인자로 받음
    Training_Data, Labels = [], []

    # glob을 사용하여 모든 이미지 파일 경로 가져오기
    image_paths = glob.glob(os.path.join(train_path, "*.jpg")) # 특정 폴더 내의 모든 jpg 파일

    for i, image_path in enumerate(image_paths):
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)  # 각 사용자 폴더 내에서 라벨링

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    # 모델 저장 (train_path에 저장)
    model_file_path = os.path.join(train_path, "trained_model.yml")
    model.save(model_file_path) # 각 사용자 폴더에 모델 저장
    print("-----STEP_2. 모델 학습이 완료되었습니다. -----")


###################################################################################
# PROCESS 01_실행 > 사용자 입력, 폴더 생성, 이미지 캡처                              #
###################################################################################

# 사용자_이름 입력받고 폴더 생성
user_name = input("등록할 사용자 이름을 입력하세요 : ")
folder_name, train_path = generate_folder(user_name)
os.makedirs(train_path)


    # 웹캠 열기
cap = cv2.VideoCapture(0)
    # 저장된 얼굴 이미지 카운트
count = 0

    # 100개의 얼굴 이미지를 수집할 때까지 반복
while True:
    ret, frame = cap.read()             # 웹캠에서 프레임 읽기
    faces = face_extractor(frame)       # 함수 호출 결과 저장
    
    if faces:
        for i, face in enumerate(faces):
            count += 1
            face = cv2.resize(face, (200, 200))             # 추출한 얼굴 크기 조정 및 회색조 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 이미지 파일 저장
            file_name_path = os.path.join(train_path, f'{folder_name}_{count}.jpg')  # 파일 이름에 폴더명 포함
            cv2.imwrite(file_name_path, face)

            # 이미지에 카운트 표시
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # 이미지 출력
            cv2.imshow('FACE TRAIN', face)
            
    else:
        print("얼굴을 못 찾았습니다.")

    # ESC 키 누름 or 100개의 이미지를 수집하면 종료
    if cv2.waitKey(1) == 27 or count == 100:
        break

    # 웹캠 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
print(f'-----{user_name}님 ({folder_name})의 인코딩용 데이터를 수집완료하였습니다-----')
print('-----STEP_1. 인코딩용 데이터를 수집완료하였습니다-----')



# 2번 프로세스 호출 > train
train_model(train_path) # 1번 프로세스에서 생성된 train_path 전달