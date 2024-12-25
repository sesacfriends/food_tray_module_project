
# 공지
# 사진찍을때 공지하기 > 사진 촬영 시작
    # 안면인식 시스템은 얼굴 전체를 인식하도록 설정되었을 때 가장 정확하게 동작합니다.
    # 안경, 마스크, 모자를 벗고 사진을 촬영해주세요.
    # 1. 프로필 사진을 등록합니다. 웃으세요 스마일
    # 2. 얼굴인식용 사진을 촬영합니다. + 프로그레스 바
        # 얼굴 위치를 프레임에 맞춰주세요
        # 머리를 천천히 움직여 원을 만드세요
        # 정면을 바라봐주세요
    # 3. 사진촬영이 완료되었습니다. 감사합니다. 3초 후 HOME으로 자동 이동합니다.

import cv2
import os
import numpy as np
import glob
import csv
import re

    #pip install opencv-contrib-python

###################################################################################
# PROCESS 01_정의 > 얼굴 인식을 위한 얼굴 데이터, 이름 추출                          
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
# PROCESS 02_정의 > 사용자 정보 입력                              
###################################################################################

    # 입력 유효성검사
def validate_input(input_str, pattern, message):
    while True:
        value = input(message)
        if re.fullmatch(pattern, value):    # 정규표현식 > 입력 유효성 검사
            return value
        else:
            print("잘못된 입력입니다. 다시 입력하세요.")
            
            
    # 사용자 정보 수집
def collect_user_info():
    user_id, _ = generate_folder("") # 사용자 ID 생성
    user_name = input("등록할 사용자 이름을 입력하세요 : ")
    if len(user_name) > 6:
        print("사용자 이름은 최대 6글자입니다.")
        return None
    
    birth_date = validate_input(
        "", r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", "생년월일 (YYYYMMDD): "
    )

    gender = validate_input("", r"^[FM]$", "성별 (F/M): ")
    phone_number = validate_input("", r"^010\d{8}$", "연락처 (010xxxxxxxx): ")

    while True:
        address = input("주소 (시/군/구): ")
        if address: # 주소 입력되면 루프 종료
            break
        else:
            print("주소를 입력해야 합니다.")
            
    height = float(validate_input("", r"^\d{1,3}(\.\d)?$", "키 (소수점 첫째 자리까지): "))
    weight = float(validate_input("", r"^\d{1,3}(\.\d)?$", "몸무게 (소수점 첫째 자리까지): "))

    return {
        "아이디": user_id,
        "이름": user_name,
        "생년월일": birth_date,
        "성별": gender,
        "연락처": phone_number,
        "주소": address,
        "키": height,
        "몸무게": weight,
        "운동량": ""    # 초기값 빈 문자열
    }

def save_csv(user_info, filename ="user_info.csv"):
    file_exists = os.path.exists(filename)
    
    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "아이디", "이름", "생년월일", "성별", "연락처", "주소", "키", "몸무게", "운동량"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(user_info)
            

###################################################################################
# PROCESS 03_정의 > 데이터 학습                                                    
###################################################################################


import numpy as np
import glob

    # 모델 학습
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
# PROCESS 실행 > 사용자 정보 입력, 폴더 생성, 이미지 캡처                              #
###################################################################################


if __name__ == "__main__":
    user_info = collect_user_info()
    if user_info:
        train_path = os.path.join("train", user_info["아이디"])
        os.makedirs(train_path, exist_ok=True)

        # 웹캠 열기
    cap = cv2.VideoCapture(0)
        # 저장된 얼굴 이미지 카운트
    count = 0

        # 200개의 얼굴 이미지를 수집할 때까지 반복
    while True:
        ret, frame = cap.read()             # 웹캠에서 프레임 읽기
        faces = face_extractor(frame)       # 함수 호출 결과 저장
        
        if faces:
            for i, face in enumerate(faces):
                count += 1
                face = cv2.resize(face, (200, 200))             # 추출한 얼굴 크기 조정 및 회색조 변환
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # 이미지 파일 저장
                file_name_path = os.path.join(train_path, f'{user_info["아이디"]}_{count}.jpg')  # 파일 이름에 폴더명 포함
                cv2.imwrite(file_name_path, face)

                # 이미지에 카운트 표시
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # 이미지 출력
                cv2.imshow('FACE TRAIN', face)
                
        else:
            print("얼굴을 못 찾았습니다.")

        # ESC 키 누름 or 200개의 이미지를 수집하면 종료
        if cv2.waitKey(1) == 27 or count == 200:
            break

        # 웹캠 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()
    print(f'-----{user_info["이름"]}님 ({user_info["아이디"]})의 인코딩용 데이터를 수집완료하였습니다-----')
    print('-----STEP_1. 인코딩용 데이터를 수집완료하였습니다. -----')


    # 2번 프로세스 호출 > train
    train_model(train_path) # 1번 프로세스에서 생성된 train_path 전달
    print('-----STEP_2. 인코딩용 데이터를 학습 완료하였습니다. -----')

    # 학습 정보 저장
    save_csv(user_info)
    print("-----STEP_3. 사용자 정보가 CSV 파일에 저장되었습니다. -----")