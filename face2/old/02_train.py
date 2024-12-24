import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# 훈련 이미지가 저장된 경로
data_path = 'train/'
# 해당 경로에 있는 파일 목록 가져오기 (폴더는 제외)
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# 훈련 데이터와 레이블을 저장할 리스트 초기화
Training_Data, Labels = [], []

# 각 이미지 파일을 순회하며 훈련 데이터와 레이블 생성
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    # 이미지를 회색조로 읽어오기
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 훈련 데이터에 이미지 추가 (NumPy 배열로 변환)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    # 레이블에 인덱스 추가 (0, 1, 2, ...)
    Labels.append(i)

# 레이블을 NumPy 배열로 변환 (int32 타입)
Labels = np.asarray(Labels, dtype=np.int32)

# LBPH 얼굴 인식 모델 생성
    # OpenCV 4 이상에서 LBPHFaceRecognizer 생성
model = cv2.face_LBPHFaceRecognizer.create()
    # OpenCV 3 이하에서는 아래 코드로 작성
    # model = cv2.face.LBPHFaceRecognizer_create()

# 훈련 데이터와 레이블을 사용하여 모델 학습
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("-----STEP_2. 모델 학습이 완료되었습니다. -----")