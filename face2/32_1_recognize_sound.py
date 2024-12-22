import os
import cv2
import csv
import time
import pyttsx3


# 음성 엔진 초기화 (여자 목소리 설정)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if voice.gender == 'female':
        engine.setProperty('voice', voice.id)
        break

# 3번 프로세스: 얼굴 인식
def face_recognizer(user_name, train_path):
    # train_path에서 모델 로드
    model_file_path = os.path.join(train_path, "trained_model.yml")

    # 모델 파일 존재 여부 확인
    if not os.path.exists(model_file_path):
        print(f"모델 파일이 존재하지 않습니다: {model_file_path}")
        return  # 함수 종료

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_file_path)  # 저장된 모델 파일 로드

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is None or len(faces) == 0:
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    # 캠 켜기
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            if face is not None and len(face) > 0: # 얼굴 감지 확인
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)

                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1]) / 300))
                    display_string = str(confidence) + '% Confidence it is ' + user_name
                cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

                if confidence > 70:
                    cv2.putText(image, "Enjoy your meal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('FACE RECOGNITION', image)
                    print("-------------debug1-------------")
                    engine.say(f"{user_name}님 식사 맛있게 하세요") # 음성 출력 추가
                    engine.runAndWait() # 음성 출력 완료 대기
                    # 인식 후 바로 종료 (1번만 인식)
                    print("-------------debug2-------------")                
                    time.sleep(10)
                    break
                else:
                    cv2.putText(image, "You are not our member", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('FACE RECOGNITION', image)
                    
            else:
                raise Exception("Face Not Found")

        except Exception as e: # except문에서 e로 에러 정보 받기
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('FACE RECOGNITION', image)
            print(f"에러 발생: {e}") # 에러 정보 출력


        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()


# 학습 정보 로드 (user_info.csv에서 정보 가져오기)
try:
    with open("user_info.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_name = row["이름"]
            train_path = os.path.join("train", row["아이디"])
            break
except FileNotFoundError:
    print("학습 정보 파일이 없습니다. 먼저 사용자를 등록해주세요.")
    exit()

# 안면인식 시작
face_recognizer(user_name, train_path)