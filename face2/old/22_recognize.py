import os
import cv2
import json

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

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # try문 안으로 이동
            result = model.predict(face) # try문 안으로 이동

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% Confidence it is ' + user_name
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if confidence > 75:
                cv2.putText(image, "Enjoy your meal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('FACE RECOGNITION', image)
            else:
                cv2.putText(image, "You are not our member", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('FACE RECOGNITION', image)

        except Exception as e: # except문에서 e로 에러 정보 받기
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('FACE RECOGNITION', image)
            print(f"에러 발생: {e}") # 에러 정보 출력


        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()


# 학습 정보 로드
try:
    with open("account_json", "r") as f:
        account_info = json.load(f)
    user_name = account_info["user_name"]
    train_path = account_info["train_path"]
except FileNotFoundError:
    print("학습 정보 파일이 없습니다. 먼저 사용자를 등록해주세요.")
    exit()

# 3번 프로세스 호출 > face_recognition
face_recognizer(user_name, train_path)