
import os
import cv2

###################################################################################
# PROCESS 03_정의 > 사용자 안면 인식                                                #
###################################################################################


# 3번 프로세스: 얼굴 인식
def face_recognizer(user_name, train_path):
    # train_path에서 모델 로드
    model_file_path = os.path.join(train_path, "trained_model.yml")
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
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% Confidence it is ' + user_name # 사용자 이름 표시
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
#####
# 3번 프로세스 호출 > face_recognition

#face_recognizer(user_name, train_path)