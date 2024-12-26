
import os
import cv2
import csv
import numpy as np
import time
import glob

# global 변수
correct_uid = ''


# 개인 별로 나뉘어진 학습 데이터 통합
def create_unified_model(csv_filepath="user_info.csv"):
    """Creates and trains a unified face recognition model from CSV data."""

    # global
    global correct_uid

    Training_Data, Labels = [], []
    label_mapping = {}  # Map user IDs to labels

    try:
        with open(csv_filepath, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            label_count = 0
            for row in reader:
                user_id = row["아이디"]
                user_name = row["이름"]
                train_path = os.path.join("train", user_id)

                # Add user to label mapping
                if user_id not in label_mapping:
                    
                    correct_uid = ''
                    
                    label_mapping[user_id] = {"label": label_count, "name": user_name}
                    label_count += 1

                image_paths = glob.glob(os.path.join(train_path, "*.jpg"))
                for image_path in image_paths:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    Training_Data.append(np.asarray(image, dtype=np.uint8))
                    Labels.append(label_mapping[user_id]["label"])

        Labels = np.asarray(Labels, dtype=np.int32)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(Training_Data), np.asarray(Labels))
        model.save("unified_trained_model.yml")  # Save the unified model
        return model, label_mapping


    except FileNotFoundError:
        print("CSV file not found for training.")
        return None, None


# 캠으로 인식된 얼굴을 식별하는 모델
def face_detector(img, size=0.5):  # Helper function (unchanged)
    global correct_uid
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is None or len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# 학습된 얼굴을 인식하는 추론 모델
def face_recognizer(model, label_mapping):
    """Recognizes faces using the provided unified model."""
    
    width = 640
    height = 480
    
    cv2.namedWindow('FACE RECOGNITION', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FACE RECOGNITION', width, height)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            if face is not None and len(face) > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)
                user_id = list(label_mapping.keys())[result[0]]  # Get user ID
                user_name = label_mapping[user_id]["name"] # and name from mapping

                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1]) / 300))
                    display_string = f"{confidence}% Confidence: {user_name} ({user_id})"    
                               
                
                cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

                if confidence > 80:
                    cv2.putText(image, "Enjoy your meal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('FACE RECOGNITION', image)
                    
                    # 제대로 인식했을 때
                    correct_uid = user_id
                    print(display_string)
                    print(correct_uid)
                    
                    #break
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


# Main execution block
if __name__ == "__main__":
    model, label_mapping = create_unified_model()
    if model is not None:
        face_recognizer(model, label_mapping)
    else:
        print("Error creating unified model. Exiting.")