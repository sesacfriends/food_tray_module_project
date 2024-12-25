import cv2
import csv
import numpy as np
import pyttsx3
import glob
import os

class FaceRecognitionSystem:
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml', model_save_path="unified_trained_model.yml"):
        self.cascade_path = cascade_path
        self.model_save_path = model_save_path
        self.correct_uid = ''
        self.engine = self._initialize_voice_engine()
        self.face_classifier = cv2.CascadeClassifier(self.cascade_path)

    def _initialize_voice_engine(self):
        """Initialize the voice engine with a female voice."""
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if voice.gender == 'female':
                engine.setProperty('voice', voice.id)
                break
        return engine

    def create_unified_model(self, csv_filepath="FOOD_DB/user_info.csv"):
        """Creates and trains a unified face recognition model from CSV data."""
        training_data, labels = [], []
        label_mapping = {}

        try:
            with open(csv_filepath, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                label_count = 0
                for row in reader:
                    user_id = row["id"]
                    user_name = row["name"]
                    train_path = os.path.join("train", user_id)

                    if user_id not in label_mapping:
                        label_mapping[user_id] = {"label": label_count, "name": user_name}
                        label_count += 1

                    image_paths = glob.glob(os.path.join(train_path, "*.jpg"))
                    for image_path in image_paths:
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        training_data.append(np.asarray(image, dtype=np.uint8))
                        labels.append(label_mapping[user_id]["label"])

            labels = np.asarray(labels, dtype=np.int32)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(training_data), labels)
            model.save(self.model_save_path)
            return model, label_mapping

        except FileNotFoundError:
            print("CSV file not found for training.")
            return None, None

    def face_detector(self, img, size=0.5):
        """Detects a face and returns the ROI."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is None or len(faces) == 0:
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    def face_recognizer(self, model, label_mapping):
        """Recognizes faces using the provided unified model."""
        width, height = 640, 480
        cv2.namedWindow('FACE RECOGNITION', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FACE RECOGNITION', width, height)
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

        while True:
            ret, frame = cap.read()
            image, face = self.face_detector(frame)

            try:
                if face is not None and len(face) > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face)
                    user_id = list(label_mapping.keys())[result[0]]
                    user_name = label_mapping[user_id]["name"]

                    if result[1] < 500:
                        confidence = int(100 * (1 - (result[1]) / 300))
                        display_string = f"{confidence}% Confidence: {user_name} ({user_id})"

                        self.correct_uid = user_id
                        print(self.correct_uid)

                    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

                    if confidence > 75:
                        cv2.putText(image, "Enjoy your meal", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('FACE RECOGNITION', image)
                        self.engine.say(f"{user_name}님 식사 맛있게 하세요")
                        self.engine.runAndWait()
                        print(display_string)
                        break
                    else:
                        cv2.putText(image, "You are not our member", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('FACE RECOGNITION', image)

            except Exception as e:
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('FACE RECOGNITION', image)
                print(f"Error occurred: {e}")

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.correct_uid

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    model, label_mapping = system.create_unified_model()

    if model is not None:
        correct_uid = system.face_recognizer(model, label_mapping)
        print(f"Recognized UID: {correct_uid}")
    else:
        print("Error creating unified model. Exiting.")