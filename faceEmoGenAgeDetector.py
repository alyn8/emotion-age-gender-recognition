from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tensorflow as tf
import cv2
import numpy as np
from time import sleep

custom_objects = {"mse":tf.keras.losses.MeanSquaredError()}


face_classifier = cv2.CascadeClassifier('haarscascadeModels/haarcascade_frontalface_default.xml')
emotion_model = load_model('models/emotion_detection_60reg.h5')
age_model = load_model('models/age_model_50epochs.h5', custom_objects=custom_objects)
gender_model = load_model('models/gender_model_50epochs.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

#"C:\\Users\\Aleyna\\Downloads\\cryingWoman_Clipchamp ile yapıldı (1).mp4"
#"C:\\Users\\Aleyna\\Downloads\\oldCoupleSmiling _Clipchamp ile yapıldı (1).mp4"
#"C:\\Users\\Aleyna\\Downloads\\iamnotindanger.mp4"
#"C:\\Users\\Aleyna\\Downloads\\iamthedanger‐ Clipchamp ile yapıldı (1).mp4"
#C:\\Users\\Aleyna\\Downloads\\iamthedanger_Clipchamp ile yapıldı.mp4

video_path = "C:\\Users\\Aleyna\\Downloads\\oldCoupleSmiling _Clipchamp ile yapıldı (1).mp4"

#cap=cv2.VideoCapture(video_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Video file has not been found.Check video path again.")
    exit()

frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    labels = []

    if not ret:
        print("Video is over or occured error while is playing")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

        preds = emotion_model.predict(roi)[0]  # Yields one hot encoded result for 7 classes
        label = class_labels[preds.argmax()]  # Find the label
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gender
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (128, 128), interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, 128, 128, 3))
        gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
        gender_label = gender_labels[gender_predict[0]]
        gender_label_position = (x, y + h + 50)  # 50 pixels below to move the label outside the face
        cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Age
        roi_color = roi_color.astype('float32') / 255.0
        age_predict = age_model.predict(np.expand_dims(roi_color, axis=0))
        #age_predict = age_model.predict(np.array(roi_color).reshape(-1, 128, 128, 3))
        age = int(round(age_predict[0, 0]))

        age_label_position = (x + h, y + h)
        cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()