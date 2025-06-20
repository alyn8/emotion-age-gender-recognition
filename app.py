from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import ffmpeg
import time
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok= True)


print("Models downloading..")
custom_objects = {"mse" : tf.keras.losses.MeanSquaredError()}

models = {
    "face_classifier": cv2.CascadeClassifier('haarscascadeModels/haarcascade_frontalface_default.xml'),
    "emotion_model": load_model('models/emotion_detection_60reg.h5'),
    "age_model": load_model('models/age_model_50epochs.h5', custom_objects=custom_objects),
    "gender_model": load_model('models/gender_model_50epochs.h5'),
    "class_labels": ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    "gender_labels": ['Male', 'Female']
}
print("Models downloaded!")



#frame by frame detection from videos
def detect_faces_from_frame(frame):
    height, width = frame.shape[:2]
    new_width = 1000
    new_height = int((new_width / width) * height)
    frame = cv2.resize(frame, (new_width, new_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = models["face_classifier"].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return frame


    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    roi_gray = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    emotion_pred = models["emotion_model"].predict(roi, verbose=0)[0]
    emotion = models["class_labels"][np.argmax(emotion_pred)]

    roi_color = cv2.resize(frame[y:y + h, x:x + w], (128, 128))


    gender_pred = models["gender_model"].predict(np.expand_dims(roi_color, axis=0), verbose=0)
    gender = models["gender_labels"][int(gender_pred > 0.5)]

    roi_color = roi_color.astype('float32') / 255.0

    age_pred = models["age_model"].predict(np.expand_dims(roi_color, axis=0))

    age = int(round(age_pred[0, 0]))



    label = f"{emotion},{gender}, Age:{age}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame



#mainPage
@app.route('/')
def home():
    return render_template('base.html')

#upload_photo
@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        file = request.files['photo']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            session['photo_path'] = path

            return redirect(url_for('upload_photo'))
        else:
            flash("No file selected.", "error")
    if 'photo_path' not in session:
        session.pop('photo_path',None)

    return render_template('upload_photo.html')

#upload_video
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files['video']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            session['video_path'] = path
            return redirect(url_for('video_stream'))
        else:
            flash("No file selected.", "error")

    return render_template('upload_video.html')

@app.route('/photo_stream')
def photo_stream():
    photo_path = session.get('photo_path')
    if not photo_path or not os.path.exists(photo_path):
        return "No photo found", 404

    img = cv2.imread(photo_path)
    result = detect_faces_from_frame(img)
    _, buffer = cv2.imencode('.jpg', result)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/video_stream')
def video_stream():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "No video found", 404

    def generate():
        cap = cv2.VideoCapture(video_path)
        frame_skip = 3
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            processed = detect_faces_from_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')




#webcam
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_feed')
def webcam_feed():
    def gen():
        cap = cv2.VideoCapture(0)

        frame_skip = 5
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = detect_faces_from_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)