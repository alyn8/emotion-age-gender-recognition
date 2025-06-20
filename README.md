# emotion-age-gender-recognition
# 👤 Real-Time Emotion, Age, and Gender Recognition with CNNs

This project presents a Flask-based web application capable of recognizing **emotion**, **age**, and **gender** from human faces using **Convolutional Neural Networks (CNNs)**. It supports real-time webcam input, image upload, and video file analysis. The trained models are deployed via an intuitive web interface, making it practical for human-computer interaction scenarios.

🌐 Features
Real-time webcam inference

Image and video upload support

Visual feedback with bounding boxes

Safety and demographic insights from visual input

🖼️ Sample Outputs
🎞️ Video Output

🖼️ Image Analysis

📊 Model Architecture Overview
A custom CNN with Conv2D, MaxPooling, Dropout, and Dense layers was used for each task, optimized via early stopping and batch normalization.

## 🧠 Models

Three separate CNN models were developed for:

| Task                | Dataset    | Evaluation Metric    | Performance            |
|---------------------|------------|-----------------------|------------------------|
| Emotion Detection   | FER2013    | Accuracy, F1-Score    | ~67.28% Accuracy       |
| Age Estimation      | UTKFace    | MAE, MSE, R² Score    | MAE: 5.98 / R²: 0.82   |
| Gender Classification | UTKFace  | Accuracy, F1-Score    | ~88.9% Accuracy        |

🧩 **Download Trained Models** from Kaggle:

- Emotion: [FER2013 Emotion CNN](https://www.kaggle.com/datasets/msambare/fer2013)
- Age & Gender: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)

(*Note: Direct model links or download instructions can be added in `models/README.md`*)

## 🚀 How to Run

```bash

# 1. Clone the repo
git clone https://github.com/yourusername/emotion-age-gender-flask-app.git
cd emotion-age-gender-flask-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

```

## 📁 Project Structure

├── app.py
├── static/
│ └── (CSS, images, icons)
├── templates/
│ ├── index.html
│ ├── photo.html
│ ├── video.html
│ ├── webcam.html
├── models/
│ └── (Optional local placeholder or download script)
├── utils/
│ └── haarcascade_frontalface_default.xml
├── output_samples/
│ └── example_photos/
│ └── example_videos/
├── README.md
