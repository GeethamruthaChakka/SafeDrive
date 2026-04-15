# 🚗 SafeDrive – Driver Drowsiness Detection System

> A real-time AI-based system that detects driver fatigue using facial landmarks, MAR, and deep learning (MobileNetV2), and alerts the driver to prevent accidents.

---

## 📖 Overview

SafeDrive is an intelligent driver monitoring system that detects drowsiness using computer vision and deep learning techniques. It combines Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and a MobileNetV2-based model to improve detection accuracy.

---

## ✨ Features

* 👁️ Eye tracking using Eye Aspect Ratio (EAR)
* 😮 Yawning detection using Mouth Aspect Ratio (MAR)
* 🤖 Deep learning model using MobileNetV2
* 🔔 Real-time alert system (alarm)
* 🎥 Works with webcam/live video
* 🌗 Robust under different lighting conditions

---

## 🧠 How It Works

1. Capture video using webcam
2. Detect face using MediaPipe/dlib
3. Extract facial landmarks (eyes & mouth)
4. Calculate:

   * EAR (Eye Aspect Ratio) → detects eye closure
   * MAR (Mouth Aspect Ratio) → detects yawning
5. Pass frames to **MobileNetV2 model** for classification
6. If drowsiness detected → trigger alarm

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** OpenCV, MediaPipe / dlib, NumPy
* **Deep Learning:** MobileNetV2 (CNN)
* **Backend (Optional):** Flask

---

## 📂 Project Structure

```id="c5yxgm"
SafeDrive/
│── drowsiness/
│   ├── detector.py
│   ├── mobilenet_model/
│   ├── alarm.wav
│   ├── utils/
│── app.py
│── requirements.txt
│── README.md
```

---

## 🚀 Getting Started

### 🔹 Clone Repository

```id="ehg2zm"
git clone https://github.com/GeethamruthaChakka/SafeDrive.git
cd SafeDrive
```

### 🔹 Install Dependencies

```id="sj4hnh"
pip install -r requirements.txt
```

### 🔹 Run the Application

```id="99mrg0"
python app.py
```

---

## 📊 Model Details

* **Model Used:** MobileNetV2
* **Purpose:** Classifies driver state (Alert / Drowsy)
* **Input:** Facial frames
* **Output:** Drowsiness prediction

---

## 📈 Future Scope

* 📱 Mobile app integration
* ☁️ Cloud-based monitoring
* 🚘 Integration with smart vehicles
* 📊 Driver behavior analytics

