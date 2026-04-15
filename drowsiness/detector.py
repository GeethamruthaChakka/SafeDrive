import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
import joblib

# -------------------- MEDIAPIPE --------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -------------------- LANDMARKS --------------------
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
UPPER_LIP = 13
LOWER_LIP = 14

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 25

# -------------------- LOAD SVM --------------------
MODEL_PATH = "svm_drowsiness_final.pkl"
svm = joblib.load(MODEL_PATH)

# -------------------- MOBILENET --------------------
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# -------------------- PREPROCESS --------------------
def preprocess(frame):
    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

# -------------------- EAR --------------------
def eye_aspect_ratio(eye, landmarks, w, h):

    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]

    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])

    if C == 0:
        return 0

    return (A + B) / (2 * C)

# -------------------- MAR --------------------
def mouth_ratio(landmarks, w, h):

    x1 = int(landmarks[UPPER_LIP].x * w)
    y1 = int(landmarks[UPPER_LIP].y * h)

    x2 = int(landmarks[LOWER_LIP].x * w)
    y2 = int(landmarks[LOWER_LIP].y * h)

    return distance.euclidean((x1, y1), (x2, y2))

# -------------------- MOBILENET + SVM --------------------
def hybrid_predict(frame, ear, mar):

    img = cv2.resize(frame, (224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    features = mobilenet.predict(img, verbose=0)[0]  # 1280 features

    combined = np.append(features, [ear, mar]).reshape(1, -1)

    pred = svm.predict(combined)[0]

    return "DROWSY" if pred == 1 else "NOT DROWSY"

# -------------------- MAIN DETECTION --------------------
def detect_frame(frame):

    gray = preprocess(frame)
    h, w = gray.shape[:2]

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    face_result = face_mesh.process(rgb)

    if not face_result.multi_face_landmarks:
        return "FACE NOT DETECTED"

    landmarks = face_result.multi_face_landmarks[0].landmark

    # EAR
    leftEAR = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
    rightEAR = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
    ear = (leftEAR + rightEAR) / 2

    # MAR
    mar = mouth_ratio(landmarks, w, h)


    if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
        return "DROWSY"

    return hybrid_predict(frame, ear, mar)

# -------------------- IMAGE DETECTION--------------------
def detect_image(path):

    img = cv2.imread(path)

    if img is None:
        return "IMAGE ERROR"

    return detect_frame(img)