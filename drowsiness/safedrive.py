
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from scipy.spatial import distance
from playsound import playsound

# -------------------- ALARM --------------------
alarm_on = False

def sound_alarm():
    global alarm_on
    while alarm_on:
        playsound("alarm.wav")

# -------------------- EAR FUNCTION --------------------
def eye_aspect_ratio(eye, landmarks, w, h):

    points = []

    for i in eye:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))

    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])

    if C == 0:
        return 0, points

    ear = (A + B) / (2.0 * C)

    return ear, points

# -------------------- LANDMARK INDEX --------------------
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

# -------------------- THRESHOLDS --------------------
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 35
FRAME_LIMIT = 15

frame_counter = 0
ear_list = []   

# -------------------- MEDIAPIPE --------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)


cap.set(3, 640)
cap.set(4, 480)

print("SafeDrive Started...")

# -------------------- LOOP --------------------
while True:

    start = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status_text = "Happy Journey"
    color = (0,255,0)

    eyes_drowsy = False
    yawning = False

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            # -------------------- EAR --------------------
            leftEAR, left_coords = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            rightEAR, right_coords = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)

            ear = (leftEAR + rightEAR) / 2.0

            ear_list.append(ear)
            if len(ear_list) > 5:
                ear_list.pop(0)
            ear = sum(ear_list) / len(ear_list)

            # Draw eye points
            for (x,y) in left_coords + right_coords:
                cv2.circle(frame,(x,y),2,(0,255,0),-1)

            # -------------------- MAR --------------------
            mx1 = int(landmarks[13].x * w)
            my1 = int(landmarks[13].y * h)

            mx2 = int(landmarks[14].x * w)
            my2 = int(landmarks[14].y * h)

            mar = distance.euclidean((mx1,my1),(mx2,my2))

            # -------------------- EYE DETECTION --------------------
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= FRAME_LIMIT:
                    eyes_drowsy = True
            else:
                frame_counter = 0

            # -------------------- YAWN DETECTION --------------------
            if mar > MAR_THRESHOLD:
                yawning = True

            # -------------------- STATUS --------------------
            if eyes_drowsy:
                status_text = "YOU ARE DROWSY!"
                color = (0,0,255)

            elif yawning:
                status_text = "YAWNING DETECTED!"
                color = (255,0,0)

            else:
                status_text = "Happy Journey..."
                color = (0,255,0)

            # -------------------- ALARM --------------------
            if eyes_drowsy or yawning:
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
            else:
                alarm_on = False

            # -------------------- DISPLAY --------------------
            cv2.putText(frame,f"EAR: {ear:.2f}",(30,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,255,0),2)

            cv2.putText(frame,f"MAR: {mar:.2f}",(30,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,0,255),2)

    # FPS
    fps = 1/(time.time()-start)

    cv2.putText(frame,f"FPS: {int(fps)}",(500,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)

    cv2.putText(frame,status_text,(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,color,3)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()