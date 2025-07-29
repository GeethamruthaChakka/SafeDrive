
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize pygame mixer
mixer.init()
mixer.music.load("music.wav")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Vertical distance
    B = distance.euclidean(mouth[4], mouth[8])   # Vertical distance
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

# Threshold values
eye_thresh = 0.25  # EAR threshold for drowsiness
mouth_thresh = 0.6  # MAR threshold for yawning
frame_check = 20  # Number of frames to trigger alert

# Initialize dlib face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\gopal\Downloads\DriverDrowsiness\models\shape_predictor_68_face_landmarks.dat")

# Get facial landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Start webcam
cap = cv2.VideoCapture(0)
eye_flag = 0
mouth_flag = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Compute EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Compute MAR
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw facial landmarks
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < eye_thresh:
            eye_flag += 1
            if eye_flag >= frame_check:
                cv2.putText(frame, "*ALERT! DROWSY*", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                mixer.music.play()
        else:
            eye_flag = 0  # Reset eye counter

        # Check for yawning
        if mar > mouth_thresh:
            mouth_flag += 1
            if mouth_flag >= frame_check:
                cv2.putText(frame, "*ALERT! YAWNING*", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                mixer.music.play()
        else:
            mouth_flag = 0  # Reset mouth counter

    cv2.imshow("Frame", frame)
    
    # Break on 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()