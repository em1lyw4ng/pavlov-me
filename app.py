import cv2
import dlib
import mediapipe as mp

# initialize camera
cap = cv2.VideoCapture(0)

# initialize dlib face detector (HOG-based)
# and initialize face predictor from dataset
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# initialize mediapipe hands object
mp_hands = mp.solutions.hands.Hands()

while True:
    # 🎥
    ret, frame = cap.read()

    # convert frame to grayscale and rgb
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces and hands
    faces = detector(gray)
    hands = mp_hands.process(rgb)

    for face in faces:
        # extract facial landmarks from detection
        shape = predictor(gray, face)

        # mouth is in landmarks 48-68 
        mouth = shape.parts()[48:68]

        # iterate over each landmark and draw 
        for landmark in mouth:
            cv2.circle(frame, (landmark.x, landmark.y), 2, (0, 255, 0), -1)

    # check if hand is detected
    if hands.multi_hand_landmarks:
        # iterate over each detected hand
        for hand in hands.multi_hand_landmarks:
            # iterate over each landmark and draw
            for landmark in hand.landmark:
                h, w, c = frame.shape
                # convert normalized position to pixel coordinates in frame
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    
    # display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

