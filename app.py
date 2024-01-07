import cv2
import dlib
import time
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

# for cooldown calculation
last_detection_time = 0

while True:
    # 🎥
    ret, frame = cap.read()

    # convert frame to grayscale and rgb
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces and hands
    faces = detector(gray)
    hands = mp_hands.process(rgb)

    # saves detected points of mouth and hand
    mouth_landmarks = []
    hand_landmarks = []

    for face in faces:
        # extract facial landmarks from detection
        shape = predictor(gray, face)

        # mouth is in landmarks 48-68 
        mouth = shape.parts()[48:68]

        # iterate over each landmark and draw 
        for landmark in mouth:
            mouth_landmarks.append((landmark.x, landmark.y))
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
                hand_landmarks.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    
    current_time = time.time()
    hands_on = False

    # calculate time elapsed for cooldown
    if current_time - last_detection_time > 2:
        # check if any hand landmark is in proximity of mouth landmarks
        for hand_point in hand_landmarks:
            for mouth_point in mouth_landmarks:
                # sqrt((x1 - x2)**2 + (y1-y2)**2)
                distance = ((hand_point[0] - mouth_point[0]) ** 2 + (hand_point[1] - mouth_point[1]) ** 2) ** 0.5
                if distance < 50 and not hands_on:
                    hands_on = True
                    last_detection_time = current_time
                    print("hands off!!")
                
    # display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

