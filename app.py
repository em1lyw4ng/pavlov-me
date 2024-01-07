import cv2
import dlib

# initialize camera
cap = cv2.VideoCapture(0)

# initialize dlib face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# initialize predictor from dataset
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

while True:
    # 👁
    ret, frame = cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = detector(gray)

    for face in faces:
        # extract facial landmarks
        shape = predictor(gray, face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()

