# IMPORTS
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils  
from unicornhatmini import UnicornHATMini
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# SETUP LED MATRIX
fl = UnicornHATMini()
fl.set_brightness(0.5)

# GET THE ASPECT RATIO OF THE EYES
def get_eye_aspect_ratio(eye):
    landmarkHeight1 = dist.euclidean(eye[1], eye[5])
    landmarkHeight2 = dist.euclidean(eye[2], eye[4])


    landmarkWidth = dist.euclidean(eye[0], eye[3])

    ratio = (landmarkHeight1 + landmarkHeight2) / (2.0 * landmarkWidth)

    return ratio

# CHECK IF THE EYES ARE CLOSED LONG ENOUGH
def wake_up():
    if alarm:
        cv2.putText(frame, "Eye: {}".format("sleeping"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        flash()
    else:
        idle()

# FLASH THE LED MATRIX
def flash():
    while a < 5:
        for x in range(17):
            for y in range(7):
                fl.set_pixel(x,y,255,255,255)
        
        a += 1
        fl.show()
        time.sleep(0.5)

        fl.clear()
        fl.show()
        time.sleep(0.5)

        print(a)
        
# IDLE THE LED MATRIX
def idle():
    for x in range(17):
        for y in range(7):
            fl.set_pixel(x,y,0,0,255)
    fl.show()

#CONDITIONS TO BE MET FOR THE SLEEP CLASSIFICATION
THRESH = 0.35
DURATION = 3

# INITIALIZE COUNTERS AND DLIB MODEL 
counter = 0
alarm = False
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# STORE LANDMARKS OF EACH EYE 
(l_R_Start, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_Start, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# LOAD FROM WEBCAM 0 (Raspberry PI Camera)
vs = VideoStream(src=0).start()
time.sleep(1)

# LOOP OVER FRAMES
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=690)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # grayscale frame
    rects = detector(gray, 0)
    
    # LOOP THROUGH EACH EYE IN FACE
    for rect in rects:
        # PREDICT EYE
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        l_eye = shape[l_R_Start:lEnd]
        r_eye = shape[R_Start:rEnd]
        
        l_ratio = get_eye_aspect_ratio(l_eye)
        r_ratio = get_eye_aspect_ratio(r_eye)

        ratio = (l_ratio + r_ratio)

        l_eyeHull = cv2.convexHull(l_eye)
        r_eyeHull = cv2.convexHull(r_eye)
        cv2.drawContours(frame, [l_eyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [r_eyeHull], -1, (0, 255, 0), 1)

        #TRACKS THE RATIO TO DETERMINE REAL TIME CLASSIFICATION OF EYE
        if ratio < THRESH:
            cv2.putText(frame, "Eye {}".format("close"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            counter += 1
            print(counter)

        else:
            cv2.putText(frame, "Eye {}".format("open"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            counter = 0
            print(counter)

        if counter > 10:
            alarm = True
            wake_up()

    cv2.imshow("Live Preview", frame)

    # EXIT CONDITION
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
#CLEANUP AFTER PROCESSES
cv2.destroyAllWindows()
vs.stop()
