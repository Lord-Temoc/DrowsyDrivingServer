from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import numpy as nyp
import argparse
import imutils
import time
import dlib
import cv2


def eye_ratio(eye):
    vertical_landmark_A = dist.euclidean(eye[1], eye[5])
    vertical_landmark_B = dist.euclidean(eye[2], eye[4])
    horizontal_landmark_C = dist.euclidean(eye[0], eye[3])
    
    ratio_eye = (vertical_landmark_A + vertical_landmark_B) / (2.0 * horizontal_landmark_C)
    
    return ratio_eye

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
            help="path to input video file")
args = vars(ap.parse_args())

EYE_THRESH = 0.3
EYE_FRAMES = 3

UNCONSIOUS = False

print("info -- loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("info -- starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)        
while True:
    if fileStream and not vs.more():
        break

    frapytme = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_ratio(leftEye)
        rightEAR = eye_ratio(rightEye)

        ratio_eye = (leftEAR + rightEAR) / 2.0


        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        if ratio_eye < EYE_THRESH: #update boolean value to send through API
            UNCONSIOUS = True
            cv2.putText(frame, "UNCONSIOUS!", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("info -- closing...")
            break

cv2.destroyAllWindows()
vs.stop()
