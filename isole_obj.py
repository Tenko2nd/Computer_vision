import numpy as np
import cv2 as cv


cap = cv.VideoCapture("Ressources/Video/Mousse.mp4")
low_mousse = np.array([100,85,85])
high_mousse = np.array([155,255,175])
list_angle=[]
while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #cv.imshow('frame', frame)
    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, low_mousse, high_mousse)
    mask = cv.dilate(mask, kernel=np.ones((3, 3), np.uint8), iterations=3)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('res', res)
    if cv.waitKey(200) == ord('q'):
        break
cap.release()