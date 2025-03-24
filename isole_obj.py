import numpy as np
import cv2 as cv

def isole_mousse():
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

def isole_rugby():
    cap = cv.VideoCapture("Ressources/Video/Rugby.mp4")
    low_rugby = np.array([10,150,40])
    high_rugby = np.array([50,255,120])
    list_angle=[]
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #cv.imshow('frame', frame)
        hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low_rugby, high_rugby)
        mask = cv.dilate(mask, kernel=np.ones((4, 4), np.uint8), iterations=3)
        res = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('res', res)
        if cv.waitKey(200) == ord('q'):
            break
    cap.release()

def isole_tennis():
    cap = cv.VideoCapture("Ressources/Video/Tennis.mp4")
    low_tennis = np.array([60, 50, 80])
    high_tennis = np.array([170, 150, 210])
    list_angle = []
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # cv.imshow('frame', frame)
        hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low_tennis, high_tennis)
        mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)
        res = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('res', res)
        if cv.waitKey(200) == ord('q'):
            break
    cap.release()