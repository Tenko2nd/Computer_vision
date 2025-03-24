import numpy as np
import cv2 as cv

def homographies(img):
    table_pt = np.array([[19,382], [657,440], [675, 175], [120,-145]])
    reel_pt = np.array([[0,600], [800,600], [800,0], [0,0]])
    h, status = cv.findHomography(table_pt, reel_pt)
    im_dst = cv.warpPerspective(img, h, (800,600))
    return im_dst

cap = cv.VideoCapture("Ressources/Video/Mousse.mp4")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        frame = homographies(frame)
        cv.imshow('Frame', frame)
        # Press Q on keyboard to exit
        if cv.waitKey(200) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()