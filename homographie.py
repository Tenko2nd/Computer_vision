import numpy as np
import cv2 as cv

def homographies(img):
    table_pt = np.array([[19,382], [657,440], [675, 175], [120,-145]])
    tab_h, tab_w, factor = 115, 200, 4
    reel_pt = np.array([[0,tab_h*factor], [tab_w*factor,tab_h*factor], [tab_w*factor,0], [0,0]])
    h, status = cv.findHomography(table_pt, reel_pt)
    im_dst = cv.warpPerspective(img, h, (tab_w*factor,tab_h*factor))
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