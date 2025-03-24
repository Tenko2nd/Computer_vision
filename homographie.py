import numpy as np
import cv2 as cv

def homographies(img):
    table_pt = np.array([[19,382], [657,440], [675, 175], [120,-145]])
    tab_h, tab_w, factor = 115, 200, 4
    reel_pt = np.array([[0,tab_h*factor], [tab_w*factor,tab_h*factor], [tab_w*factor,0], [0,0]])
    h, status = cv.findHomography(table_pt, reel_pt)
    im_dst = cv.warpPerspective(img, h, (tab_w*factor,tab_h*factor))
    return im_dst

def isolation(img):
    low_mousse = np.array([100,85,85])
    high_mousse = np.array([155,255,175])
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, low_mousse, high_mousse)
    mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)
    res = cv.bitwise_and(img, img, mask=mask)
    res = apply_moments(mask, res)
    return res

def apply_moments(mask, img):
    moments = cv.moments(mask)
    if moments['m00'] == 0:
        return img
    """ Centroid """
    centroid = (int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00']))
    img = cv.circle(img, centroid, radius=5, color=(255, 0, 0), thickness=-1)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)
    ellipse = cv.fitEllipse(cnt)
    cv.ellipse(img,ellipse,(0,255,0),2)

    return img

cap = cv.VideoCapture("Ressources/Video/Mousse.mp4")
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        frame = homographies(frame)
        frame = isolation(frame)
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