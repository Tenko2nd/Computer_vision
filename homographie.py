import numpy as np
import cv2 as cv
import Constants as C
from KalmanFilter import KalmanFilter


def homographies(img):
    table_pt = np.array([[19, 382], [657, 440], [675, 175], [120, -145]])
    reel_pt = np.array(
        [[0, C.tab_h * C.factor], [C.tab_w * C.factor, C.tab_h * C.factor], [C.tab_w * C.factor, 0], [0, 0]])
    h, status = cv.findHomography(table_pt, reel_pt)
    im_dst = cv.warpPerspective(img, h, (C.tab_w * C.factor, C.tab_h * C.factor))
    return im_dst


def get_speed(tab_centre):
    speed = None
    if len(tab_centre) >= 2:
        if tab_centre[-2] is not None and tab_centre[-1] is not None:
            dist = np.linalg.norm(np.array(tab_centre[-2]) / C.factor - np.array(
                tab_centre[-1]) / C.factor)  # divide by factor to cenvert pixel in cm
            speed = dist / (C.delta_t * 100)  # Pour passer de cm en m
            speed = round(speed, 2)
    return speed


def get_parabola(tab_centre):
    parab_pts = None
    if len(tab_centre) >= 3:
        if all(centre is not None for centre in tab_centre[-3:]):
            posListX, posListY = [], []
            for centre in tab_centre:
                if centre is not None:
                    posListX.append(centre[0])
                    posListY.append(centre[1])
            coeffs = np.polyfit(posListX, posListY, 2)
            if len(posListX) >= 3:  # Vérification supplémentaire après filtrage des None
                coeffs = np.polyfit(posListX, posListY, 2)
            if coeffs is not None:
                poly = np.poly1d(coeffs)
                x_range = np.linspace(0, C.tab_w * C.factor, 500)
                y_range = poly(x_range)
                parab_pts = np.array([x_range, y_range], dtype=np.int32).T
    return parab_pts


def print_value(img, tab_centre):
    if any(centre is not None for centre in tab_centre):
        parab_pts = get_parabola(tab_centre)
        cv.polylines(img, [parab_pts], False, (255, 0, 255), 3)
        for centre in tab_centre:
            if centre is not None:
                img = cv.circle(img, centre, radius=5, color=(255, 0, 0), thickness=-1)
        speed = get_speed(tab_centre)
        if speed is not None:
            cv.putText(img, str(speed) + "m/s", (tab_centre[-1][0] + 20, tab_centre[-1][1] + 20), color=(0, 0, 255),
                       thickness=2,
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75)
    return img


def apply_kalman(img, kalman_filter, tab_centre):
    if any(centre is not None for centre in tab_centre):
        etat = kalman_filter.predict().astype(np.int32)
        print(etat)
        cv.circle(img, (int(etat[0]), int(etat[1])), 2, (0, 255, 0), 5)
        cv.arrowedLine(img, (int(etat[0]), int(etat[1])), (int(etat[0] + etat[2]), int(etat[1] + etat[3])),
                       color=(0, 255, 0),
                       thickness=3,
                       tipLength=0.2)
        if tab_centre[-1] is not None:
            cv.circle(img, (tab_centre[-1][0], tab_centre[-1][1]), 10, (0, 0, 255), 2)
            kalman_filter.update(np.expand_dims(tab_centre[-1], axis=-1))
    return img


def isolation(img, vid, tab_centre, kalman_filter):
    if vid == "mousse":
        low = np.array([100, 85, 85])
        high = np.array([155, 255, 175])
        min_area = 1225
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)
    elif vid == "rugby":
        low = np.array([10, 150, 40])
        high = np.array([50, 255, 120])
        min_area = 4500
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((4, 4), np.uint8), iterations=3)
    else:
        low = np.array([60, 50, 80])
        high = np.array([170, 150, 210])
        min_area = 600
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)

    # img = cv.bitwise_and(img, img, mask=mask)
    img = apply_moments(mask, img, tab_centre, min_area)
    img = print_value(img, tab_centre)
    if kalman_filter is None and tab_centre[-1] is not None:
        kalman_filter = KalmanFilter(C.delta_t, tab_centre[-1])
    img = apply_kalman(img, kalman_filter, tab_centre)
    return img


def apply_moments(mask, img, tab_centre, min_area):
    moments = cv.moments(mask)
    if moments['m00'] == 0:
        tab_centre.append(None)
        return img

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)
    area = cv.contourArea(cnt)
    if area < min_area:
        tab_centre.append(None)
        return img
    if len(cnt) >= 5:
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(img, ellipse, (0, 255, 0), 2)
    """ Centroid """
    centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    tab_centre.append(centroid)

    return img


def play_vid(cap, vid, tab_centre, kalman_filter):
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            frame = homographies(frame)
            frame = isolation(frame, vid, tab_centre, kalman_filter)
            cv.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv.waitKey(200) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break


if __name__ == '__main__':
    vid_name = 'mousse'
    if vid_name == 'mousse':
        cap = cv.VideoCapture("Ressources/Video/Mousse.mp4")
    elif vid_name == 'rugby':
        cap = cv.VideoCapture("Ressources/Video/Rugby.mp4")
    else:
        vid_name = 'tennis'
        cap = cv.VideoCapture("Ressources/Video/Tennis.mp4")

    KF = None

    tab_centroid = []
    play_vid(cap, vid_name, tab_centroid, KF)

    # When everything done, release
    # the video capture object
    cap.release()
