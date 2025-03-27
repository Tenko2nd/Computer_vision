import numpy as np
import math
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


def get_speed():
    speed = None
    tab = [centre for centre in tab_centre if centre is not None]
    if len(tab_centre_pred) >= 1:
        tab.extend(tab_centre_pred)
    if len(tab) >= 2:
        dist = np.linalg.norm(np.array(tab[-2]) / C.factor - np.array(
            tab[-1]) / C.factor)  # divide by factor to convert pixel in cm
        speed = dist / (C.delta_t * 100)  # Pour passer de cm en m
        speed = round(speed, 2)
        if len(tab_centre_pred) >= 1:
            tab_speed_pred.append(speed)
        else:
            tab_speed.append(speed)
    return speed, tab


def get_parabola():
    parab_pts = None
    coeffs = None
    if len(tab_centre) >= 3:
        if sum(1 for centre in tab_centre if centre is not None) >= 3:
            posListX, posListY = [], []
            for centre in tab_centre:
                if centre is not None:
                    posListX.append(centre[0])
                    posListY.append(centre[1])
            if len(posListX) >= 3:  # Vérification supplémentaire après filtrage des None
                coeffs = np.polyfit(posListX, posListY, 2)
            if coeffs is not None:
                if len(tab_coeff_parabole)<1:
                    tab_coeff_parabole.append(coeffs)
                elif (coeffs != tab_coeff_parabole[-1]).any():
                    tab_coeff_parabole.append(coeffs)
                poly = np.poly1d(coeffs)
                x_range = np.linspace(0, C.tab_w * C.factor, 500)
                y_range = poly(x_range)
                parab_pts = np.array([x_range, y_range], dtype=np.int32).T
    return parab_pts


def print_value(img):
    if any(centre is not None for centre in tab_centre):
        parab_pts = get_parabola()
        cv.polylines(img, [parab_pts], False, (255, 0, 255), 3)
        for centre in tab_centre:
            if centre is not None:
                img = cv.circle(img, centre, radius=5, color=(255, 0, 0), thickness=-1)
        for centre_pred in tab_centre_pred:
            if centre_pred is not None:
                img = cv.circle(img, centre_pred, radius=5, color=(0, 255, 0), thickness=-1)
        speed, tab = get_speed()
        if speed is not None:
            cv.putText(img, str(speed) + "m/s", (tab[-1][0] + 20, tab[-1][1] + 20), color=(0, 0, 255),
                       thickness=2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75)
    return img


def apply_kalman(img):
    if sum(1 for centre in tab_centre if centre is not None) > 2 and KF is not None:
        etat = KF.predict().astype(np.int32)
        cv.circle(img, (int(etat[0].item()), int(etat[1].item())), 2, (0, 255, 0), 5)
        cv.arrowedLine(img, (int(etat[0].item()), int(etat[1].item())),
                       (int(etat[0].item() + etat[2].item()), int(etat[1].item() + etat[3].item())),
                       color=(0, 255, 0), thickness=3, tipLength=0.2)
        if tab_centre[-1] is not None:
            KF.update(np.expand_dims(tab_centre[-1], axis=-1))
        else:
            tab_centre_pred.append((etat[0].item(), etat[1].item()))
    return img


def isolation(img, vid):
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
    img = apply_moments(mask, img, min_area)
    img = apply_kalman(img)
    img = print_value(img)
    return img


def create_kalman():
    global KF
    centroid = tab_centre[-1]
    if tab_centre[-2] is not None and tab_centre[-1] is not None:
        speedx = (tab_centre[-1][0] - tab_centre[-2][0]) / C.delta_t
        speedy = (tab_centre[-1][1] - tab_centre[-2][1]) / C.delta_t
        KF = KalmanFilter(C.delta_t, centroid, (speedx, speedy))


def apply_moments(mask, img, min_area):
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

    if sum(1 for centre in tab_centre if centre is not None) == 2 and KF is None:
        create_kalman()
    return img


def play_vid(cap, vid):
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            frame = homographies(frame)
            frame = isolation(frame, vid)
            cv.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv.waitKey(100) & 0xFF == ord('q'):
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

    global KF
    KF = None

    global tab_centre, tab_centre_pred
    tab_centre = []
    tab_centre_pred = []
    global tab_speed, tab_speed_pred
    tab_speed = []
    tab_speed_pred = []
    global tab_coeff_parabole
    tab_coeff_parabole = []

    play_vid(cap, vid_name)

    # When everything done, release the video capture object
    cap.release()

    print("liste centre : ", tab_centre)
    print("liste centre predis: ", tab_centre_pred)
    print("liste vitesse : ", tab_speed)
    print("liste vitesse predis : ", tab_speed_pred)
    print("liste parabole : ", tab_coeff_parabole)
    a,b,c = tab_coeff_parabole[-1]*-1 # car repère inversé par rapport au monde réel
    c = c+460+400 # ajout de la hauteur du tableau et de la hauteur entre le sol et le tableau
    poly = np.poly1d([a,b,c])
    print("\nVoici les résultats pour notre parabole : "
          f"\nLes coefficient finaux de notre parabole sont : A={a:.2E} | B={b:.2E} | C={c:.2E}"
          f"\nL'angle au début du tableau est de : {math.atan(b) * (180/math.pi)}"
          f"\nLa vitesse initiale était de : {tab_speed[0]}m/s"
          f"\nLa balle touchera le sol à {max(poly.roots)/(100*C.factor):.2f}m du début du tableau") #conversion px->cm->m

    print("\nVoici les résultats pour notre filtre de Kahlman : ")
    for centre in tab_centre_pred:
        if centre[1] >= 400+460:
            print(f"La balle touchera le sol à {centre[0] / (100 * C.factor):.2f}m du début du tableau")
            break
