import numpy as np
import math
import cv2 as cv
import Constants as C
from KalmanFilter import KalmanFilter


def homographies(img):
    """
    Effectue l'homographie de l'image
    :param img: image avant homographie
    :return: image après homographie
    """
    # coordonnées des coins du tableau en pixel sur l'image en param
    table_pt = np.array([[19, 382], [657, 440], [675, 175], [120, -145]])
    # coordonnées des coins du tableau aux coin de l'images (en veux tablaeu en plein écran)
    reel_pt = np.array(
        [[0, C.tab_h * C.factor], [C.tab_w * C.factor, C.tab_h * C.factor], [C.tab_w * C.factor, 0], [0, 0]])
    h, status = cv.findHomography(table_pt, reel_pt)
    im_dst = cv.warpPerspective(img, h, (C.tab_w * C.factor, C.tab_h * C.factor))
    return im_dst


def get_speed():
    """
    Calcul la vitesse entre les deux derniers centre de la balle et stocke la vitesse dans un tableau
    :return: la vitesse en m/s
    """
    speed = None
    # retire les centres None (ceux avant et après que la balle soit visible)
    tab = [centre for centre in tab_centre if centre is not None]
    # ajoute les centres prédis selon kalman
    if len(tab_centre_pred) >= 1:
        tab.extend(tab_centre_pred)
    # vérifie qu'il y ai au moins deux centres
    if len(tab) >= 2:
        # divise par le fecteur pour passer de pixel en cm
        dist = np.linalg.norm(np.array(tab[-2]) / C.factor - np.array(tab[-1]) / C.factor)
        speed = dist / (C.delta_t * 100)  # Pour passer de cm en m
        speed = round(speed, 2)
        # sauvegarde les vitesse dans un tableau de prédiction si centre prédis sinon dans un autre tableau
        if len(tab_centre_pred) >= 1:
            tab_speed_pred.append(speed)
        else:
            tab_speed.append(speed)
    return speed, tab


def get_parabola():
    """
    Récupère les points qui forme une parabole en se basant sur les centres de la balles
    Il doit y avoir au moins 3 centres
    :return: les points formant la parabole
    """
    parab_pts = None
    coeffs = None
    # si il y a au moins trois centre ...
    if len(tab_centre) >= 3:
        # et que au moins trois ne sont pas None ...
        if sum(1 for centre in tab_centre if centre is not None) >= 3:
            posListX, posListY = [], []
            # Créer deux tableau qui contiennent les coordonnées x et un autre avec les y des centre non None
            for centre in tab_centre:
                if centre is not None:
                    posListX.append(centre[0])
                    posListY.append(centre[1])
            if len(posListX) >= 3:  # Vérification supplémentaire après filtrage des None
                coeffs = np.polyfit(posListX, posListY, 2)  # Récupération des coefficients de la parabole
            if coeffs is not None:
                # ajoute les coefficients dans un tableau pour pouvoir avoir l'évolution de la parabole
                if len(tab_coeff_parabole)<1:
                    tab_coeff_parabole.append(coeffs)
                elif (coeffs != tab_coeff_parabole[-1]).any():  # Seuleument s'il y a eu un changement
                    tab_coeff_parabole.append(coeffs)
                poly = np.poly1d(coeffs)
                x_range = np.linspace(0, C.tab_w * C.factor, 500)
                y_range = poly(x_range)
                parab_pts = np.array([x_range, y_range], dtype=np.int32).T
    return parab_pts


def print_value(img):
    """
    Affiche les valeur et les informations visuelles sur l'image
    :param img: image avant écriture des valeurs
    :return: image après écriture des valeurs
    """
    if any(centre is not None for centre in tab_centre):  # S'il y a matière à écrire
        parab_pts = get_parabola()
        if parab_pts is not None:  # Si une parabole a été trouvé, l'affiche
            cv.polylines(img, [parab_pts], False, (255, 0, 255), 3)
        for centre in tab_centre:  # affiche tous les centres précédents de la balle
            if centre is not None:
                img = cv.circle(img, centre, radius=5, color=(255, 0, 0), thickness=-1)
        for centre_pred in tab_centre_pred:  # affiche les centres prédis par Kalman
            if centre_pred is not None:
                img = cv.circle(img, centre_pred, radius=5, color=(0, 255, 0), thickness=-1)
        speed, tab = get_speed()
        if speed is not None: # Si on a une vitesse, l'affiche à coté du centre de la balle
            cv.putText(img, str(speed) + "m/s", (tab[-1][0] + 20, tab[-1][1] + 20), color=(0, 0, 255),
                       thickness=2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75)
    return img


def apply_kalman(img):
    """
    prédis le centre futur de la balle en appliquant un filtre de Kalman et affiche la direction prédite
    :param img: image avant prédiction du centre et de la derection de la balle
    :return: image après prédiction des centres et de la direction de la balle
    """
    if sum(1 for centre in tab_centre if centre is not None) > 2 and KF is not None:  # S'il existe au moins 2 centres
        etat = KF.predict().astype(np.int32)  # prédit le prochain centre et la direction
        cv.circle(img, (int(etat[0].item()), int(etat[1].item())), 2, (0, 255, 0), 5)
        cv.arrowedLine(img, (int(etat[0].item()), int(etat[1].item())),
                       (int(etat[0].item() + etat[2].item()/(C.factor*3)), int(etat[1].item() + etat[3].item()/(C.factor*3))),
                       color=(0, 255, 0), thickness=3, tipLength=0.2)
        if tab_centre[-1] is not None:  # Si le centre actuel existe (la balle est encore visible
            KF.update(np.expand_dims(tab_centre[-1], axis=-1))  # met a jour les valeur du filtre de Kalman
        else:
            tab_centre_pred.append((etat[0].item(), etat[1].item()))  # Sinon ajoute un centre prédit
    return img


def isolation(img, vid):
    """
    Isole la balle selon la video choisi et applique les différents effets
    :param img: image avant l'application des différents effets
    :param vid: la video choisi
    :return: l'image après l'application des effets
    """
    if vid == "mousse":
        low = np.array([100, 85, 85])
        high = np.array([155, 255, 175])
        min_area = 124 * math.pow(math.pi, math.log2(C.factor))  # ajustement automatique peu fiable a cause homographie
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)
    elif vid == "rugby":
        low = np.array([10, 150, 40])
        high = np.array([50, 255, 120])
        min_area = 455 * math.pow(math.pi, math.log2(C.factor))  # ajustement automatique peu fiable a cause homographie
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((4, 4), np.uint8), iterations=3)
    else:
        low = np.array([60, 50, 80])
        high = np.array([170, 150, 210])
        min_area = 61 * math.pow(math.pi, math.log2(C.factor))  # ajustement automatique peu fiable a cause homographie
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.dilate(mask, kernel=np.ones((2, 2), np.uint8), iterations=4)

    # img = cv.bitwise_and(img, img, mask=mask)
    img = apply_moments(mask, img, min_area)
    img = apply_kalman(img)
    img = print_value(img)
    return img


def create_kalman():
    """
    Créer une instance de filtre de Kalman global en prenant en compte le centre et la vitesse actuelle
    """
    global KF
    centroid = tab_centre[-1]
    if tab_centre[-2] is not None and tab_centre[-1] is not None:
        speedx = (tab_centre[-1][0] - tab_centre[-2][0]) / C.delta_t
        speedy = (tab_centre[-1][1] - tab_centre[-2][1]) / C.delta_t
        KF = KalmanFilter(C.delta_t, centroid, (speedx, speedy))


def apply_moments(mask, img, min_area):
    """
    Troue les moments du mask et créer un centroide et une boite englobante
    :param mask: Le masque de la balle isolée
    :param img: l'image avant l'ajout de la boite englobante
    :param min_area: l'aire minimal de la balle
    :return: l'image après l'ajout de la boite englobante
    """
    moments = cv.moments(mask)
    if moments['m00'] == 0:  # si rien est detecté returne l'image telle quelle
        tab_centre.append(None)
        return img

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # trouve le contour ...
    cnt = max(contours, key=cv.contourArea)  # maximal...
    area = cv.contourArea(cnt)  # et son aire
    if area < min_area:  # si l'aire n"est pas assez grande (la balle n'est pas entièrement visible), ne fait rien
        tab_centre.append(None)
        return img
    if len(cnt) >= 5:  # créer une boite englobante de forme ellipsoidale
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(img, ellipse, (0, 255, 0), 2)
    # troue le centroid de la balle etl'ajoute au tableau des centres
    centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    tab_centre.append(centroid)
    # Si le deuxième centre vient d'être trouvé, créer un filtre de Kalman
    if sum(1 for centre in tab_centre if centre is not None) == 2 and KF is None:
        create_kalman()
    return img


def play_vid(cap, vid):
    """
    lit la video
    :param cap: les images de la video
    :param vid: le nom de la video
    """
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = homographies(frame)
            frame = isolation(frame, vid)
            # Display the resulting frame
            cv.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv.waitKey(250) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break


if __name__ == '__main__':
    # choisir un no de video pour la selectionner
    vid_name = 'tennis'
    if vid_name == 'mousse':
        cap = cv.VideoCapture("Ressources/Video/Mousse.mp4")
    elif vid_name == 'rugby':
        cap = cv.VideoCapture("Ressources/Video/Rugby.mp4")
    else:
        vid_name = 'tennis'
        cap = cv.VideoCapture("Ressources/Video/Tennis.mp4")

    # initialise un filtre de Kalman vide
    global KF
    KF = None

    # initialise tous les tableau de résultats
    global tab_centre, tab_centre_pred
    tab_centre = []
    tab_centre_pred = []
    global tab_speed, tab_speed_pred
    tab_speed = []
    tab_speed_pred = []
    global tab_coeff_parabole
    tab_coeff_parabole = []

    # joue la video
    play_vid(cap, vid_name)

    # When everything done, release the video capture object
    cap.release()

    """ ******* Les outputs et résultats ******* """
    print("liste centre : ", tab_centre)
    print("liste centre predis: ", tab_centre_pred)
    print("liste vitesse : ", tab_speed)
    print("liste vitesse predis : ", tab_speed_pred)
    print("liste parabole : ", tab_coeff_parabole)
    a, b, c = tab_coeff_parabole[-1]*-1  # car repère inversé par rapport au monde réel (prend la dernière parabole)
    c = c+115*C.factor+100*C.factor  # ajout de la hauteur du tableau et de la hauteur entre le sol et le tableau (en pixel)
    poly = np.poly1d([a, b, c])
    print("\nVoici les résultats pour notre parabole : "
          f"\nLes coefficient finaux de notre parabole sont : A={a:.2E} | B={b:.2E} | C={c:.2E}"
          f"\nL'angle au début du tableau est de : {math.atan(b) * (180/math.pi)}"
          f"\nLa vitesse initiale était de : {tab_speed[0]}m/s"
          f"\nLa balle touchera le sol à {max(poly.roots)/(100*C.factor):.2f}m du début du tableau")  #conversion px->cm->m

    for centre in tab_centre_pred:
        if centre[1] >= 115*C.factor+100*C.factor:
            print("\nVoici les résultats pour notre filtre de Kahlman : ")
            print(f"La balle touchera le sol à {centre[0] / (100 * C.factor):.2f}m du début du tableau")
            break
