import cv2
import numpy as np

def _clean(mask255):
    k = np.ones((3,3), np.uint8)
    mask255 = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, k, iterations=1)
    mask255 = cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, k, iterations=2)
    return mask255

def otsu_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(th)

def kmeans_mask(img_bgr, k=2, use_lab=True):
    img = cv2.medianBlur(img_bgr, 3)
    if use_lab:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    Z = x.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(img.shape[:2])

    if use_lab:
        # L канали кичик бўлган кластерни объект деб фараз (қоронғироқ объект)
        obj = int(np.argmin(centers[:,0]))
    else:
        # HSV: S катта бўлган кластер объект (оддий фараз)
        obj = int(np.argmax(centers[:,1]))

    m = (labels == obj).astype(np.uint8) * 255
    return _clean(m)
