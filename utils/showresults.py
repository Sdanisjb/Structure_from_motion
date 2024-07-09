import cv2 as cv
import numpy as np

def showKeypoints(img, kpts, pause=False):
    # Mostrar los resultados de cada extracción de features
    img_marked = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv.drawKeypoints(img, kpts, img_marked)
    cv.namedWindow("Keypoints", cv.WINDOW_NORMAL)
    cv.resizeWindow("Keypoints", 960,540)
    cv.imshow("Keypoints", img_marked)

    if pause:
        cv.waitKey()

def showMatches(img1, img2, kpts1, kpts2, matches):
    #Visualizar matches en cada imagen
    res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1],3), dtype=np.uint8)
    cv.drawMatches(img1, kpts1, img2, kpts2, matches, res)
    res = cv.resize(res, (1560,540), interpolation= cv.INTER_LINEAR)
    cv.imshow("result", res)
    cv.waitKey()