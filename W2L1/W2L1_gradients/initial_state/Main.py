import cv2
import os
import numpy as np
import tools as tl
# import matplotlib.pyplot as plt

for root, dirs, files in os.walk("../Data/"):
    for file in files:
        imagepath = os.path.join(root, file)
        if file.endswith('.jpg'):
                
            img = cv2.imread(imagepath)
            vis = img.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.line(gray, (0,0), (gray.shape[1], gray.shape[0]), 0, 2)
            gray = 255 - gray
                
            verP = np.sum(gray, axis = 1)/255
            verPVis = tl.getDrawProjectionVer(vis, verP)

            gradX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
            gradX = np.abs(gradX)

            (minVal, maxVal) = (np.min(gradX), np.max(gradX)) 
            if maxVal - minVal > 0:
                gradX = (255 * ((gradX - minVal) / float(maxVal - minVal))).astype("uint8")
            else:
                gradX  = np.zeros(gray.shape, dtype = "uint8")

            verPX = np.sum(gradX, axis = 1)/255
            verPXVis = tl.getDrawProjectionVer(vis, verPX)


            row1 = tl.concat_hor((gray, verPVis))
            row2 = tl.concat_hor((gradX, verPXVis))
            bigImg = tl.concat_ver((row1, row2))
            scale = 0.8
            cv2.imshow("Window2", cv2.resize(bigImg, (0,0), fx = scale, fy = scale))
            cv2.moveWindow("Window2", 0, 0)
            k = cv2.waitKey(0)

            if k == 27:
                exit()
            elif k == ord('q'):
                exit()