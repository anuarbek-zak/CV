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

            '''
                finding gradient by Sobel filter
            '''
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            useGradient = 0

            if useGradient:
                gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)

                '''
                    take absolute value of gradient to use negative gradient
                '''
                gradX = np.absolute(gradX)

                '''
                    Normalization of gradient
                '''
                (minVal, maxVal) = (np.min(gradX), np.max(gradX)) 
                if maxVal - minVal > 0:
                    gradX = (255 * ((gradX - minVal) / float(maxVal - minVal))).astype("uint8")
                else:
                    gradX  = np.zeros(gray.shape, dtype = "uint8")
            else:
                gradX = 255 - gray

            '''
                Take median filter by horizontal axis
            '''
            blur = cv2.blur(gradX,(21,1))


            '''
                Binarization
            '''
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            '''
                Morphological opening
            '''
            kernel = np.ones((1,15), np.uint8)
            opening = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            '''
                Find contours
            '''
            im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0,255,0), 1)

            boxRegions = []
            hsum = 0
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                hsum += h

                boxRegions.append((x, y, w, h))


            '''
                Choose only big countours
            '''
            meanHeight = hsum / float(len(contours))

            textRegions = []
            for box in boxRegions:
                if box[3] > meanHeight:
                    x,y,w,h = box
                    textRegions.append((x,y, x+w,y+h))
                    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),1)


            '''
                Extract all text regions
            '''
            regionImgs = []
            for textRegion in textRegions:
                x1,y1,x2,y2 = textRegion
                regionImgs.append(img[y1:y2,x1:x2])


            scale = 0.8

            row1 = tl.concat_hor((vis, gradX, blur))
            row2 = tl.concat_hor((th, opening))
            allTexts = tl.concat_ver(regionImgs)
            bigImg = tl.concat_ver((row1, row2))


            cv2.imshow("Window1", cv2.resize(bigImg, (0,0), fx = scale, fy = scale))
            cv2.imshow("Window2", cv2.resize(allTexts, (0,0), fx = 1.4, fy = 1.4))
            k = cv2.waitKey(0)

            if k == 27:
                exit()
            elif k == ord('q'):
                exit()