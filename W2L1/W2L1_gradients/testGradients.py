import cv2
import numpy as np
import matplotlib.pyplot as plt

# gray = np.full((200,200), 255, dtype = np.uint8)
# gray[50:100, 50:100] = 0

gray = cv2.imread("car.jpg", 0)

# gradX = cv2.Laplacian(gray, cv2.CV_8U)

# gradX = cv2.Sobel(gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
gradX = np.abs(gradX)

(minVal, maxVal) = (np.min(gradX), np.max(gradX)) 
if maxVal - minVal > 0:
    gradX = (255 * ((gradX - minVal) / float(maxVal - minVal))).astype("uint8")
else:
    gradX  = np.zeros(gray.shape, dtype = "uint8")

cv2.imshow("IMG", gray)
cv2.imshow("IMG2", gradX)
cv2.waitKey(0)