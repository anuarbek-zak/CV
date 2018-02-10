import cv2
import numpy as np

'''
	Show small images
'''
# img = cv2.imread("small.png")

# print img.shape
# print len(img.shape)
# print type(img)
# print type(img[0,0,0])

# print img[0:5,0:5,:]

# cv2.imshow("Window", img)
# k = cv2.waitKey(0)


'''
	make grayscal image manually and with opencv
'''
# car = cv2.imread("car.jpg")

# grayManual = car[...,0] * 0.114 + car[...,1] * 0.587 + car[...,2] * 0.299
# print type(grayManual)
# print type(grayManual[0,0])

# # grayManual = np.asarray(grayManual, dtype = np.uint8)
# grayManual = grayManual.astype(np.uint8)
# print type(grayManual[0,0])

# grayOpenCV = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Manual", grayManual)
# cv2.imshow("OpenCV", grayOpenCV)
# cv2.waitKey(0)

'''
	Make white square image with black box inside
'''
# import time

# img = np.full((500,500), 255, dtype = np.uint8)
# st = time.time()

# img[200:300,200:300] = 0

# print time.time() - st

# cv2.imshow("Win", img)
# cv2.waitKey(0)


'''
	Apply filter
'''
# img = cv2.imread("car.jpg")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.ones((5,5), dtype = np.float64) / 25
# output = cv2.filter2D(gray, -1, kernel)

# cv2.imshow("Gray", gray)
# cv2.imshow("Blur", output)
# cv2.waitKey(0)


'''
	Apply gaussian filter
'''
img = cv2.imread("car.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5), dtype = np.float64) / 25
output = cv2.GaussianBlur(gray,(25,25),10)

cv2.imshow("Gray", gray)
cv2.imshow("Blur", output)
cv2.waitKey(0)