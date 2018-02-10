import cv2
import os
import numpy as np
import tools as tl
# import matplotlib.pyplot as plt

img = cv2.imread('../Data/1.jpg',0)
vis = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img_inv = 255 - img
filtered = cv2.blur(img_inv,(21,1))
ret_o, th_o = cv2.threshold(filtered, 0, 255,  cv2.THRESH_OTSU)
closing = cv2.morphologyEx(th_o,cv2.MORPH_CLOSE,np.ones((1,15), np.uint8))
im2,contours,hier = cv2.findContours(closing,cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis,contours,-1,(0,0,255),1)


boxRegions = []
hsum = 0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	hsum += h
	boxRegions.append((x, y, w, h))


meanHeight = hsum / float(len(contours))

textRegions = []
for box in reversed(boxRegions):
	if box[3] > meanHeight:
		x,y,w,h = box
		textRegions.append((x,y, x+w,y+h))
		cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1)

textImg = []
for i in textRegions:
	x1,y1,x2,y2 = i
	textImg.append(img[y1:y2,x1:x2])

allTexts = tl.concat_ver(textImg)



row1 = tl.concat_hor((img,img_inv,filtered))
row2 = tl.concat_hor((th_o,closing,vis,allTexts))

# final = tl.concat_ver((row2))
cv2.imshow('myimg',cv2.resize(row2,(0,0),fx = 1,fy=1))
cv2.waitKey(0)

# for root, dirs, files in os.walk("../Data/"):
#     for file in files:
#         imagepath = os.path.join(root, file)
#         if file.endswith('.jpg'):

#             img = cv2.imread(imagepath)

#             '''
#                 ToDo: find text areas
#             '''



#             scale = 0.8
#             cv2.imshow("Window2", cv2.resize(img, (0,0), fx = scale, fy = scale))
#             k = cv2.waitKey(0)

#             if k == 27:
#                 exit()
#             elif k == ord('q'):
#                 exit()