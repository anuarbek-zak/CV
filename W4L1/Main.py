import cv2
import numpy as np
import tools as tl
import os

# for root, dirs, files in os.walk("data"):
# 	for file in files:
# 		if file.endswith('.txt'):
# 			txtpath = os.path.join(root, file)
# 			with open(txtpath) as f:
# 				data = f.readlines()
# 				print (data,txtpath)

img = cv2.imread('data/41.jpg')
f = open('data/41_c.txt','r')
print(img.shape[1])
points1 = []
for line in f:
	x = int(line.split(' ')[0])
	y = int(line.split(' ')[1])
	points1.append([x,y])

points1 = np.float32(points1)
points2 = np.float32(([0,0],[img.shape[0],0],[0,img.shape[1]],[img.shape[0],img.shape[1]]))

M = cv2.getPerspectiveTransform(points1,points2)
new_img = cv2.warpPerspective(img,M,(1000,800))
vis = new_img.copy()

gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

filtered = cv2.blur(gray,(5,5))
ret_o, th_o = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY)
im2, countours, hier = cv2.findContours(th_o, 
										cv2.RETR_EXTERNAL, 
										cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(vis,countours,-1,(255,0,0),1)

hsum = 0
for cnt in countours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	hsum+=h

meanH = hsum/float(len(countours))

for cnt in countours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)


# cv2.imshow('img',img)
# cv2.imshow('new_img',new_img)
scale = 0.6
cv2.imshow('vis',cv2.resize(vis, (0,0), 
			fx = scale, fy = scale))
cv2.waitKey(0)