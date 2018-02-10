# task 2 part 1

import cv2
import numpy as np
import tools as tl
import genSmallTextImg as gsti

img, text = gsti.genSmallTextImg()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.blur(img,(21,1))
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_inv = 255 - img
filtered = cv2.blur(img_inv,(5,5))
ret_o, th_o = cv2.threshold(filtered, 0, 255,  cv2.THRESH_OTSU)
im2, countours, hier = cv2.findContours(th_o, 
										cv2.RETR_EXTERNAL, 
										cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis,countours,-1,(255,0,0),1)


hsum = 0
for cnt in countours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	hsum+=h

meanH = hsum/float(len(countours))

for cnt in countours:
	(x,y,w,h) = cv2.boundingRect(cnt)
	print (h)
	if(h>meanH-meanH*0.1):
		cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)

final = tl.concat_ver((img,vis))
cv2.imshow("Sudoku",final)
cv2.waitKey(0)