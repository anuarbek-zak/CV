import cv2
import numpy as np
import tools as tl

img = cv2.imread("morphology2.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
print ret

kernel = np.ones((5,5), dtype = np.uint8)

# erosion = cv2.erode(th, kernel, iterations = 1)
# dilation = cv2.dilate(erosion, kernel, iterations = 1)

closing = cv2.morphologyEx(th, cv2.MORPH_OPEN, 
						   kernel, iterations = 2)

im2, countours, hier = cv2.findContours(closing, 
										cv2.RETR_EXTERNAL, 
										cv2.CHAIN_APPROX_SIMPLE)

print len(countours)

vis = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, countours, -1, (0,255,0), -1)

for cnt in countours:
	x,y,w,h = cv2.boundingRect(cnt)

	cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)


final_img = tl.concat_ver((img, th, vis))
scale = 0.9
cv2.namedWindow("Final", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Final", cv2.resize(final_img, (0,0), 
			fx = scale, fy = scale))
cv2.waitKey(0)