import cv2
import numpy as np
import tools as tl

img = cv2.imread("sudoku.png", 0)

ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, th_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

th_adap11 = cv2.adaptiveThreshold(img, 255, 
								cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
								cv2.THRESH_BINARY, 11, 2)

th_adap5 = cv2.adaptiveThreshold(img, 255, 
								cv2.ADAPTIVE_THRESH_MEAN_C, 
								cv2.THRESH_BINARY, 11, 2)

ret_o, th_o = cv2.threshold(img, 0, 255, 
							cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print (ret_o)
print (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

row1 = tl.concat_hor((img, th, th_o))
row2 = tl.concat_hor((img, th_adap11, th_adap5))
final_img = tl.concat_ver((row1, row2))
cv2.imshow("Sudoku", final_img)
cv2.waitKey(0)