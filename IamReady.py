import cv2
import numpy as np

img = np.zeros((300, 300, 3), dtype = np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "I'm ready!!!", (20, 60), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(img, "CVT", (60, 180), font, 3, (255, 255, 0), 4, cv2.LINE_AA)
cv2.putText(img, "academy", (40, 250), font, 1.5, (255, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("I am Ready", img)
cv2.waitKey(0)