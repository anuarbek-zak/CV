import cv2
import numpy as np
import tools as tl
import matplotlib.pyplot as plt


def getGradient(gray, x = 0, y = 0, useGradient = True):
    if useGradient:
        grad = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=x, dy=y, ksize=3)

        '''
            take absolute value of gradient to use negative gradient
        '''
        grad = np.absolute(grad)

        '''
            Normalization of gradient
        '''
        (minVal, maxVal) = (np.min(grad), np.max(grad)) 
        if maxVal - minVal > 0:
            grad = (255 * ((grad - minVal) / float(maxVal - minVal))).astype("uint8")
        else:
            grad  = np.zeros(gray.shape, dtype = "uint8")

    else:
        grad = cv2.adaptiveThreshold(  gray,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        11,
                                        2)

    return grad


# star

img = cv2.imread("1.jpg")
vis = img.copy()

'''
    finding gradient by Sobel filter
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.line(gray, (0,0), (gray.shape[1], gray.shape[0]), 0, 2)

grad1 = getGradient(gray, x = 1, useGradient = True)


'''
    Calculate vertical projection
'''
verp = np.sum(grad1, axis = 1) / 255
drawedverp = tl.getDrawProjectionVer(vis, verp)

'''
    Find median height of text of vertical projection
'''
half = int(np.max(verp) / 2)
sliceLine = drawedverp[:,(half-1):(half+1)]
im2, contours, hierarchy = cv2.findContours(cv2.cvtColor(   sliceLine, 
                                                            cv2.COLOR_BGR2GRAY), 
                                                            cv2.RETR_EXTERNAL, 
                                                            cv2.CHAIN_APPROX_SIMPLE)

heights = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    heights.append(h)

medianHeight = int(np.median(np.asarray(heights)) * 1.5)
print ("medianHeight", medianHeight)

cv2.line(drawedverp, 
        (half,0), 
        (half,drawedverp.shape[0]), 
        (0,0,255), 
        1)

'''
    Convolve vertical projection
'''
kernel = medianHeight
verpConvolved = np.convolve(verp, 
                            np.ones((kernel,))/kernel, 
                            mode='same')

drawedverpConvolved = tl.getDrawProjectionVer(vis, verpConvolved)


'''
    Find peaks Band Clipping Phase 1
'''
bandP1ranges = []
peaks = []
c1 = 0.2
c2 = 0.4
while np.max(verpConvolved) > 10:
    ybm = np.argmax(verpConvolved)

    yb0 = tl.findb0(verpConvolved, 
                    ybm, 
                    c1 * verpConvolved[ybm])
    yb1 = tl.findb1(verpConvolved, 
                    ybm, 
                    c2 * verpConvolved[ybm])

    if yb1 - yb0 > medianHeight:
        bandP1ranges.append((yb0,yb1))
        peaks.append((int(verpConvolved[ybm]), ybm))

    verpConvolved[yb0:yb1] = 0

# draw peaks
for peak in peaks:
    cv2.circle(drawedverpConvolved, peak, 2, (0,0,255), -1)

# draw bands
bandsImg = np.zeros(vis.shape, dtype = np.uint8)
for band in bandP1ranges:
    yt, yb = band
    bandsImg[yt:yb] = [0,255,0]

vis = cv2.addWeighted(vis, 0.6, bandsImg, 0.4, 0)


'''
    Find peaks Row Clipping Phase 2
'''

# TODO crop words using vertical projections


'''
    Final drawing
'''
scale1 = 0.8
bigImg1 = tl.concat_hor((vis, grad1, drawedverp, sliceLine, drawedverpConvolved))
cv2.imshow("Window1", cv2.resize(bigImg1, (0,0), fx = scale1, fy = scale1))


k = cv2.waitKey(0)

if k == 27:
    exit()
elif k == ord('q'):
    exit()