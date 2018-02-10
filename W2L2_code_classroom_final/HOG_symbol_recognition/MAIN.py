import cv2
import numpy as np
import os
import tools as tl
from genSymbolImg import genSymbolImg
import string
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generateData(vis = False):
    chars = string.ascii_uppercase + string.digits
    # chars = string.digits
    font = np.arange(0, 6, 1) #randint(0, 5)
    # size = np.arange(2.5, 4, 0.5) #uniform(2.5, 3.5)
    line_size = np.arange(1, 4, 1) #randint(1, 3)
    blur = np.arange(0, 2, 1) #randint(0, 1)
    kw = np.arange(1, 9, 2) #randint(3, 7)
    kh = np.arange(1, 9, 2) #randint(3, 7)

    generatedImgs = []
    rows = []

    for c in chars:
        print ("Generating - ",c)
        row = []
        for f in font:
            for l in line_size:
                # for b in blur:
                for i in kw:
                    for j in kh:
                        img, _ = genSymbolImg(c, f, l, 1, i, j)
                        # _, croped = segmentSymbolsByContours(img)
                        resized = cv2.resize(img, (30,40))
                        row.append(resized)

                        # cv2.imshow("W", resized)
                        # k = cv2.waitKey(0)
                        # if k == 27:
                        #     exit()

        generatedImgs.append(row)
        rows.append(tl.concat_hor(row))

        # cv2.imshow("ROW", tl.concat_hor(row))
        # k = cv2.waitKey(0)
        # if k == 27:
        #     break

    print ("Generation done")
    generatedVisImg = tl.concat_ver(rows)

    return generatedImgs, generatedVisImg


def segmentSymbolsByContours(img):
    vis = cv2.cvtColor((255 - img), cv2.COLOR_GRAY2BGR)
    gray = 255 - img

    '''
        Binarization
    '''
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    textRegions = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if w * h > 10:
            textRegions.append((x, y, x+w, y+h))
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),1)
            break

    bigImg = tl.concat_ver((vis, gray, th))
    return bigImg, img[y:y+h, x:x+w]


if __name__ == '__main__':
    data, visData = generateData()

    # cv2.imwrite("visData.jpg", visData)

    '''
        Get features
    '''
    win_size = (5, 5)
    nbins = 4  # number of orientation bins
    # cell = (10,10)  # h x w in pixels

    hog = cv2.HOGDescriptor(_winSize=(win_size[0], win_size[1]),
                            _blockSize=(win_size[0], win_size[1]),
                            _blockStride=(win_size[0], win_size[1]),
                            _cellSize=(win_size[0], win_size[1]),
                            _nbins=nbins, _histogramNormType = 0, _gammaCorrection = True)

    features = []
    for row in data:
        rowfd = []
        for s in row:
            rowfd.append(hog.compute(s).reshape((192)))

        features.append(rowfd)

    print (len(features))
    print (len(features[0]))
    print (len(features[0][0]))

    '''
        Prepare data
    '''
    X = np.asarray(features).reshape(-1, 192)
    print (X.shape)

    y = np.repeat(np.asarray([c for c in string.ascii_uppercase + string.digits]), 288)
    print (y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
    
    '''
        Train SVM
    '''
    clf = svm.SVC()
    clf.fit(X_train, y_train)


    '''
        Test
    '''
    # y_pred = clf.predict(X_test)
    # print (accuracy_score(y_test, y_pred))


    '''
        Testing
    '''
    for i in range(10000):
        img, text = genSymbolImg()

        '''
            Generate random data
        '''
        vis, res = segmentSymbolsByContours(img)

        '''
            Make prediction
        '''
        res_r = cv2.resize(res, (30,40))
        fd = np.asarray(hog.compute(res_r)).reshape(-1, 192)
        pred = clf.predict(fd)
        print (pred)


        cv2.imshow("W", vis)
        cv2.imshow("S", res)
        k = cv2.waitKey(0)
        if k == 27:
            break