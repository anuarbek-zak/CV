import cv2
import numpy as np
import os
import tools as tl

from skimage import feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing

show = False
labels = []
with open('dataset/dataset/annotation.txt', 'r') as f:
    data = f.readlines()
    names = [x.split(' ')[0] for x in data]
    targets = [x.split(' ')[1] for x in data]

le = preprocessing.LabelEncoder()


main_data = []
hog_features = []
hog_targets = []
counter = 0
for root, dirs, files in os.walk("dataset/dataset/images"):
    for file in files:
        counter+=1
        if(counter>5000):
            break
        imagepath = os.path.join(root, file)
        if file.endswith('.jpg'):
            img = cv2.imread(imagepath,0)
            tmp = [img, targets[names.index(file)]]
            H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
     transform_sqrt=True)
            print (counter,H,file,tmp[1],'\n')
            hog_features.append(H)
            hog_targets.append(tmp[1])
            main_data.append(tmp)
            if show:
                scale = 0.8
                cv2.imshow("Window2", cv2.resize(img, (0,0), fx = scale, fy = scale))
                k = cv2.waitKey(0)

                if k == 27:
                    exit()
                elif k == ord('q'):
                    exit()

le.fit(hog_targets)
hog_targets = le.transform(hog_targets)
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, 
    hog_targets,
     test_size=0.3)
clf = svm.SVC()
clf.fit(X_train, y_train)
print ('trained')
print ('predicting ...')
prediction = clf.predict(X_test)

print("accuracy:", accuracy_score(y_test,prediction))
print("precision:", precision_score(y_test,prediction, average=None))
print("recall:", recall_score(y_test,prediction, average=None))
print("f1:", f1_score(y_test,prediction, average=None))

# Main data with [img = (224,224,3), target]
# print(main_data[0])