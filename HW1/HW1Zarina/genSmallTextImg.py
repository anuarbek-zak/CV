import cv2
import numpy as np
from random import randint, uniform
import string, random


def addNoise(image):    
    row,col = image.shape
    s_vs_p = 0.4
    amount = 0.01
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out


def addLines(img):
    for i in range(randint(0,2)):
        y1 = randint(0, img.shape[0])
        y2 = randint(0, img.shape[0])
        cv2.line(img, (0, y1), (img.shape[1], y2), 0, 1)


def addBlur(img):
    kw = randint(3, 7)
    kh = randint(3, 7)

    return cv2.blur(img, (kw, kh))


def text_generator(size = 8, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def addText(img):
    font = randint(0, 5)
    size = uniform(2.5, 3.5)
    text = text_generator(randint(5, 10))
    line_size = randint(1, 3)

    cv2.putText(img, text, (10, img.shape[0] - 15), font, size, (0, 0, 255), line_size, cv2.LINE_AA)

    return text


def genSmallTextImg(lines = False):
    genImg = np.full((100, 800), 255, dtype= np.uint8)

    text = addText(genImg)

    if randint(0, 1):
        genImg = addNoise(genImg)
        
    if lines:
        addLines(genImg)

    if randint(0, 1):
        genImg = addBlur(genImg)


    return genImg, text



if __name__ == '__main__':

    for i in range(10000):
        img, text = genSmallTextImg()
        print (text)

        cv2.imshow("W", img)
        k = cv2.waitKey(0)
        if k == 27:
            break