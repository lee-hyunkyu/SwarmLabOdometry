from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
def test():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/ansel.jpg')
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/noisy.png')
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/chessboard.jpg')
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/lena.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #rgb2gray(image)
    #image = cv2.medianBlur(image, 7)
    corners = detect(image)
    implot = plt.imshow(image, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10)
    plt.show()

def testMedianBlur():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/lena.png')
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = shape(image)
    startSearchRow = int(0.25*rows)
    endSearchRow = int(0.75*rows)
    startSearchCol = int(0.25*cols)
    endSearchCol = int(0.75*cols)
    image = medianBlur(image, startSearchRow, endSearchRow, startSearchCol, endSearchCol)
    implot = plt.imshow(image, cmap='gray')
    plt.show()

def testgray():
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/balloons_noisy.png')
    rgb2gray(image)
    implot = plt.imshow(image, cmap='gray')
    plt.show()

#testgray()
testMedianBlur()
#test()
