from fast import *
from matplotlib import pyplot as plt
import cv2
def test():
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/ansel.jpg');
    image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/noisy.png');
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/square.jpg');
    #image = cv2.imread('/Users/timmytimmyliu/research/odometry/test_images/chessboard.jpg');
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = detect(image)
    implot = plt.imshow(image, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10)
    plt.show()

test()
