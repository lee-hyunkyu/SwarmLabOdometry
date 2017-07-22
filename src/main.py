import numpy as np 
import cv2

def detectFeatures(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp

def main():
    img = cv2.imread('../KITTIDataset/dataset/sequences/00/image_0/000000.png');

    features = detectFeatures(img)
    img_with_features = cv2.drawKeypoints(img, features, color=(255, 0, 0), outImage=None)
    cv2.imshow('image', img_with_features)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

