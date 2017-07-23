import numpy as np 
import cv2

def detectFeatures(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp

def main():
    img0 = None; img1 = None;
    base = '../KITTIDataset/dataset/sequences/00/images_0/{:06d}.png'

    # Get the first two file names
    f1 = base.format(0)
    f2 = base.format(0)

    # Read Images
    img0 = cv2.imread(f1)
    img1 = cv2.imread(f2)

    # Convert images to grayscale
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    features = detectFeatures(img)
    img_with_features = cv2.drawKeypoints(img, features, color=(255, 0, 0), outImage=None)
    cv2.imshow('image', img_with_features)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

