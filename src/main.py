import numpy as np 
import cv2
import pdb

def detectFeatures(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp

def trackFeatures(img0, img1, p0):
    ''' Tracks features from img0 to img1 using LK Tracking 
        Assumes p0 is a numpy array of <KeyPoint>
    '''
    # Change shape/format of p0 for calcOpticalFlowPyrLK
    p0 = np.float32([[p.pt[0], p.pt[1]] for p in p0]).reshape(-1, 1, 2)
    p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None)

    # Reshape p1, status, err
    p1      = p1[:,0]
    status  = status[:,0]
    err     = err[:,0] 

    # Modify status; marks points tracked off screen to be invalid
    status = np.array([s if (p[0] > 0 and p[1] > 1) else 0 for s, p in zip(status, p1)])

    # Filter the points that failed to be tracked
    p1 = [p if s else None for s, p in zip(status, p1)]
    p1 = filter(lambda p: p is not None, p1)
    p1 = list(p1)
    p1 = np.float32(p1)

    assert(sum(status) == len(p1))
    return (p1, status, err)


def main():
    img0 = None; img1 = None;
    base = '../KITTIDataset/dataset/sequences/00/image_0/{:06d}.png'

    # Get the first two file names
    f0 = base.format(0)
    f1 = base.format(0)

    # Read Images
    img0 = cv2.imread(f0)
    img1 = cv2.imread(f1)

    # Convert images to grayscale
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Detect features in first image
    p0 = detectFeatures(img0)
    # img_with_features = cv2.drawKeypoints(img, p0, color=(255, 0, 0), outImage=None)

    p1, status, err = trackFeatures(img0, img1, p0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

