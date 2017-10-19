import numpy as np 
import cv2 
import re 
import sys
import pdb
import os

def crop_features(feature_pts, image_size, crop_size=(500, 500)):
    '''
    Given a set of feature pts of shape (N, 1, 2), returns features that are not within a
    CROP_SIZE square in the center of the original image
    '''
    if feature_pts is None:
        return None
    height, width = image_size
    min_y = int(height/2 - crop_size[0]/2)
    min_x = int(width/2-crop_size[1]/2)
    
    feature_pts = feature_pts[:,0] # Remove the additional dimension

    # The indices are reverses here b/c feature_pts returns an array of [[x, y]...]
    crop_lambda = lambda pt: pt[1] > min_y and pt[1] < min_y + crop_size[1] \
                             and pt[0] > min_x and pt[0] < min_x + crop_size[0]
    feature_pts = np.array(list(filter(crop_lambda, feature_pts)))
    feature_pts = np.reshape(feature_pts, (len(feature_pts), 1, 2))
    return feature_pts

def crop_img(img, crop_size=(128, 128)):
    if crop_size == (0, 0):
        return img
    height, width = img.shape
    min_y = int(height/2 - crop_size[0]/2)
    min_x = int(width/2 - crop_size[1]/2)
    return img[min_y:min_y + crop_size[0], min_x:min_x+crop_size[1]]

class MonoVisualOdometer:
    def __init__(self, dataset):

        self.dataset = dataset

        # Set the needed variables to None or create if possible
        self.prev_img               = None;
        self.curr_img               = None;
        self.prev_feature_pts       = None;
        self.curr_feature_pts       = None;
        self.fast_detector          = cv2.FastFeatureDetector_create()
        self.curr_R                 = None;
        self.curr_t                 = None;

        # Read the first two images
        self.focal = self.dataset.get_focal()
        self.principal_point = self.dataset.get_principal_point()
        self.prev_img = self.dataset.get_image(0)
        self.curr_img = self.dataset.get_image(1)
        
        # Crop Image
        self.crop_size = (128, 128); crop_size_y, crop_size_x = self.crop_size
        print(self.crop_size)
        self.principal_point = (crop_size_x/2-1, crop_size_y/2-1)
        self.prev_img = crop_img(self.prev_img, self.crop_size)
        self.curr_img = crop_img(self.curr_img, self.crop_size)
        _, self.prev_feature_pts = self.detectFeatures(self.prev_img)
        self.prev_feature_pts, self.curr_feature_pts = self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)
        _, self.curr_R, self.curr_t = self.getTranslationRotation()
        self.step()

    def end(self):
        self.ground_truth_file.close()

    def getGroundtruthXYZ(self):
        ''' 
        Returns the XYZ of the ground truth 
        Assumes KITTI Dataset 
        '''
        return self.dataset.get_ground_truth()

    def detectFeatures(self, img):
        ''' Returns the features detected by FAST method '''
        feature_key_pts = self.fast_detector.detect(img, None) # array of KeyPoint objects    
        # numpy array of pts represented as [ [[x,y]] ... ]
        feature_pts     = np.float32([[f.pt[0], f.pt[1]] for f in feature_key_pts]).reshape(-1, 1, 2);
        return (feature_key_pts, feature_pts)

    def trackFeatures(self, img0, img1, img0_features):
        ''' 
        Tracks features from IMG0 to IMG1 
        IMG0_FEATURES should be of shape (-1, 1, 2)
        You should pass in the second return value from detectFeatures
        '''
        img1_features, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, img0_features, None); # TODO: Find others? 

        # Reshape to be simpler row vectors and remove the extra dimension
        status = status[:,0] 
        img1_features = img1_features[:,0]

        # Invalid features whose coordinates are negative (out of frame)
        status = np.array([s if (p[0] > 0 and p[1] > 0) else 0 for s, p in zip(status, img1_features)])

        # Filter the invalid features
        img1_features = img1_features[status==1]
        img0_features = img0_features[:,0]
        img0_features = img0_features[status==1]

        # Reshape to be correct shape
        img1_features = img1_features.reshape(-1, 1, 2)
        img0_features = img0_features.reshape(-1, 1, 2)
        return (img0_features, img1_features)

    def step(self):
        ''' Sets current to be the previous '''
        self.prev_img = self.curr_img;
        self.curr_img = None;
        self.prev_feature_pts = self.curr_feature_pts;
        self.curr_feature_pts = None; 

    def run(self, limit):
        # Initialize values
        prev_x, prev_y, prev_z = self.getGroundtruthXYZ()

        # For video
        cv2.namedWindow('Road facing camera', cv2.WINDOW_AUTOSIZE);
        cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE);
        traj = np.zeros((600, 600, 3), np.uint8)

        error_x, error_y, error_z = 0, 0, 0

        for i in range(2, limit):
            # Get the current image
            print(self.prev_feature_pts.shape)
            curr_img_file = self.dataset.get_image_file_path(i)
            self.curr_img = cv2.cvtColor(cv2.imread(curr_img_file), cv2.COLOR_BGR2GRAY)
            self.curr_img = crop_img(self.curr_img, self.crop_size)
            self.prev_feature_pts, self.curr_feature_pts = \
                self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)

            # If you lose too many features, redetect
            if len(self.prev_feature_pts < 2000):
                _, self.prev_feature_pts = self.detectFeatures(self.prev_img)
                self.prev_feature_pts, self.curr_feature_pts = \
                    self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)

            # Get the translation and rotation
            E, R, t = self.getTranslationRotation();

            x, y, z = self.getGroundtruthXYZ()
            scale   = self.getAbsoluteScale(x, prev_x, y, prev_y, z, prev_z)
            
            real_t  = -1*np.array([prev_x - x, prev_y - y, prev_z - z])
            # Update values of curr_t, curr_R
            t_x, t_y, t_z = t
            if max(t_x, t_y, t_z) == t_z:
                self.curr_t = self.curr_t + scale*np.dot(self.curr_R, t)
                self.curr_R = np.dot(R, self.curr_R)

            curr_x = self.curr_t[:,0][0]
            curr_y = self.curr_t[:,0][1]
            curr_z = self.curr_t[:,0][2]

            prev_x, prev_y, prev_z = (x, y, z)
            # Draw
            cv2.circle(traj, (int(curr_x + 300), int(curr_z + 100)), 1, (255, 0, 0), 2)
            cv2.circle(traj, (int(x + 300), int(z + 100)), 1, (0, 255, 0), 2)
            cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), thickness=-1)
            text = "Coordinates: x = {:02f}m y = {:02f}m z = {:02f}m   "
            text = text.format(curr_x, curr_y, curr_z)
            cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            cv2.imshow("Road facing camera", self.curr_img);
            cv2.imshow("Trajectory", traj);
            cv2.waitKey(1)


            self.step()

        print(len(self.prev_feature_pts))
        print(abs(curr_x - x))
        print(abs(curr_y - y))
        cv2.waitKey(0)
        cv2.putText(traj, "Done", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, 255)

    def getAbsoluteScale(self, x, prev_x, y, prev_y, z, prev_z):
        return np.sqrt((x-prev_x)**2 + (y-prev_y)**2 + (z-prev_z)**2)
            
    def getTranslationRotation(self):
        E, mask = cv2.findEssentialMat( self.curr_feature_pts, 
                                        self.prev_feature_pts,
                                        focal=self.focal, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.curr_feature_pts,
                                        self.prev_feature_pts,
                                        focal=self.focal, pp=self.principal_point)
        return (E, R, t)



