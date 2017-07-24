import numpy as np 
import cv2 
import re 

import pdb

class MonoVisualOdometer:
    def __init__(self, images_file_path, ground_truth_file_path, calib_file_path):
        ''' Assumes KITTI Dataset '''
        self.images_file_path       = images_file_path
        self.ground_truth_file_path = ground_truth_file_path

        last_char_in_path = self.images_file_path[-1]
        if last_char_in_path == '/':
            self.images_file_path   = self.images_file_path + "{06d}.png"
        else:
            self.images_file_path   = self.images_file_path + "/{06d}.png"
        self.ground_truth_file      = open(self.ground_truth_file_path)

        # Get Camera Caliberation settings
        calib_file              = open(calib_file_path)
        line                    = calib_file.readline()
        words                   = re.split(' ', line)
        self.focal              = float(words[1])
        self.principal_point    = (float(words[3]), float(words[7]))

        # Set the needed variables to None or create if possible
        self.prev_img               = None;
        self.curr_img               = None;
        self.prev_feature_pts       = None;
        self.curr_feature_pts       = None;
        self.fast_detector          = cv2.FastFeatureDetector_create()
        self.E                      = None;
        self.R                      = None;
        self.t                      = None;

        # Read the first two images
        f0 = self.images_file_path.format(0)
        f1 = self.images_file_path.format(1)

        self.prev_img = cv2.imread(f0)
        self.curr_img = cv2.imread(f1)
        self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)
        self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)
        _, self.prev_feature_pts = self.detectFeatures(self.prev_img)
        self.prev_feature_pts, self.curr_feature_pts = self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)

        self.E, self.R, self.t = getTranslationRotation()
        self.step()

    def end(self):
        self.ground_truth_file.close()

    def getGroundtruthXYZ(self):
        ''' 
        Returns the XYZ of the ground truth 
        Assumes KITTI Dataset 
        '''
        line = self.ground_truth_file.readline()
        words = re.split(' ', line)
        x = float(words[3])
        y = float(words[7])
        z = float(words[-1])
        return (x, y, z)

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
        img1_features, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, img0_features, None);

        # Reshape to be simpler row vectors and remove the extra dimension
        status = status[:,0] 
        img1_features = img1_features[:,0]

        # Invalid features whose coordinates are negative (out of frame)
        status = np.array([s if (p[0] > 0 and p[1] > 0) else 0 for s, p in zip(status, img1_features)])

        # Filter the invalid features
        img1_features = img1_features[status==1]
        img0_features = img0_features[status==1]

        return (img0_features, img1_features)

    def step(self):
        ''' Sets current to be the previous '''
        self.prev_img = self.curr_img;
        self.curr_img = None;
        self.prev_feature_pts = self.curr_feature_pts;
        self.curr_feature_pts = None; 

    def run(self, limit):
        # Initialize values
        t_f = self.t
        R_f = self.R

        # For video
        cv2.namedWindow('Road facing camera', cv2.WINDOW_AUTOSIZE);
        cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE);
        traj = np.zeros((600, 600, 3), np.uint8)

        for i in range(2, limit):
            curr_img_file = self.images_file_path.format(i)
            self.curr_img = cv2.cvtColor(cv2.imread(curr_img_file), cv2.COLOR_BGR2GRAY)

            self.prev_feature_pts, self.curr_feature_pts = \
                self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)

            # If you lose too many features, redetect
            if len(self.prev_feature_pts < 2000):
                _, self.prev_feature_pts = detectFeatures(self.prev_img)
                self.prev_feature_pts, self.curr_feature_pts = \
                    self.trackFeatures(self.prev_img, self.curr_img, self.prev_feature_pts)

            self.step()

            

    def getTranslationRotation(self):
        E, mask = cv2.findEssentialMat( self.curr_feature_pts, 
                                        self.prev_feature_pts,
                                        focal=self.focal, pp=self.principal_point,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.curr_feature_pts,
                                        self.prev_feature_pts,
                                        focal=self.focal, pp=self.principal_point)
        return (E, R, t)


    



