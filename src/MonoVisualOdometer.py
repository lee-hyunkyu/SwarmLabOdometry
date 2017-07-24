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
            self.images_file_path   = self.images_file_path + "{:06d}.png"
        else:
            self.images_file_path   = self.images_file_path + "/{:06d}.png"
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
        self.curr_R                 = None;
        self.curr_t                 = None;

        # Read the first two images
        f0 = self.images_file_path.format(0)
        f1 = self.images_file_path.format(1)

        self.prev_img = cv2.imread(f0)
        self.curr_img = cv2.imread(f1)
        self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)
        self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)
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

        for i in range(2, limit):
            # Get the current image
            curr_img_file = self.images_file_path.format(i)
            self.curr_img = cv2.cvtColor(cv2.imread(curr_img_file), cv2.COLOR_BGR2GRAY)

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
            
            # Update values of curr_t, curr_R
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



