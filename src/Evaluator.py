import numpy as np 
import cv2 
import re 
import sys
import pdb
import os
import logging

class Evaluator:
    '''
    A class designed to be used to evaluate the accuracy of our implementation of MVO given certain parameters
    Pass in as a parameter to MonoVisualOdometer to use
    '''

    def __init__(self, min_num_of_features, log_file_path, reshape_size):
        self.min_num_of_features = min_num_of_features
        if not log_file_path:
            log_file_path = '../logs/Evaluator/default.txt'
        logging.basicConfig(filename= log_file_path, 
                        level       = logging.DEBUG,
                        filemode    = 'w',
                        format      = '%(message)s')
        self.logger = logging.getLogger()

        self.reshape_size = reshape_size

        self.diffs = {};
        self.diffs['x'] = [];
        self.diffs['y'] = [];
        self.diffs['z'] = [];

    def process_img(self, img):
        ''' Downsamples/reshapes image as determiend by the object's variables '''
        img_h, img_w = np.shape(img)
        height, width = self.reshape_size
        x_min = int(img_w/2 - width/2)
        y_min = int(img_h/2 - height/2)
        new_img = img[y_min:y_min+height, x_min:x_min+width]
        return new_img

    def log(self, frame_num, estimated_x, ground_truth_x, estimated_y, ground_truth_y, estimated_z, ground_truth_z):
        msg_format = '{:06d}: diff_x = {:03f}, diff_y = {:03f}, diff_z = {:03f}'
        diff_x = estimated_x - ground_truth_x
        diff_y = estimated_y - ground_truth_y
        diff_z = estimated_z - ground_truth_z
        msg = msg_format.format(frame_num, diff_x, diff_y, diff_z)
        self.logger.info(msg)
        self.diffs['x'] += [diff_x]
        self.diffs['y'] += [diff_y]
        self.diffs['z'] += [diff_z]
        return [frame_num, diff_x, diff_y, diff_z, msg]

    def get_diffs_x(self):
        return self.diffs['x']

    def get_diffs_y(self):
        return self.diffs['y']

    def get_diffs_z(self):
        return self.diffs['z']
