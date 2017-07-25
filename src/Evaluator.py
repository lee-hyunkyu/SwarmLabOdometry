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
        return cv2.resize(img, self.reshape_size)

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
