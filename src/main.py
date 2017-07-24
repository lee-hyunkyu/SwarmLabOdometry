import numpy as np 
import cv2
import pdb
import re
from MonoVisualOdometer import *

def main():
    images_file_path = '../KITTIDataset/dataset/sequences/00/image_0'
    groundtruth_file_path = '../KITTIDataset/dataset/poses/00.txt'
    calib_file_path = '../KITTIDataset/dataset/sequences/00/calib.txt'
    a = MonoVisualOdometer(images_file_path, groundtruth_file_path, calib_file_path)   
    a.run(20)   

if __name__ == "__main__":
    main()

