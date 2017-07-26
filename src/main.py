import numpy as np 
import cv2
import pdb
import re
from MonoVisualOdometer import *
from Dataset import *
from Evaluator import *

def main():
    kittidataset_00 = KittiDataset(0, 0)
    log_file_path   = './../logs/Evaluator/00.txt'
    reshape_size    = (128, 128) # As we expect the images to be
    evaluator       = Evaluator(1000, log_file_path, reshape_size)
    a = MonoVisualOdometer(kittidataset_00, None)   
    a.run(300)   

if __name__ == "__main__":
    main()

