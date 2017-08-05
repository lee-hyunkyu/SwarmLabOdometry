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
    evaluator       = Evaluator(1000, log_file_path)
    a = MonoVisualOdometer(kittidataset_00, None)   
    a.run(1000)   

if __name__ == "__main__":
    main()

