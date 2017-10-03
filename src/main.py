import numpy as np 
import cv2
import pdb
import re
from MonoVisualOdometer import *
from Dataset import *

def main():
    kittidataset_00 = KittiDataset(0, 0)
    a = MonoVisualOdometer(kittidataset_00)   
    a.run(50)   

if __name__ == "__main__":
    main()

