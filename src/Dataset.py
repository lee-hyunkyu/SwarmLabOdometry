import os
import re
import pdb
import sys

try:
    sys.path.index(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # Or os.getcwd() for this directory

class Dataset:
    def __init__(self, images_file_path, ground_truth_file_path, camera_calib_file_path):
        self.images_file_path = images_file_path
        self.append_image_file_path()
        self.ground_truth_file_path = ground_truth_file_path
        self.camera_calib_file_path = camera_calib_file_path
        self.focal=None; self.principal_point=None

    def get_image_file_path(self, i):
        return self.images_file_path.format(i)

    # Override the following below

    def append_image_file_path(self):
        self.image_file_path = self.image_file_path + ""

    def get_ground_truth(self):
        return None

    def get_focal(self):
        return self.focal

    def get_principal_point(self):
        return self.principal_point

class KittiDataset(Dataset):

    BASE = '../Datasets/KITTIDataset/dataset/'
    IMAGES_FILE_PATH_BASE = BASE + 'sequences/{:02d}/image_{:01d}'
    GROUND_TRUTH_FILE_PATH_BASE = BASE + 'poses/{:02d}.txt'
    CALIB_FILE_PATH_BASE = BASE + 'sequences/{:02d}/calib.txt'

    def __init__(self, sequence_num, camera_num):
        camera_num = max(camera_num, 0)
        camera_num = min(camera_num, 1) # It's either a 0 or a 1
        super().__init__(KittiDataset.IMAGES_FILE_PATH_BASE.format(sequence_num, camera_num), 
                         KittiDataset.GROUND_TRUTH_FILE_PATH_BASE.format(sequence_num), 
                         KittiDataset.CALIB_FILE_PATH_BASE.format(sequence_num))
        # Get calibration settings
        calib_file              = open(self.camera_calib_file_path)
        line                    = calib_file.readline()
        words                   = re.split(' ', line)
        self.focal              = float(words[1])
        self.principal_point    = (float(words[3]), float(words[7]))

        self.ground_truth_file  = open(self.ground_truth_file_path)
        calib_file.close()

    def append_image_file_path(self):
        last_char = self.images_file_path[-1]
        if last_char == '/':
            self.images_file_path += '{:06d}.png'
        else:
            self.images_file_path += '/{:06d}.png'

    def get_ground_truth(self):
        line = self.ground_truth_file.readline()
        words = re.split(' ', line)
        x = float(words[3])
        y = float(words[7])
        z = float(words[-1])
        return (x, y, z)

class ComputerVisionDataset(Dataset):
    BASE = '../Datasets/ComputerVisionDataset/'
    IMAGES_FILE_PATH_BASE = BASE + 'sequences_{:02d}/image_{:02d}'
    GROUND_TRUTH_FILE_PATH_BASE = BASE + 'groundtruthSync.txt'
    CALIB_FILE_PATH_BASE = BASE + 'camera.txt'

    def __init__(self, sequence_num):
        super().__init__(KittiDataset.IMAGES_FILE_PATH_BASE.format(sequence_num), 
                         KittiDataset.GROUND_TRUTH_FILE_PATH_BASE.format(sequence_num), 
                         KittiDataset.CALIB_FILE_PATH_BASE.format(sequence_num))
        # Get caliberation settings
        calib_file              = open(self.camera_calib_file_path)
        line                    = calib_file.readline()
        fx                      = line[0]
        fy                      = line[1]
        cx                      = line[2]
        cy                      = line[3]
        line                    = calib_file.readline()
        width                   = line[0]
        height                  = line[1]
        # Assumes that the focal length in both directions are the same
        # focal length = (fx * in_width + fy * in_height)/2
        self.focal              = (fx * width + fy * height)/2.0
        self.principal_point    = (float(cx*width-0.5), float(cy*width-0.5))
        # That weird 0.5 offset? This is due to an assumption made with the vsion.in.tum dataset

    def append_img_file_path(self):
        last_char = self.images_file_path[-1]
        if last_char == '/':
            self.images_file_path += '{:05d}.jpg'
        else:
            self.images_file_path += '/{:05d}.jpg'

    def get_ground_truth(self):
        ''' 
        There's something a little weird about the groundtruthSync.txt file for this dataset
        Will look into why
        '''
        pass 