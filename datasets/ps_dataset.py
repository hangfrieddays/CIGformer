import os
from torch.utils.data import DataLoader

from models.utils.registry import DATASET_REGISTRY
from .utils import *
import cv2

def _is_pan_image(filename):
    return filename.endswith("pan.tif")

@DATASET_REGISTRY.register()
class PanSharpeningDataset(torch.utils.data.Dataset):
    def __init__(self, root:str, bit_depth, norm_input = False, mode = 'train'):
        r""" Build dataset from folders

        Args:
            root (list[str]): image directories
            bit_depth (int): data value range in n-bit
            norm_input (bool): normalize the input to [0, 1]
        """

        self.root = root  
        self.bit_depth = bit_depth
        self.norm_input = norm_input
        self.mode = mode
        self.image_ids = []
        self.image_prefix_names = []  # full-path filename prefix
        root = [root]
        for y in root:
            for x in os.listdir(y):
                if _is_pan_image(x):
                    self.image_ids.append(get_image_id(x))
                    self.image_prefix_names.append(os.path.join(y, get_image_id(x)))


    def __getitem__(self, index):
        prefix_name = self.image_prefix_names[index]

        input_dict = dict(
            image_ms=load_image('{}_ms.tif'.format(prefix_name)),  # [4,64,64] LR MS
            image_pan=load_image('{}_pan.tif'.format(prefix_name))[np.newaxis, :],  # [1,256,256] PAN
            image_ms_label= load_image('{}_ms_label.tif'.format(prefix_name)),  # [C,256,256] ms_label
            # image_pan_label= load_image('{}_pan_label.tif'.format(prefix_name))[np.newaxis, :],  # [1,256,256] pan_label
        )

        for key in input_dict:  # numpy2torch
            input_dict[key] = torch.from_numpy(input_dict[key]).float()

        if self.norm_input:
            input_dict = data_normalize(input_dict, self.bit_depth)

        input_dict['image_id'] = index
        return input_dict  

    def __len__(self):
        return len(self.image_ids)   

