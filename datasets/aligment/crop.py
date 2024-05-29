import os
# from PIL import Image
import tifffile as tiff

import matplotlib.pyplot as plt
import numpy as np
import torch

def crop_image(img, target_size, center):
    x, y = center
    w, h = target_size
    left = int(y - w/2)
    top = int(x - h/2)
    right = int(y + w/2)
    bottom = int(x + h/2)
    if len(img.shape)>2:
        return img[:,top:bottom, left:right]
    else:
        return img[top:bottom, left:right]

def process_images(folder_path, des_dir):
    ms_images = {}
    pan_images = {}
    new_index = 1
    ms_size = 128  # TODO can be changed
    pan_size = ms_size*4
    # Create directories for cropped images
    cropped_ms_path = os.path.join(des_dir, 'MS')
    cropped_pan_path = os.path.join(des_dir, 'PAN')
    # os.makedirs(cropped_ms_path, exist_ok=True)
    # os.makedirs(cropped_pan_path, exist_ok=True)

    # Locate and sort the images
    for filename in os.listdir(folder_path):
        if filename.endswith('-MUL.TIF'):
            index = int(filename.split('-')[0])
            ms_images[index] = filename
        elif filename.endswith('-PAN.TIF'):
            index = int(filename.split('-')[0])
            pan_images[index] = filename
        index += 1

    # Process and crop images
    for index in ms_images.keys():
        if index in pan_images.keys():
            # ms_img = imageio.imread(os.path.join(folder_path, ms_images[index]))
            # pan_img = imageio.imread(os.path.join(folder_path, pan_images[index]))

            ms_img = np.array(gdal.Open(os.path.join(folder_path, ms_images[index])).ReadAsArray(), dtype=np.double)
            pan_img = np.array(gdal.Open(os.path.join(folder_path, pan_images[index])).ReadAsArray(), dtype=np.double)

            num_h = int(ms_img.shape[1]//ms_size) # 可以分割的次数
            num_w = int(ms_img.shape[2]//ms_size)
            print(f'第{index}张图片可以分割的次数为{num_h*num_w}次！')
            for i in range(num_h):
                for j in range(num_w):
                    ms_center = (ms_size*i+ms_size//2, ms_size*j+ms_size//2)
                    pan_center = (pan_size*i+pan_size//2, pan_size*j+pan_size//2)
                    
                    # Crop images
                    cropped_ms = crop_image(ms_img, (ms_size, ms_size), center = ms_center)
                    cropped_pan = crop_image(pan_img, (pan_size, pan_size), center = pan_center)


                    # imageio.imsave(os.path.join(des_dir, f'{new_index}-MS.TIF'), cropped_ms, format="TIFF")
                    # imageio.imsave(os.path.join(des_dir, f'{new_index}-PAN.TIF'), cropped_pan, format="TIFF")
                    tiff.imsave(os.path.join(des_dir, f'{new_index}-MS.TIF'), cropped_ms)

                    tiff.imsave(os.path.join(des_dir, f'{new_index}-PAN.TIF'), cropped_pan)
                    new_index += 1

# Replace 'your_folder_path' with the path to your image folder
if __name__=='__main__':
    process_images(r'E:\anocondaProject\science_learning\RS\mywork\功能学习\alignment\QB', './newdata')
