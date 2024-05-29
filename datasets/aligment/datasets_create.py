import os
import random
import gdal
from osgeo import osr
import torch
from torch.nn.functional import interpolate
import numpy as np
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt

# plt.imshow(img[0])
# plt.show()
def load_image(path):
    """ Load .TIF image to np.array

    Args:
        path (str): path of TIF image
    Returns:
        np.array: value matrix in [C, H, W] or [H, W]
    """
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
    return img

def save_image(path, array):
    """ Save np.array as .TIF image

    Args:
        path (str): path to save as TIF image
        np.array: shape like [C, H, W] or [H, W]
    """
    # Meaningless Default Value
    raster_origin = (-123.25745, 45.43013)
    pixel_width = 2.4
    pixel_height = 2.4

    if array.ndim == 3:
        chans = array.shape[0]
        cols = array.shape[2]
        rows = array.shape[1]
        origin_x = raster_origin[0]
        origin_y = raster_origin[1]

        driver = gdal.GetDriverByName('GTiff')

        out_raster = driver.Create(path, cols, rows, chans, gdal.GDT_UInt16)
        # print(path, cols, rows, chans, out_raster)
        out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))
        for i in range(1, chans + 1):
            out_band = out_raster.GetRasterBand(i)
            out_band.WriteArray(array[i - 1, :, :])
        out_raster_srs = osr.SpatialReference()
        out_raster_srs.ImportFromEPSG(4326)
        out_raster.SetProjection(out_raster_srs.ExportToWkt())
        out_band.FlushCache()
    elif array.ndim == 2:
        cols = array.shape[1]
        rows = array.shape[0]
        origin_x = raster_origin[0]
        origin_y = raster_origin[1]

        driver = gdal.GetDriverByName('GTiff')

        out_raster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
        out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))

        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(array[:, :])



def down_sample(imgs, r=4, mode='bicubic'):
    r""" down-sample the images

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        r (int): scale ratio, Default: 4
        mode (str): interpolate mode, Default: 'bicubic'
    Returns:
        torch.Tensor: images after down-sampling, shape of [N, C, H//r, W//r]
    """

    # _, h, w = imgs.shape
    h, w = imgs.shape[-2], imgs.shape[-1]
    if len(imgs.shape)>2:
        imgs = imgs.view(1, 4, h, w)

        figure = F.interpolate(imgs, size=[h // r, w // r], mode=mode, align_corners=True)
        figure = figure.squeeze(dim=0)
        return figure
    else:
        imgs = imgs.view(1, 1, h, w)
        figure = F.interpolate(imgs, size=[h // r, w // r], mode=mode, align_corners=True)
        figure = figure.squeeze(dim=0) 
        return figure
    





# 定义一个辅助函数来移动图像
def move_images(src_dir, image_pairs, destination):
    for pair in image_pairs:
        for image_type in ["MS", "PAN"]:
            # pan或者ms的保存
            filename = f"{pair}-{image_type}.TIF"

            src_path = os.path.join(src_dir, filename)
            des_path = os.path.join(destination, image_type.lower()+'_label', f"{pair}.TIF")

            down_des_path = os.path.join(destination, image_type.lower(), f"{pair}.TIF")

            orignal_img = load_image(src_path)
            orignal_img_ = torch.from_numpy(orignal_img)

            down_img = down_sample(orignal_img_, r=4, mode='bicubic')
            down_img = np.array(down_img)

            save_image(des_path, orignal_img)
            save_image(down_des_path, down_img)


def create_dataset(train_ratio, test_ratio, valid_ratio, src_dir, des_dir):
    # if train_ratio + test_ratio + valid_ratio != 1:
    #     raise ValueError("The sum of ratios must be 1")

    # 创建数据集文件夹
    train_folder = os.path.join(des_dir, "TrainFolder")
    test_folder = os.path.join(des_dir, "TestFolder")
    valid_folder = os.path.join(des_dir, "ValidFolder")

    for folder in [train_folder, test_folder, valid_folder]:
        os.makedirs(os.path.join(folder, "ms"), exist_ok=True)
        os.makedirs(os.path.join(folder, "pan"), exist_ok=True)
        os.makedirs(os.path.join(folder, "pan_label"), exist_ok=True)
        os.makedirs(os.path.join(folder, "ms_label"), exist_ok=True)

    # 获取所有的MS和PAN图像
    images = [f for f in os.listdir(src_dir) if f.endswith('.TIF')]
    lens = len(images)//2  # ms and pan 

    pairs = set(f.split('-')[0] for f in images)  # 索引序号



    # 随机分配图像到训练、测试、验证集
    # 从图像当中选择子集

    print(f'图像数据集的图像个数为{lens}')
    num = int(input('请选择要摘取的图像个数'))

    sub_pairs = set(random.sample((list(pairs)), num))

    total_images = len(sub_pairs)
    train_count = int(total_images * train_ratio)
    test_count = int(total_images * test_ratio)

    train_pairs = set(random.sample(list(sub_pairs), train_count))
    sub_pairs -= train_pairs
    test_pairs = set(random.sample(list(sub_pairs), test_count))
    valid_pairs = sub_pairs - test_pairs



    # 移动图像到相应文件夹

    move_images(src_dir, train_pairs, train_folder)
    move_images(src_dir, test_pairs, test_folder)
    move_images(src_dir, valid_pairs, valid_folder)

# 接着创建下采样的ms和ms_label以及pan和pan_label

# 使用示例

create_dataset(0.7, 0.2, 0.1, './newdata', './Dataset')

