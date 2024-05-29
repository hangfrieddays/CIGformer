"""
This is a tool to clip the original .TIF images to small patches to build a dataset.
"""
import gdal
import mmcv
import numpy as np
import argparse
import sys
from osgeo import osr
sys.path.append('.')
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


def parse_args():
    parser = argparse.ArgumentParser(description='clip into patches')
    parser.add_argument('-d', '--data_dir', required=True, help='root of data directory')
    parser.add_argument('-s', '--satellite', required=True, help='name of the satellite/dataset')
    parser.add_argument('-n', required=True, type=int, help='total number of images pairs')
    parser.add_argument('-p', '--patch_num', required=True, type=int,
                        help='random clip how many patches for one training scene')
    parser.add_argument('-r', '--rand_seed', type=int, default=0,
                        help='random seed to sample training patches')
    parser.add_argument('--no_low_train', action='store_true',
                        help='whether generate low-resolution training set or not')
    parser.add_argument('--no_full_train', action='store_true',
                        help='whether generate full-resolution training set or not')
    parser.add_argument('--no_low_test', action='store_true',
                        help='whether generate low-resolution testing set or not')
    parser.add_argument('--no_full_test', action='store_true',
                        help='whether generate ful-resolution testing set or not')
    return parser.parse_args()


""""
    python clip_patch.py -d F:\3dataset\ps4data\PSData4 -s GF-1 -n 9 -p 24000 -r 0 --no_low_train --no_low_test --no_full_test
"""

""""
    python clip_patch.py -d F:\3dataset\ps4data\PSData4 -s GF-1 -n 9 -p 24000 -r 0  --no_full_train --no_full_test
"""

if __name__ == "__main__":
    args = parse_args()
    in_dir = f'{args.data_dir}/{args.satellite}/Dataset'

    # with open(f'{in_dir}/clip_patch.log', 'w') as f:
    #     for k, v in args._get_kwargs():
    #         f.write(f'{k} = {v}\n')

    # train patch_size = 64
    patch_size = 64
    if not args.no_low_train:
        # train low-res
        cnt = 0
        out_dir = f"{args.data_dir}/{args.satellite}/Dataset/train_low_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        rand = np.random.RandomState(args.rand_seed)
        # image_id from [1, n-1] is for train
        for num in range(1, args.n):
            mul = f'{in_dir}/{num}_ms_label.tif'
            lr = f'{in_dir}/{num}_ms.tif'
            lr_u = f'{in_dir}/{num}_lr_u.tif'
            pan = f'{in_dir}/{num}_pan.tif'

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)

            img_mul = dt_mul.ReadAsArray()  # (c, h, w)
            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            for _ in range(args.patch_num): # 同一张图片进行patch_num次分割
                x = rand.randint(XSize - patch_size)
                y = rand.randint(YSize - patch_size)

                save_image(f'{out_dir}/{cnt}_ms_label.tif',
                           img_mul[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_lr_u.tif',
                           img_lr_u[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_ms.tif',
                           img_lr[:, y:(y + patch_size), x:(x + patch_size)])
                save_image(f'{out_dir}/{cnt}_pan.tif',
                           img_pan[y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                cnt += 1
            print("low-res train done %d" % num)
        record.write("%d\n" % cnt)
        record.close()

    if not args.no_full_train:
        # train full-res
        cnt = 0
        out_dir = f"{args.data_dir}/{args.satellite}/Dataset/train_full_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        rand = np.random.RandomState(args.rand_seed)
        # image_id from [1, n-1] is for train
        for num in range(1, args.n):
            lr = f'{in_dir}/{num}_mul_o.tif'
            lr_u = f'{in_dir}/{num}_mul_o_u.tif'
            pan = f'{in_dir}/{num}_pan_o.tif'

            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)

            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            for _ in range(args.patch_num):
                x = rand.randint(XSize - patch_size)
                y = rand.randint(YSize - patch_size)

                save_image(f'{out_dir}/{cnt}_lr_u.tif',
                           img_lr_u[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_lr.tif',
                           img_lr[:, y:(y + patch_size), x:(x + patch_size)])
                save_image(f'{out_dir}/{cnt}_pan.tif',
                           img_pan[y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                cnt += 1
            print("full-res train set done %d" % num)
        record.write("%d\n" % cnt)
        record.close()

    # test patch_size = 100
    patch_size = 100
    if not args.no_low_test:
        # test low-res
        cnt = 0
        out_dir = f"{args.data_dir}/{args.satellite}/Dataset/test_low_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        # image_id from [0] is for test
        for num in range(1):
            mul = f'{in_dir}/{num}_ms_label.tif'
            lr = f'{in_dir}/{num}_ms.tif'
            lr_u = f'{in_dir}/{num}_lr_u.tif'
            pan = f'{in_dir}/{num}_pan.tif'

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_lr_u = gdal.Open(lr_u)

            img_mul = dt_mul.ReadAsArray()
            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            row = 0
            col = 0
            # 按顺序切(100, 100)小块 overlap 1/2
            for y in range(0, YSize, patch_size // 2):
                if y + patch_size > YSize:
                    continue
                col = 0
                for x in range(0, XSize, patch_size // 2):
                    if x + patch_size > XSize:
                        continue
                    save_image(f'{out_dir}/{cnt}_ms_label.tif',
                               img_mul[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_lr_u.tif',
                               img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_ms.tif',
                               img_lr[:, y:(y + 100), x:(x + 100)])
                    save_image(f'{out_dir}/{cnt}_pan.tif',
                               img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    cnt += 1
                    col += 1
                row += 1
            record.write("%d: %d * %d\n" % (num, row, col))
        record.write("%d\n" % cnt)
        record.close()
        print("low-res test set done")

    if not args.no_full_test:
        # test full-res
        cnt = 0
        out_dir = f"{args.data_dir}/{args.satellite}/Dataset/test_full_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        # image_id from [0] is for test
        for num in range(1):
            lr = f'{in_dir}/{num}_mul_o.tif'
            lr_u = f'{in_dir}/{num}_mul_o_u.tif'
            pan = f'{in_dir}/{num}_pan_o.tif'

            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_lr_u = gdal.Open(lr_u)

            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            row = 0
            col = 0
            for y in range(0, YSize, patch_size // 2):
                if y + patch_size > YSize:
                    continue
                col = 0
                for x in range(0, XSize, patch_size // 2):
                    if x + patch_size > XSize:
                        continue

                    save_image(f'{out_dir}/{cnt}_lr_u.tif',
                               img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_lr.tif',
                               img_lr[:, y:(y + 100), x:(x + 100)])
                    save_image(f'{out_dir}/{cnt}_pan.tif',
                               img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    cnt += 1
                    col += 1
                row += 1
            record.write("%d: %d * %d\n" % (num, row, col))
        record.write("%d\n" % cnt)
        record.close()
        print("full-res test set done")

    print("finish!!")
