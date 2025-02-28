import os
import numpy as np
import cv2 as cv
from glob import glob
from utils import save_json


def scri2label(paths, Params):
    for path in paths:
        name = path.split('\\')[-1].split(".")[0]
        image = cv.imread(path, -1)
        hight, width = image.shape[0:2]
        mask = np.zeros((hight, width), dtype=np.uint8)
        number_of_dry = 0
        number_of_water = 0
        number_of_ice = 0
        number_of_snow = 0
        number_of_non_road = 0
        number_of_free = 0
        number_of_all_pixel = hight * width
        for h in range(hight):
            for w in range(width):
                r = image[h, w, 2]
                g = image[h, w, 1]
                b = image[h, w, 0]
                if r >= 240 and g <= 15 and b <= 15:
                    mask[h, w] = 1  # 1为已经确定的干燥路面区域，红色
                    number_of_dry += 1
                elif r <= 15 and g >= 240 and b <= 15:
                    mask[h, w] = 2  # 0为已经确定的积水路面区域，绿色
                    number_of_water += 1
                elif r <= 15 and g <= 15 and b >= 240:
                    mask[h, w] = 3  # 0为已经确定的结冰路面区域，蓝色
                    number_of_ice += 1
                elif r >= 240 and g >= 240 and b <= 15:
                    mask[h, w] = 4  # 0为已经确定的积雪路面区域，黄色
                    number_of_snow += 1
                elif r <= 15 and g <= 15 and b <= 15:
                    mask[h, w] = 0  # 0为已经确定的非路面区域，黑色
                    number_of_non_road += 1
                else:
                    mask[h, w] = 5  # 5为不确定区域
                    number_of_free += 1
        save_path = Params['tar_dir'] + name
        np.save(save_path, mask)
        save_path = Params['tar_dir'] + name + '.json'
        save_json(save_path, mask)
        print("被交互到的干燥路面区域像素:%s/%s" % (number_of_dry, number_of_all_pixel))
        print("被交互到的积水路面区域像素:%s/%s" % (number_of_water, number_of_all_pixel))
        print("被交互到的结冰路面区域像素:%s/%s" % (number_of_ice, number_of_all_pixel))
        print("被交互到的积雪路面区域像素:%s/%s" % (number_of_snow, number_of_all_pixel))
        print("被交互到的非路面区域像素:%s/%s" % (number_of_non_road, number_of_all_pixel))
        print("没有被交互到的像素:%s/%s" % (number_of_free, number_of_all_pixel))


if __name__ == '__main__':

    Params = {
        # 'root': './data/scribble/',
        'root': './scribble/',
        'tar_dir': './data/scribble_np/',
    }

    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])

    paths = sorted(glob(Params['root'] + "*.jpg"))
    scri2label(paths)
