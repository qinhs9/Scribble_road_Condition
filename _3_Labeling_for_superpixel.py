
from glob import glob
import numpy as np
from utils import save_json
import os

def Label_superpixel(path1, path2,Params):
    for i in range(len(path1)):
        p1 = path1[i]
        p2 = path2[i]
        filename = p1.split("\\")[-1].split(".")[0]
        scribble_np = np.load(p1)
        superpixel = np.load(p2)
        superpixel_count = 0
        True_dry_positive = 0
        True_water_positive = 0
        True_ice_positive = 0
        True_snow_positive = 0
        True_negative = 0
        free_superpixel = 0
        Label_inforamtion = []
        while True:
            height = np.where(superpixel == superpixel_count)[0]
            width = np.where(superpixel == superpixel_count)[1]
            if not len(height):
                break
            print("----superpixel %s labeling----" % superpixel_count)
            superpixel_count += 1
            flage = True
            for h, w in zip(height, width):
                if scribble_np[h, w] == 1:
                    Label_inforamtion.append(1)
                    True_dry_positive += 1
                    flage = False
                    break
                elif scribble_np[h, w] == 2:
                    Label_inforamtion.append(2)
                    True_water_positive += 1
                    flage = False
                    break
                elif scribble_np[h, w] == 3:
                    Label_inforamtion.append(3)
                    True_ice_positive += 1
                    flage = False
                    break
                elif scribble_np[h, w] == 4:
                    Label_inforamtion.append(4)
                    True_snow_positive += 1
                    flage = False
                    break
                elif scribble_np[h, w] == 0:
                    Label_inforamtion.append(0)
                    True_negative += 1
                    flage = False
                    break
            if not flage:
                continue
            Label_inforamtion.append(5)
            free_superpixel += 1

        print("超像素总数", superpixel_count)
        print("被注释过的干燥路面区域超像素，红色", True_dry_positive)
        print("被注释过的积水路面区域超像素，绿色", True_water_positive)
        print("被注释过的结冰路面区域超像素，蓝色", True_ice_positive)
        print("被注释过的积雪路面区域超像素，黄色", True_snow_positive)
        print("被注释过的非路面区域超像素，黑色", True_negative)
        print("没有被注释到的超像素", free_superpixel)
        Label_inforamtion_np = np.array(Label_inforamtion)

        np.save(Params["sp_index2label"] + filename, Label_inforamtion_np)
        save_json(Params["sp_index2label"] + filename + ".json", Label_inforamtion_np)


if __name__ == '__main__':
    Params = {
        'scribble_np': './data/scribble_np/',
        'superpixel': './data/superpixel/',
        'sp_index2label': './data/sp_index2label/'
    }
    if not os.path.exists(Params['sp_index2label']):
        os.makedirs(Params['sp_index2label'])

    path1 = sorted(glob(Params['scribble_np'] + "*.npy"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))

    Label_superpixel(path1, path2)
