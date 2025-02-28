import os
import cv2 as cv
import numpy as np
from glob import glob
from skimage.segmentation import slic, mark_boundaries
from utils import save_json
from PIL import Image


def gen_superpixel_label(paths, n_segments, compactness,Params):
    for i, path in enumerate(paths):
        print("----[%3s/%s] superpixel segmentating----" % (i + 1, len(paths)))
        name = path.split('\\')[-1]
        img = cv.imread(path)
        segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)


        save_path = Params['tar_dir'] + name.split('.jpg')[0]
        np.save(save_path, segments)
        save_path = Params['tar_dir'] + name.split('.jpg')[0] + '.json'
        save_json(save_path, segments)

        boundary = (mark_boundaries(img, segments, mode='thick') * 255).astype(np.uint8)
        save_path = Params['tar_dir2'] + name
        cv.imwrite(save_path, boundary)
        save_path = Params['tar_dir'] + name.split('.jpg')[0] + '/'
        os.mkdir(save_path)

        maxn = max(segments.reshape(int(segments.shape[0] * segments.shape[1]), ))
        for i in range(0, maxn+1):
            a = np.array(segments == i)
            a = a.tolist()
            a = np.stack((a,) * 3, axis=-1)
            b = img * a
            w, h = [], []
            for x in range(b.shape[0]):
                for y in range(b.shape[1]):
                    if b[x][y].all() != 0:
                        w.append(x)
                        h.append(y)


            c = b[min(w):max(w), min(h):max(h)]
            c1 = np.uint8(c)
            c1 = c1[:, :, ::-1]
            img2 = Image.fromarray(c1, mode='RGB')

            img2.save(save_path + str(i) + '.png')
            print('已保存第' + str(i) + '张图片')


if __name__ == '__main__':
    Params = {
        'root': './data/image/',
        'tar_dir': './data/superpixel/',
        'tar_dir2': './data/superpixel_vis/'
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    if not os.path.exists(Params['tar_dir2']):
        os.makedirs(Params['tar_dir2'])

    n_segments, compactness = 1500, 10  #
    paths = sorted(glob(Params['root'] + "*.jpg"))
    gen_superpixel_label(paths, n_segments, compactness, Params)
