from glob import glob
import numpy as np
from utils import save_json
import os
import cv2 as cv
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern


def pixel_number(path):
    img = cv.imread(path)

    if img is None:
        return
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            if pixel.all() != 0:
                n = n + 1
    return n

def color_moments(path, n):
    img = cv.imread(path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)

    color_moments = []

    h_mean = np.sum(h)/float(n)
    s_mean = np.sum(s)/float(n)
    v_mean = np.sum(v)/float(n)
    color_moments.extend([h_mean, s_mean, v_mean])

    h_std = np.sqrt(np.sum(abs(h - h.mean())**2)/float(n))
    s_std = np.sqrt(np.sum(abs(s - s.mean())**2)/float(n))
    v_std = np.sqrt(np.sum(abs(v - v.mean())**2)/float(n))
    color_moments.extend([h_std, s_std, v_std])

    h_skewness = np.sum(abs(h - h.mean())**3)/float(n)
    s_skewness = np.sum(abs(s - s.mean())**3)/float(n)
    v_skewness = np.sum(abs(v - v.mean())**3)/float(n)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)

    color_moments.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    return color_moments

def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):

    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)


    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i, j] = cv.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm



def fast_glcm_contrast(img, n):


    h,w = img.shape
    texture_moments = []
    glcm = fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5)
    ks = 5
    nbit = 8


    max_ = np.max(glcm)
    pnorm = glcm / np.sum(glcm) + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm))


    mean = 0
    cont = 0
    diss = 0
    homo = 0
    asm = 0
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2
            cont += glcm[i, j] * (i-j)**2
            diss += glcm[i, j] * np.abs(i - j)
            homo += glcm[i, j] / (1. + (i - j) ** 2)
            asm += glcm[i, j] ** 2
    std2 = 0
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i, j] * i - mean) ** 2
    std = np.sqrt(std2)

    mean_mean = np.sum(mean) / n
    std_mean = np.sum(std) / n
    cont_mean = np.sum(cont)/n
    diss_mean = np.sum(diss) /n
    homo_mean = np.sum(homo) /n
    asm_mean = np.sum(cont) /n
    ene = np.sqrt(asm_mean)

    texture_moments.extend([ent, mean_mean, std_mean, cont_mean, diss_mean, homo_mean, asm_mean, ene])
    return texture_moments


def feature_extracting(path1, path2, path3, Params):
    number_of_image = len(path1)
    for i in range(number_of_image):
        p1 = path1[i]
        filename = p1.split("\\")[-1].split(".")[0]
        base_path = Params['superpixel'] + filename + '/'
        files = os.listdir(base_path)
        files.sort(key=lambda x: int(x.split('.')[0]))
        features = []

        for path in files:
            full_path = os.path.join(base_path, path)
            n = pixel_number(full_path)
            img = cv.imread(full_path, 0)
            feature = color_moments(full_path, n) + fast_glcm_contrast(img, n)
            features.append(feature)

        features = np.array(features).astype(np.float32)
        np.save(Params["features"] + filename, features)
        save_json(Params["features"] + filename + ".json", features)


def compute_centroids(mask):

    centroids = {}
    for sp in np.unique(mask):
        y, x = np.where(mask == sp)
        centroids[sp] = (float(np.mean(x)), float(np.mean(y)))
    return centroids


def compute_adjacency(mask):

    h, w = mask.shape
    adjacency = {}

    for i in range(h - 1):
        for j in range(w - 1):
            sp1 = mask[i, j]
            sp2 = mask[i, j + 1]
            sp3 = mask[i + 1, j]


            if sp1 != sp2:
                adjacency.setdefault(sp1, set()).add(sp2)
                adjacency.setdefault(sp2, set()).add(sp1)


            if sp1 != sp3:
                adjacency.setdefault(sp1, set()).add(sp3)
                adjacency.setdefault(sp3, set()).add(sp1)


    return {sp: list(neighbors) for sp, neighbors in adjacency.items()}

def color_moments_RGB(img, mask, n_pixels):

    region = img[mask == 1]
    mean_r = np.mean(region[:, 0]) if n_pixels > 0 else 0.0
    mean_g = np.mean(region[:, 1]) if n_pixels > 0 else 0.0
    mean_b = np.mean(region[:, 2]) if n_pixels > 0 else 0.0
    return [mean_r, mean_g, mean_b]

def extract_features_and_graph(image_path, superpixel_path, label_path):

    mask = np.load(superpixel_path)
    data = np.load(label_path, allow_pickle=True)


    if isinstance(data, dict):
        labels = data
    elif isinstance(data, np.ndarray):
        if data.dtype == 'O' and data.shape == (1,):
            labels = data.item()
        else:
            labels = dict(enumerate(data.flatten()))
    else:
        raise ValueError(f"不支持的标签格式: {type(data)}")


    unique_superpixels = np.unique(mask)
    num_superpixels = len(unique_superpixels)

    P = compute_centroids(mask)
    K_raw = compute_adjacency(mask)
    C = {}
    S = {}
    omega_ul = np.zeros(num_superpixels)


    K = {sp: list(neighbors) for sp, neighbors in K_raw.items()}


    img = cv.imread(image_path)


    for idx, sp in enumerate(unique_superpixels):

        sp_mask = (mask == sp).astype(np.uint8)


        n_pixels = np.count_nonzero(sp_mask)
        C[sp] = color_moments_RGB(img, sp_mask, n_pixels)


        S[sp] = labels.get(sp, -1)


        if S[sp] == -1:
            omega_ul[idx] = 1

    return P, C, S, K, omega_ul


if __name__ == '__main__':
    Params = {
        'image': './data/image/',
        'superpixel': './data/superpixel/',
        'sp_index2label': './data/sp_index2label/',
        'features': "./data/features/"
    }
    if not os.path.exists(Params['features']):
        os.makedirs(Params['features'])
    path1 = sorted(glob(Params['image'] + "*.jpg"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    path3 = sorted(glob(Params['sp_index2label'] + "*.npy"))
    feature_extracting(path1, path2, path3, Params)


