import os
from glob import glob
import numpy as np
from _1_slic import gen_superpixel_label
from _2_scribble2npy import scri2label
from _3_Labeling_for_superpixel import Label_superpixel
from _4_feature_from_RGB_and_location import feature_extracting, extract_features_and_graph
from _5_find_nearest import spatial_similarity
from _6_creating_graph import creating_graph
from _7_GCN_training import GCN_training, generate_mask

if __name__ == '__main__':
    # ---------------1--------------------
    Params = {
        'root': './image/',
        'tar_dir': './data/superpixel/',
        'tar_dir2': './data/superpixel_vis/'
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    if not os.path.exists(Params['tar_dir2']):
        os.makedirs(Params['tar_dir2'])

    # n_segments, compactness = 3600, 20
    n_segments, compactness = 2500, 10
    paths = sorted(glob(Params['root'] + "*.jpg"))
    gen_superpixel_label(paths, n_segments, compactness,Params)

    # ---------------2--------------------
    Params = {
        'root': './scribble/',
        'tar_dir': './data/scribble_np/',
    }
    if not os.path.exists(Params['tar_dir']):
        os.makedirs(Params['tar_dir'])
    paths = sorted(glob(Params['root'] + "*.png"))
    scri2label(paths, Params)
    # ---------------3--------------------
    Params = {
        'scribble_np': './data/scribble_np/',
        'superpixel': './data/superpixel/',
        'sp_index2label': './data/sp_index2label/'
    }
    if not os.path.exists(Params['sp_index2label']):
        os.makedirs(Params['sp_index2label'])
    path1 = sorted(glob(Params['scribble_np'] + "*.npy"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    Label_superpixel(path1, path2, Params)


    # ---------------4--------------------
    Params = {
        'image': './image/',
        'superpixel': './data/superpixel/',
        'sp_index2label': './data/sp_index2label/',
        'features': "./data/features/"
    }
    os.makedirs(Params['features'], exist_ok=True)
    path1 = sorted(glob(Params['image'] + "*.jpg"))
    path2 = sorted(glob(Params['superpixel'] + "*.npy"))
    path3 = sorted(glob(Params['sp_index2label'] + "*.npy"))
    feature_extracting(path1, path2, path3, Params)

    # ---------------5--------------------
    Params = {
        'features': './data/features/',
        'connectivity': './data/connectivity/'
    }
    if not os.path.exists(Params['connectivity']):
        os.makedirs(Params['connectivity'])
    path = sorted(glob(Params['features'] + "*.npy"))
    spatial_similarity(path, Params)

    # ---------------6--------------------
    Params = {
        'features': './data/features/',
        'connectivity': './data/connectivity/',
        'sp_index2label': "./data/sp_index2label/",
        'graph': './data/graph/'
    }
    if not os.path.exists(Params['graph']):
        os.makedirs(Params['graph'])

    features_path = sorted(glob(Params['features'] + "*.npy"))
    connectivity_path = sorted(glob(Params['connectivity'] + "*.npy"))
    sp_index2label_path = sorted(glob(Params['sp_index2label'] + "*.npy"))
    creating_graph(features_path, connectivity_path, sp_index2label_path, Params)


    # ---------------7--------------------
    Params = {
        'graph': './data/graph/',
        'superpixel': './data/superpixel/',
        'sp_index2label': "./data/sp_index2label/",
        'image': './image/',
        'scribble': './scribble/',
        'result': './result/'
    }
    os.makedirs(Params['result'], exist_ok=True)

    image_paths = sorted(glob(Params['image'] + "*.jpg"))
    superpixel_paths = sorted(glob(Params['superpixel'] + "*.npy"))
    label_paths = sorted(glob(Params['sp_index2label'] + "*.npy"))

    P, C, S, K, omega_ul = {}, {}, {}, {}, {}

    for i in range(len(image_paths)):
        p, c, s, k, omega = extract_features_and_graph(image_paths[i], superpixel_paths[i], label_paths[i])
        P[i] = p
        C[i] = c
        S[i] = s
        K[i] = k
        omega_ul[i] = omega

    model, adj, features, idx_test = GCN_training(Params, epochs=100, dropout=0.3)

    model.eval()
    gcn_th = 0.3
    output = model(features, adj)
    graph_prediction = output.max(1)[1].cpu().numpy().astype(np.uint8)

    super_paths = sorted(glob(Params['superpixel'] + "*.npy"))
    im_paths = sorted(glob(Params['image'] + "*.jpg"))
    sci_paths = sorted(glob(Params['scribble'] + "*.png"))

    generate_mask(graph_prediction, super_paths, im_paths, sci_paths, n_segments, Params)