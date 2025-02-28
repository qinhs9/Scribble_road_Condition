from glob import glob
import numpy as np
import scipy.sparse as sp
from utils import save_json
import os
import cv2 as cv
import copy
import heapq


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class Weighting():
    def __init__(self):
        super(Weighting, self).__init__()
        self.description = "颜色特征 + 纹理特征"
        self.weights1 = []
        self.weights2 = []

    def weights_for(self, idx1, idx2, features):

        rgb1 = features[idx1, 0:9]
        rgb2 = features[idx2, 0:9]

        tex1 = features[idx1, 9:18]
        tex2 = features[idx2, 9:18]

        rgb_diff = rgb1 - rgb2
        tex_diff = tex1 - tex2

        L2_rgb = np.sum(rgb_diff * rgb_diff)
        L2_tex = np.sum(tex_diff * tex_diff)

        self.weights1.append(L2_rgb)
        self.weights2.append(L2_tex)

    def post_process(self, args=None):
        self.weights1 = np.asarray(self.weights1, dtype=np.float32)  # 266896
        self.weights2 = np.asarray(self.weights2, dtype=np.float32)  # 266896

        num_nodes = args["num_nodes"]
        ne = float(self.weights1.shape[0])

        muw1 = self.weights1.sum() / ne
        muw2 = self.weights2.sum() / ne

        sig1 = 2 * np.sum((self.weights1 - muw1) ** 2) / ne
        sig2 = 2 * np.sum((self.weights2 - muw2) ** 2) / ne

        self.weights1 = np.exp(-self.weights1 / sig1)
        self.weights2 = np.exp(-self.weights2 / sig2)

        w1 = sp.coo_matrix((self.weights1, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w2 = sp.coo_matrix((self.weights2, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))

        self.weights = 2 * w1 + w2

    def get_weights(self):
        return self.weights


def creating_graph(f_p, c_p, s2l_p, Params):

    number_of_image = len(f_p)
    for i in range(number_of_image):
        p1 = f_p[i]
        p2 = c_p[i]
        p3 = s2l_p[i]
        filename = p1.split("\\")[-1].split(".")[0]
        features = np.load(p1)
        connectivity = np.load(p2)
        s2l = np.load(p3)


        number_of_sp, features_dim = features.shape[:]

        f1 = features[:, 0]
        f2 = features[:, 1]
        f3 = features[:, 2]
        f4 = features[:, 3]
        f5 = features[:, 4]
        f6 = features[:, 5]
        f7 = features[:, 6]
        f8 = features[:, 7]
        f9 = features[:, 8]
        f10 = features[:, 9]
        f11 = features[:, 10]
        f12 = features[:, 11]
        f13 = features[:, 12]
        f14 = features[:, 13]
        f15 = features[:, 14]
        f16 = features[:, 15]
        f17 = features[:, 16]


        mean_f1 = f1.sum() / number_of_sp
        mean_f2 = f2.sum() / number_of_sp
        mean_f3 = f3.sum() / number_of_sp
        mean_f4 = f4.sum() / number_of_sp
        mean_f5 = f5.sum() / number_of_sp
        mean_f6 = f6.sum() / number_of_sp
        mean_f7 = f7.sum() / number_of_sp
        mean_f8 = f8.sum() / number_of_sp
        mean_f9 = f9.sum() / number_of_sp
        mean_f10 = f10.sum() / number_of_sp
        mean_f11 = f11.sum() / number_of_sp
        mean_f12 = f12.sum() / number_of_sp
        mean_f13 = f13.sum() / number_of_sp
        mean_f14 = f14.sum() / number_of_sp
        mean_f15 = f15.sum() / number_of_sp
        mean_f16 = f17.sum() / number_of_sp
        mean_f17 = f17.sum() / number_of_sp



        sigma_f1 = np.sum((f1 - mean_f1) ** 2) / number_of_sp
        sigma_f2 = np.sum((f2 - mean_f2) ** 2) / number_of_sp
        sigma_f3 = np.sum((f3 - mean_f3) ** 2) / number_of_sp
        sigma_f4 = np.sum((f4 - mean_f4) ** 2) / number_of_sp
        sigma_f5 = np.sum((f5 - mean_f5) ** 2) / number_of_sp
        sigma_f6 = np.sum((f6 - mean_f6) ** 2) / number_of_sp
        sigma_f7 = np.sum((f7 - mean_f7) ** 2) / number_of_sp
        sigma_f8 = np.sum((f8 - mean_f8) ** 2) / number_of_sp
        sigma_f9 = np.sum((f9 - mean_f9) ** 2) / number_of_sp
        sigma_f10 = np.sum((f10 - mean_f10) ** 2) / number_of_sp
        sigma_f11 = np.sum((f11 - mean_f11) ** 2) / number_of_sp
        sigma_f12 = np.sum((f12 - mean_f12) ** 2) / number_of_sp
        sigma_f13 = np.sum((f13 - mean_f13) ** 2) / number_of_sp
        sigma_f14 = np.sum((f14 - mean_f14) ** 2) / number_of_sp
        sigma_f15 = np.sum((f15 - mean_f15) ** 2) / number_of_sp
        sigma_f16 = np.sum((f16 - mean_f16) ** 2) / number_of_sp
        sigma_f17 = np.sum((f17 - mean_f17) ** 2) / number_of_sp



        f1_features = (f1 - mean_f1) / sigma_f1
        f2_features = (f2 - mean_f2) / sigma_f2
        f3_features = (f3 - mean_f3) / sigma_f3
        f4_features = (f4 - mean_f4) / sigma_f4
        f5_features = (f5 - mean_f5) / sigma_f5
        f6_features = (f6 - mean_f6) / sigma_f6
        f7_features = (f7 - mean_f7) / sigma_f7
        f8_features = (f8 - mean_f8) / sigma_f8
        f9_features = (f9 - mean_f9) / sigma_f9
        f10_features = (f10 - mean_f10) / sigma_f10
        f11_features = (f11 - mean_f11) / sigma_f11
        f12_features = (f12 - mean_f12) / sigma_f12
        f13_features = (f13 - mean_f13) / sigma_f13
        f14_features = (f14 - mean_f14) / sigma_f14
        f15_features = (f15 - mean_f15) / sigma_f15
        f16_features = (f16 - mean_f16) / sigma_f16
        f17_features = (f17 - mean_f17) / sigma_f17



        new_f1 = np.expand_dims(f1_features, axis=1)
        new_f2 = np.expand_dims(f2_features, axis=1)
        new_f3 = np.expand_dims(f3_features, axis=1)
        new_f4 = np.expand_dims(f4_features, axis=1)
        new_f5 = np.expand_dims(f5_features, axis=1)
        new_f6 = np.expand_dims(f6_features, axis=1)
        new_f7 = np.expand_dims(f7_features, axis=1)
        new_f8 = np.expand_dims(f8_features, axis=1)
        new_f9 = np.expand_dims(f9_features, axis=1)
        new_f10 = np.expand_dims(f10_features, axis=1)
        new_f11 = np.expand_dims(f11_features, axis=1)
        new_f12 = np.expand_dims(f12_features, axis=1)
        new_f13 = np.expand_dims(f13_features, axis=1)
        new_f14 = np.expand_dims(f14_features, axis=1)
        new_f15 = np.expand_dims(f15_features, axis=1)
        new_f16 = np.expand_dims(f16_features, axis=1)
        new_f17 = np.expand_dims(f17_features, axis=1)



        features = np.concatenate((new_f1, new_f2, new_f3, new_f4, new_f5, new_f6, new_f7, new_f8, new_f9, new_f10,
                                   new_f11, new_f12, new_f13, new_f14, new_f15, new_f16, new_f17),
                                  axis=1)  # 所有节点构成的特征矩阵  (1420, 5)，现在为17维特征

        edges = []
        tabu_list = {}
        weighting = Weighting()
        for node_x in range(number_of_sp):
            for node_y in range(1, len(connectivity[node_x])):
                if (node_x, connectivity[node_x, node_y]) not in tabu_list and (
                        connectivity[node_x, node_y], node_x) not in tabu_list:  # 判断某两个节点的边是否存在
                    tabu_list[(node_x, connectivity[node_x, node_y])] = 1  # adding the edge to the tabu list
                    weighting.weights_for(node_x, connectivity[node_x, node_y], features)  # 计算权重并保存到权重矩阵中
                    weighting.weights_for(connectivity[node_x, node_y], node_x, features)  # 保存其对称位置的权重
                    edges.append([node_x, connectivity[node_x, node_y]])
                    edges.append([connectivity[node_x, node_y], node_x])
        edges = np.asarray(edges, dtype=int)  # (26814, 2)
        pp_args = {
            "edges": edges,
            "num_nodes": number_of_sp
        }
        weighting.post_process(pp_args)
        weights = weighting.get_weights()
        edges, weights, _ = sparse_to_tuple(weights)


        uncertain_mask = []
        for value in s2l:
            if value == 5:
                uncertain_mask.append(1)
            else:
                uncertain_mask.append(0)
        uncertain_mask = np.expand_dims(np.array(uncertain_mask), axis=1)

        np.save(Params['graph'] + "graph.npy", edges)
        np.save(Params['graph'] + "graph_weights.npy", weights)
        np.save(Params['graph'] + "graph_node_features.npy", features)
        np.save(Params['graph'] + "graph_ground_truth.npy", s2l)
        np.save(Params['graph'] + "unc_mask.npy", uncertain_mask)

        print("------------构建图的结果-----------")
        print("图的形状: {}".format(edges.shape))
        print("权重的形状: {}".format(weights.shape))
        print("图特征的形状: {}".format(features.shape))
        print("训练样本的标签形状: {}".format(s2l.shape))
        print("不确定性像素mask的形状: {}".format(uncertain_mask.shape))
        print("参与计算的node数量: {}".format(number_of_sp))
        print("没有被注释到的node数量: {}".format(int(np.sum(s2l[s2l == 5] / 5))))
        print("被注释到的node数量: {}".format(number_of_sp - int(np.sum(s2l[s2l == 5] / 5))))
        print("被注释到的干燥路面node数量: {}".format(int(np.sum(s2l[s2l == 1]))))
        print("被注释到的积水路面node数量: {}".format(int(np.sum(s2l[s2l == 2] / 2))))
        print("被注释到的结冰路面node数量: {}".format(int(np.sum(s2l[s2l == 3] / 3))))
        print("被注释到的积雪路面node数量: {}".format(int(np.sum(s2l[s2l == 4] / 4))))
        print("被注释到的非路面node数量: {}".format(number_of_sp - int(np.sum(s2l[s2l == 2] / 2)) - int(np.sum(s2l[s2l == 1])) - int(np.sum(s2l[s2l == 3] / 3)) - int(np.sum(s2l[s2l == 4] / 4)) - int(np.sum(s2l[s2l == 5] / 5))))


if __name__ == '__main__':

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
