import torch
import numpy as np
from GCN import GCN
import torch.nn.functional as F
import scipy.sparse as sp
import torch.optim as optim
from glob import glob
import cv2 as cv
import os

from _4_feature_from_RGB_and_location import extract_features_and_graph
from utils import supervised_loss, local_consistency_loss, deterministic_loss, normalize, normalize_adj, sparse_mx_to_torch_sparse_tensor, accuracy



def load_data(Params):
    val_portion = 0.20

    graph_path = Params['graph'] + "graph.npy"
    weights_path = Params['graph'] + "graph_weights.npy"
    features_path = Params['graph'] + "graph_node_features.npy"
    labels_path = Params['graph'] + "graph_ground_truth.npy"
    mask_path = Params['graph'] + "unc_mask.npy"  # 不参与训练但要参与测试

    graph = np.load(graph_path)
    weights = np.load(weights_path)
    features = np.load(features_path)



    test_mask = np.load(mask_path)
    full_mask = 1 - test_mask

    labels = np.load(labels_path)
    num_nodes = labels.shape[0]

    adj = sp.coo_matrix((weights, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))
    features = sp.coo_matrix(features)
    working_nodes = np.where(full_mask != 0)[0]
    random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)


    features = normalize(features)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))


    idx_train = working_nodes[random_arr > val_portion]
    idx_val = working_nodes[random_arr <= val_portion]

    idx_test = np.where(test_mask != 0)


    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.FloatTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test[0])

    return adj, features, labels, idx_train, idx_val, idx_test



def train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, P, C, S, K, omega_ul):

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)


    loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
    acc_train = accuracy(output[idx_train], labels[idx_train])


    loss_lc = local_consistency_loss(P, C, S, K)
    loss_det = deterministic_loss(labels[idx_val], output[idx_val], omega_ul)
    total_loss = loss_train + loss_lc + 0.3 * loss_det
    total_loss.backward()
    optimizer.step()


    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    print(f'Epoch: {epoch + 1:04d}  loss_train: {loss_train.item():.4f}  acc_train: {acc_train.item():.4f}  '
          f'loss_lc: {loss_lc.item():.4f}  loss_det: {loss_det.item():.4f}  '
          f'loss_val: {loss_val.item():.4f}  acc_val: {acc_val.item():.4f}')



def GCN_training(Params, epochs, dropout):

    seed = 42
    lr = 0.1
    weight_decay = 1e-5
    hidden = 32
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(Params)


    image_paths = sorted(glob(Params['image'] + "*.jpg"))
    superpixel_paths = sorted(glob(Params['superpixel'] + "*.npy"))
    label_paths = sorted(glob(Params['sp_index2label'] + "*.npy"))

    P_list, C_list, S_list, edge_lists, omega_ul_list = [], [], [], [], []

    for i in range(len(image_paths)):

        p_dict, c_dict, s_dict, k_dict, omega = extract_features_and_graph(
            image_paths[i], superpixel_paths[i], label_paths[i]
        )


        P_list.append(np.array(list(p_dict.values()), dtype=np.float32))  # (num_nodes, 2)
        C_list.append(np.array(list(c_dict.values()), dtype=np.float32))  # (num_nodes, color_dim)
        S_list.append(np.array(list(s_dict.values()), dtype=np.int64))  # (num_nodes,)


        edge_list = []
        for src, dsts in k_dict.items():
            for dst in dsts:
                edge_list.append([src, dst])
        edge_lists.append(np.array(edge_list, dtype=np.int64))  # (num_edges, 2)

        omega_ul_list.append(omega)


    P = [torch.tensor(p, dtype=torch.float32).cuda() for p in P_list]
    C = [torch.tensor(c, dtype=torch.float32).cuda() for c in C_list]
    S = [torch.tensor(s, dtype=torch.long).cuda() for s in S_list]
    edge_indices = [torch.tensor(edges, dtype=torch.long).cuda().t() for edges in edge_lists]
    omega_ul = [torch.tensor(ul, dtype=torch.float32).cuda() for ul in omega_ul_list]


    all_labels = np.concatenate([s[s != -1].cpu().numpy() for s in S])
    n_classes = len(np.unique(all_labels)) if len(all_labels) > 0 else 1


    model = GCN(nfeat=P_list[0].shape[1], nhid=hidden, nclass=n_classes, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.cuda()

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for graph_id in range(len(P_list)):

            p = P[graph_id]
            c = C[graph_id]
            s = S[graph_id]
            edges = edge_indices[graph_id]
            ul_mask = omega_ul[graph_id]


            optimizer.zero_grad()
            output = model(features, adj)
            loss = local_consistency_loss(p, c, s, edges, ul_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(P_list):.4f}")

    return model, adj, features, idx_test

def generate_mask(graph_prediction, super_paths, im_paths, sci_paths, Params):
    for i, p in enumerate(super_paths):
        image_path = im_paths[i]
        image = cv.imread(image_path,-1)
        sci_path = sci_paths[i]
        sci = cv.imread(sci_path)
        filename = p.split("\\")[-1].split(".")[0]
        superpixel = np.load(p)
        height, width = superpixel.shape[:]
        result = np.zeros((height, width), dtype=np.uint8)
        for index, prediction in enumerate(graph_prediction):
            h = np.where(superpixel == index)[0]
            w = np.where(superpixel == index)[1]
            result[h, w] = prediction * 255
        cv.imwrite(Params['result'] + filename + '.png', result)
        cv.imshow("shadow_image", image)
        cv.imshow("scibble annotations", sci)
        cv.imshow("generated shadow mask", result)
        cv.waitKey()


if __name__ == '__main__':
    Params = {
        'graph': './data/graph/',
        'superpixel': './data/superpixel/',
        'result': './data/result/'
    }
    if not os.path.exists(Params['result']):
        os.makedirs(Params['result'])

    path = sorted(glob("D:\\visibility\Annotation_is_easy_wuwen\data\\superpixel\\" + "*.npy"))
    model, adj, features, idx_test = GCN_training(epochs=300, dropout=0.3)
    model.eval()
    gcn_th = 0.3
    output = model(features, adj)
    graph_prediction = (output > gcn_th).cpu().numpy().astype(np.uint8)
    generate_mask(graph_prediction, path)
