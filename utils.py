import json
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(save_path, file):
    with open(save_path, 'w') as f:
        json.dump(file, f, cls=NpEncoder)


def supervised_loss(prediction, label, weight=None):
    cost = F.nll_loss(prediction, label)
    return cost


def local_consistency_loss(P, C, S_tensor, K, sigma_P=1.0, sigma_C=1.0, omega_lc=1.0):

    N = P.shape[0]
    loss = 0.0
    valid_pairs = 0

    for i in range(N):
        if S_tensor[i] == -1:
            continue
        for j in K[i]:
            if j < 0 or j >= N or S_tensor[j] == -1:
                continue
            dist_P = torch.norm(P[i] - P[j], p=2) ** 2
            dist_C = torch.norm(C[i] - C[j], p=2) ** 2
            weight = torch.exp(-dist_P / (2 * sigma_P ** 2) - dist_C / (2 * sigma_C ** 2))
            similarity_loss = torch.abs(S_tensor[i] - S_tensor[j])
            loss += (1 / omega_lc) * weight * similarity_loss
            valid_pairs += 1
    return loss / (valid_pairs + 1e-8) if valid_pairs > 0 else torch.tensor(0.0, device=P.device)

def deterministic_loss(y_true, y_pred, omega_ul):

    mask = omega_ul.bool().unsqueeze(1)
    y_true_ul = y_true[mask].view(-1, y_true.shape[1])
    y_pred_ul = y_pred[mask].view(-1, y_pred.shape[1])

    loss = y_true_ul * torch.log(y_pred_ul + 1e-8)  # 避免 log(0)
    loss = loss.sum(dim=1)
    loss = loss.mean()

    C = y_true.shape[1]
    loss = loss / C

    return -loss


def normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
