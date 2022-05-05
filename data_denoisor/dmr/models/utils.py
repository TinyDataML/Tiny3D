import torch

import numpy as np
import sklearn.neighbors
import time


def get_knn_idx_dist(pos:torch.FloatTensor, query:torch.FloatTensor, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query  = query.unsqueeze(2).expand(B, M, N, F)   # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)   # B * M * N
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k+offset]   # B * M * k
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)           # B * M * k

    return knn_idx, knn_dist
    

def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    knn_idx, _ = get_knn_idx_dist(pos=pos, query=query, k=k, offset=offset)

    return knn_idx


def group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def gather(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M)
    :return (B, M, F)
    """
    # x       : B * N * F
    # idx     : B * M
    # returns : B * M * F
    B, N, F = tuple(x.size())
    _, M    = tuple(idx.size())

    idx = idx.unsqueeze(2).expand(B, M, F)

    return torch.gather(x, dim=1, index=idx)


def feature_interp(k, feat, pos, pos_new, avg_dist, feat_new=None, avg_feat_diff=None):
    """
    :param  feat:     (B, N, F)
    :param  pos:      (B, N, 3)
    :param  pos_new:  (B, M, 3)
    :param  feat_new: (B, M, F)
    :return (B, M, F)
    """
    knn_idx = get_knn_idx(pos, pos_new, k=k, offset=0)
    pos_grouped = group(pos, idx=knn_idx)    # (B, M, k, 3)
    feat_grouped = group(feat, idx=knn_idx)  # (B, M, k, F)

    d_pos = ((pos_grouped - pos_new.unsqueeze(-2).expand_as(pos_grouped)) ** 2).sum(dim=-1)     # (B, M, k)
    weight = - d_pos / (avg_dist ** 2)

    if feat_new is not None:
        d_feat = ((feat_grouped - feat_new.unsqueeze(-2).expand_as(feat_grouped)) ** 2).sum(dim=-1) # (B, M, k)
        weight = weight - d_feat / (avg_feat_diff ** 2)

    weight = weight.softmax(dim=-1)   # (B, M, k)
    return (feat_grouped * weight.unsqueeze(-1).expand_as(feat_grouped)).sum(dim=-2)


def get_1d_mesh(steps, start=-0.2, end=0.2):
    return torch.linspace(start=start, end=end, steps=steps).unsqueeze(-1)

def get_2d_mesh(steps, start=-0.2, end=0.2):
    mesh_1d = get_1d_mesh(steps=steps, start=start, end=end).flatten()
    return torch.cartesian_prod(mesh_1d, mesh_1d)

def get_mesh(dim, steps, start=-0.2, end=0.2):
    assert dim in (1, 2)
    if dim == 1:
        return get_1d_mesh(steps, start=start, end=end)
    elif dim == 2:
        return get_2d_mesh(steps, start=start, end=end)

def get_sample_points(dim, samples, num_points=1, num_batch=None, start=-0.3, end=0.3):
    length = end - start
    if num_batch is None:
        size = [samples * num_points, dim]
    else:
        size = [num_batch, samples * num_points, dim]
    return (torch.rand(size) * length) - (length / 2)

if __name__ == '__main__':
    nbatch = 1
    npts = 5
    mesh_dim = 2
    bth = torch.arange(0, nbatch * npts * 3).float().reshape([nbatch, npts, 3]).repeat(1, 4, 1)
    print(bth)
    mesh = get_2d_mesh(2)
    print(mesh)

    mesh = mesh.unsqueeze(-1).repeat(1, 1, npts).transpose(1, 2).reshape(1, -1, mesh_dim).repeat(nbatch, 1, 1)
    print(mesh.size())
    # expanded = torch.cat([bth, mesh], dim=-1)
    # print(torch.cat([bth, mesh], dim=-1))

class Aggregator(torch.nn.Module):

    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class FullyConnected(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))

        
def normalize_point_cloud(pc, center=None, scale=None):
    """
    :param  pc: (B, N, 3)
    :return (B, N, 3)
    """
    if center is None:
        center = torch.mean(pc, dim=-2, keepdim=True).expand_as(pc)
    if scale is None:
        scale, _ = torch.max(pc.reshape(pc.size(0), -1).abs(), dim=1, keepdim=True)
        scale = scale.unsqueeze(-1).expand_as(pc)
    norm = (pc - center) / scale
    return norm, center, scale


def denormalize_point_cloud(pc, center, scale):
    """
    :param  pc: (B, N, 3)
    :return (B, N, 3)
    """
    return pc * scale + center
