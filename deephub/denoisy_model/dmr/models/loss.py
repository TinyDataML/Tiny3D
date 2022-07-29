import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .utils import get_knn_idx_dist, group, gather
from deephub.denoisy_model.dmr.ops.emd.emd_module import emdFunction


def cd_loss(preds, gts):
    def batch_pairwise_dist(x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        diag_ind_x = torch.arange(0, num_points_x).to(device=x.device)
        diag_ind_y = torch.arange(0, num_points_y).to(device=y.device)

        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
    
    P = batch_pairwise_dist(gts, preds)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.sum(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.sum(mins)

    return loss_1 + loss_2


def emd_loss(preds, gts, eps=0.005, iters=50):
    loss, _ = emdFunction.apply(preds, gts, eps, iters)
    return torch.sum(loss)


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, preds, gts, **kwargs):
        return cd_loss(preds, gts)


class EMDLoss(nn.Module):

    def __init__(self, eps=0.005, iters=50):
        super().__init__()
        self.eps = eps
        self.iters = iters

    def forward(self, preds, gts, **kwargs):
        return emd_loss(preds, gts, eps=self.eps, iters=self.iters)


class ProjectionLoss(nn.Module):
    
    def __init__(self, knn=8, sigma_p=0.03, sigma_n=math.radians(15)):
        super().__init__()
        self.sigma_p = sigma_p
        self.sigma_n = sigma_n
        self.knn=knn

    def distance_weight(self, dist):
        """
        :param  dist: (B, N, k), Squared L2 distance
        :return (B, N, k)
        """
        return torch.exp(- dist / (self.sigma_p ** 2))

    def angle_weight(self, nb_normals):
        """
        :param  nb_normals: (B, N, k, 3), Normals of neighboring points 
        :return (B, N, k)
        """
        estm_normal = nb_normals[:, :, 0:1, :]  # (B, N, 1, 3)
        inner_prod = (nb_normals * estm_normal.expand_as(nb_normals)).sum(dim=-1)   # (B, N, k)
        return torch.exp(- (1 - inner_prod) / (1 - math.cos(self.sigma_n)))

    def forward(self, preds, gts, normals, **kwargs):
        knn_idx, knn_dist = get_knn_idx_dist(gts, query=preds, k=self.knn, offset=0) # (B, N, k), squared L2 distance
        
        nb_points = group(gts, idx=knn_idx)         # (B, N, k, 3)
        nb_normals = group(normals, idx=knn_idx)    # (B, N, k, 3)

        distance_w = self.distance_weight(knn_dist) # (B, N, k)
        angle_w = self.angle_weight(nb_normals)     # (B, N, k)
        weights = distance_w * angle_w  # (B, N, k)

        inner_prod = ((preds.unsqueeze(-2).expand_as(nb_points) - nb_points) * nb_normals).sum(dim=-1)  # (B, N, k)
        inner_prod = torch.abs(inner_prod)  # (B, N, k)
        
        point_displacement = (inner_prod * weights).sum(dim=-1) / weights.sum(dim=-1)   # (B, N)
        
        return point_displacement.sum()
        

class UnsupervisedLoss(nn.Module):
    
    def __init__(self, k=64, radius=0.05, pdf_std=0.5657, inv_scale=0.05, decay_epoch=80, emd_eps=0.005, emd_iters=50):
        super().__init__()
        self.knn = k
        self.radius = radius
        self.pdf_std = pdf_std
        self.inv_scale = inv_scale
        self.decay_epoch = decay_epoch
        self.emd_eps = emd_eps
        self.emd_iters = emd_iters

    def stochastic_neighborhood(self, inputs):
        """
        param:  inputs:  (B, N, 3)
        return: knn_idx: (B, N, k), Indices of neighboring points
        return: mask:    (B, N, k), Mask
        """
        knn_idx, knn_dist = get_knn_idx_dist(inputs, query=inputs, k=self.knn, offset=1)    # (B, N, k), exclude self

        # Gaussian spatial prior
        SQRT_2PI = 2.5066282746
        prob = torch.exp(- (knn_dist / (self.inv_scale ** 2)) / (2 * self.pdf_std ** 2)) / (self.pdf_std * SQRT_2PI)    # (B, N, k)
        mask = torch.bernoulli(prob)    # (B, N, k)

        prob = prob * (torch.sqrt(knn_dist) <= self.radius) # Radius cutoff

        # If all the neighbor of a point are rejected, then accept at least one to avoid zero loss
        # Here we accept the farthest one, because all-rejected probably happens when the point is displaced too far (high noise). 
        mask_sum = mask.sum(dim=-1, keepdim=True)   # (B, N, 1)
        mask_farthest = torch.where(mask_sum == 0, torch.ones_like(mask_sum), torch.zeros_like(mask_sum))       # (B, N, 1)
        mask_delta = torch.cat([torch.zeros_like(mask_sum).repeat(1, 1, self.knn-1), mask_farthest], dim=-1)    # (B, N, k)
        mask = mask + mask_delta    # (B, N, k)
        
        return knn_idx, mask, prob

    def forward(self, preds, inputs, epoch, **kwargs):
        _, assignment = emdFunction.apply(inputs, preds, self.emd_eps, self.emd_iters) # (B, N), assign each input point to a predicted point

        # Permute the predicted points according to the assignment
        # One-to-one correspondent to input points
        permuted_preds = gather(preds, idx=assignment.long())  # (B, N, 3)

        input_nbh_idx, input_nbh_mask, _ = self.stochastic_neighborhood(inputs)    # (B, N, k), (B, N, k)
        input_nbh_pos = group(inputs, idx=input_nbh_idx)    # (B, N, k, 3)

        dist = (permuted_preds.unsqueeze(dim=-2).expand_as(input_nbh_pos) - input_nbh_pos)  # (B, N, k, 3)
        dist = (dist ** 2).sum(dim=-1)  # (B, N, k), squared-L2 distance

        num_nbh = input_nbh_mask.sum(dim=-1)    # (B, N), number of neighbors
        avg_dist = (dist * input_nbh_mask).sum(dim=-1) / num_nbh    # (B, N), average distance

        return avg_dist.sum()
        

def get_loss_layer(name):
    if name == 'emd':
        return EMDLoss()
    elif name == 'cd':
        return ChamferLoss()
    elif name == 'proj':
        return ProjectionLoss()
    elif name == 'unsupervised':
        return UnsupervisedLoss()
    elif name is None or name == 'None':
        return None
    else:
        raise ValueError('Unknown loss: %s ' % name)


class RepulsionLoss(nn.Module):

    def __init__(self, knn=4, h=0.03):
        super().__init__()
        self.knn = knn
        self.h = h

    def forward(self, pc):
        knn_idx, knn_dist = get_knn_idx_dist(pc, pc, k=self.knn, offset=1)  # (B, N, k)
        weight = torch.exp(- knn_dist / (self.h ** 2))
        loss = torch.sum(- knn_dist * weight)
        return loss
