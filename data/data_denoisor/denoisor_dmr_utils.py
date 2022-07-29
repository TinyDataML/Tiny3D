import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree

import argparse
import math

from deephub.denoisy_model.dmr.models.denoise import PointCloudDenoising
from deephub.denoisy_model.dmr.models.utils import *

from tqdm import tqdm


def normalize_pointcloud(v):
    center = v.mean(axis=0, keepdims=True)
    v = v - center
    scale = (1 / np.abs(v).max()) * 0.999999
    v = v * scale
    return v, center, scale


def run_denoise(pc, patch_size, ckpt, device, random_state=0, expand_knn=16):
    pc, center, scale = normalize_pointcloud(pc)
    print('[INFO] Center: %s | Scale: %.6f' % (repr(center), scale))

    n_clusters = math.ceil(pc.shape[0] / patch_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=16).fit(pc)

    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False, n_jobs=8)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    patches = []
    extra_points = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i
        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        extra_idx = np.setdiff1d(expand_idx, np.where(pts_idx))

        patches.append(pc[expand_idx])
        extra_points.append(pc[extra_idx])

    model = PointCloudDenoising.load_from_checkpoint(ckpt).to(device=device)

    denoised_patches = []
    downsampled_patches = []

    for patch in tqdm(patches):
        patch = torch.FloatTensor(patch).unsqueeze(0).to(device=device)
        # print(patch.size())
        with torch.no_grad():
            pred = model(patch)
            pred = pred.detach().cpu().reshape(-1, 3).numpy()

        denoised_patches.append(pred)

        downsampled_patches.append(model.model.adjusted.detach().cpu().reshape(-1, 3).numpy())

    denoised = np.concatenate(denoised_patches, axis=0)
    downsampled = np.concatenate(downsampled_patches, axis=0)

    denoised = (denoised / scale) + center
    downsampled = (downsampled / scale) + center

    return denoised, downsampled


def run_denoise_middle_pointcloud(pc, num_splits, patch_size, ckpt, device, random_state=0, expand_knn=16):
    np.random.shuffle(pc)
    split_size = math.floor(pc.shape[0] / num_splits)
    splits = []
    for i in range(num_splits):
        if i < num_splits - 1:
            splits.append(pc[i * split_size:(i + 1) * split_size])
        else:
            splits.append(pc[i * split_size:])

    denoised = []
    downsampled = []
    for i, splpc in enumerate(tqdm(splits)):
        den, dow = run_denoise(splpc, patch_size, ckpt, device, random_state, expand_knn)
        denoised.append(den)
        downsampled.append(dow)

    return np.vstack(denoised), np.vstack(downsampled)


def run_denoise_large_pointcloud(pc, cluster_size, patch_size, ckpt, device, random_state=0, expand_knn=16):
    n_clusters = math.ceil(pc.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=16).fit(pc)

    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False, n_jobs=8)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    centers = []
    patches = []
    # extra_points = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i

        raw_pc = pc[pts_idx]
        centers.append(raw_pc.mean(axis=0, keepdims=True))

        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        # extra_idx = np.setdiff1d(expand_idx, np.where(pts_idx))

        patches.append(pc[expand_idx])
        # extra_points.append(pc[extra_idx])

        print('[INFO] Cluster Size:', patches[-1].shape[0])

    denoised = []
    downsampled = []
    for i, patch in enumerate(tqdm(patches)):
        den, dow = run_denoise(patch - centers[i], patch_size, ckpt, device, random_state, expand_knn)
        den += centers[i]
        dow += centers[i]
        denoised.append(den)
        downsampled.append(dow)

    return np.vstack(denoised), np.vstack(downsampled)


def run_test(input_fn, output_fn, patch_size, ckpt, device, random_state=0, expand_knn=16, ds_output_fn=None,
             large=False, cluster_size=30000):
    pc = np.loadtxt(input_fn).astype(np.float32)
    if not os.path.exists(os.path.dirname(output_fn)):
        os.makedirs(os.path.dirname(output_fn))
    if large:
        denoised, downsampled = run_denoise_large_pointcloud(pc, cluster_size, patch_size, ckpt, device,
                                                             random_state=random_state, expand_knn=expand_knn)
    else:
        denoised, downsampled = run_denoise(pc, patch_size, ckpt, device, random_state=random_state,
                                            expand_knn=expand_knn)
    np.savetxt(output_fn, denoised)
    if ds_output_fn is not None:
        np.savetxt(ds_output_fn, downsampled)


def auto_denoise(args):
    print('[INFO] Loading: %s' % args.input)
    pc = np.loadtxt(args.input).astype(np.float32)
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    num_points = pc.shape[0]
    if num_points >= 120000:
        print('[INFO] Denoising large point cloud.')
        denoised, downsampled = run_denoise_large_pointcloud(
            pc=pc,
            cluster_size=args.cluster_size,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    elif num_points >= 60000:
        print('[INFO] Denoising middle-sized point cloud.')
        denoised, downsampled = run_denoise_middle_pointcloud(
            pc=pc,
            num_splits=args.num_splits,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    elif num_points >= 10000:
        print('[INFO] Denoising regular-sized point cloud.')
        denoised, downsampled = run_denoise(
            pc=pc,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    else:
        assert False, "Our pretrained model does not support point clouds with less than 10K points."

    np.savetxt(args.output, denoised)
    print('[INFO] Saving to: %s' % args.output)
    if args.downsample_output is not None:
        np.savetxt(args.downsample_output, downsampled)

