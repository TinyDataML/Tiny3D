from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import random

REAL_DATA = True
TRAINING = False
def load_shape(point_filename, normals_filename, curv_filename, pidx_filename, clean_points_filename, outliers_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None#np.load(point_filename[:-4]+'.pidx.npy')

    if clean_points_filename != None:
        clean_points = np.load(clean_points_filename+'.npy')
    else:
        clean_points= None
    if outliers_filename != None:
        outliers = np.load(outliers_filename+'.npy')
    else:
        outliers= None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)
    clean_points_kdtree = None
    if clean_points is not None:
        clean_points_kdtree = spatial.cKDTree(clean_points, 10)
    sh = Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices, clean_points = clean_points,
        clean_kdtree = clean_points_kdtree, outliers = outliers, point_filename=point_filename)
    return sh

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count



class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count



class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None, clean_points=None, clean_kdtree=None, outliers = None, point_filename=None):
        self.pts = pts
        self.kdtree = kdtree
        self.clean_kdtree = clean_kdtree
        self.normals = normals
        self.clean_points = clean_points
        self.curv = curv
        self.outliers = outliers
        seed = 3627473
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)
        indexes = np.array(range(len(pts)))
        inlier_idx = indexes[outliers==0]
        outlier_idx = indexes[outliers==1]
        random.seed(seed)
        if len(inlier_idx)>len(outlier_idx):
            majority = inlier_idx
            minority = outlier_idx
        else:
            majority = outlier_idx
            minority = inlier_idx
        balanced_idx = random.sample(list(majority), len(minority))
        balanced_idx += list(minority)
        random.shuffle(balanced_idx)
        # balance data distribution at training time (optional)
        if TRAINING:
            self.pidx = balanced_idx
        np.save(point_filename[:-4]+'.pidx.npy', balanced_idx)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shapes_list_file, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=True, center='point', point_tuple=1, cache_capacity=1, point_count_std=0.0, sparse_patches=False, eval=False):

        # initialize parameters
        self.root = root
        self.shapes_list_file = shapes_list_file
        self.patch_features = patch_features
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed
        self.include_normals = False
        self.include_curvatures = False
        self.include_clean_points = False
        self.include_original = False
        self.include_outliers = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            elif pfeat == 'clean_points':
                self.include_clean_points = True
            elif pfeat == 'original':
                self.include_original = True
            elif pfeat == 'outliers':
                self.include_outliers = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shapes_list_file)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))
        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None
            if self.include_outliers:
                outliers_filename = os.path.join(self.root, shape_name + ".outliers")
                outliers = np.loadtxt(outliers_filename).astype('float32')
                np.save(outliers_filename + '.npy', outliers)
            if self.include_clean_points:
                clean_points_filename = os.path.join(self.root, shape_name + ".clean_xyz")
                clean_points = np.loadtxt(clean_points_filename).astype('float32')
                np.save(clean_points_filename + '.npy', clean_points)
            else:
                clean_points_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                curvatures = np.loadtxt(curv_filename).astype('float32')
                np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)
            if eval:
                shape.pidx = None
            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))
            if REAL_DATA:
                bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
                self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])
            else:
                # find the radius of the ground truth points
                real_points = shape.pts[[True if x==0 else False for x in outliers]]
                bbdiag = float(np.linalg.norm(real_points.max(0) - real_points.min(0), 2))
                self.patch_radius_absolute.append([bbdiag*1 * rad for rad in self.patch_radius])



    def select_patch_points(self, patch_radius, global_point_index, center_point_ind, shape, radius_index,
    scale_ind_range, patch_pts_valid, patch_pts, clean_points=False):
        if clean_points:
            patch_point_inds = np.array(shape.clean_kdtree.query_ball_point(shape.clean_points[center_point_ind, :], patch_radius))
            # patch_point_inds = np.array(shape.clean_kdtree.query_ball_point(shape.pts[center_point_ind, :], patch_radius))
        else:
            patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], patch_radius))

        # optionally always pick the same points for a given patch index (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed((self.seed + global_point_index) % (2**32))

        point_count = min(self.points_per_patch, len(patch_point_inds))
        # randomly decrease the number of points to get patches with different point densities
        if self.point_count_std > 0:
            point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
            point_count = min(point_count, len(patch_point_inds))

        # if there are too many neighbors, pick a random subset
        if point_count < len(patch_point_inds):
            patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]
        start = radius_index*self.points_per_patch
        end = start+point_count
        scale_ind_range[radius_index, :] = [start, end]

        patch_pts_valid += list(range(start, end))

        if clean_points:
            points_base = shape.clean_points
        else:
            points_base = shape.pts

        # convert points to torch tensors
        patch_pts[start:end, :] = torch.from_numpy(points_base[patch_point_inds, :])

        # center patch (central point at origin - but avoid changing padded zeros)
        if self.center == 'mean':
            patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[start:end, :].mean(0)
        elif self.center == 'point':
            patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
        elif self.center == 'none':
            pass # no centering
        else:
            raise ValueError('Unknown patch centering option: %s' % (self.center))

        # normalize size of patch (scale with 1 / patch radius)
        patch_pts[start:end, :] = patch_pts[start:end, :] / patch_radius

        return patch_pts, patch_pts_valid, scale_ind_range


    def get_gt_point(self, index):
        shape_ind, patch_ind = self.shape_index(index)
        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]
        return shape.pts[center_point_ind]

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.FloatTensor(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3).zero_()
        patch_pts_valid = []
        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        for radius_index, patch_radius in enumerate(self.patch_radius_absolute[shape_ind]):
            patch_pts, patch_pts_valid, scale_ind_range = self.select_patch_points(patch_radius, index,
                    center_point_ind, shape, radius_index, scale_ind_range, patch_pts_valid, patch_pts)
        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        if self.include_curvatures:
            patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            patch_curv = patch_curv * self.patch_radius_absolute[shape_ind][0]
        if self.include_original:
            original = shape.pts[center_point_ind]

        if self.include_clean_points:
            tmp = []
            patch_clean_points = torch.FloatTensor(self.points_per_patch, 3).zero_()
            scale_clean_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
            clean_patch_radius = max(self.patch_radius_absolute[shape_ind])
            patch_clean_points, _, _,_ = self.select_patch_points(clean_patch_radius, index,
                    center_point_ind, shape, 0, scale_clean_ind_range, tmp, patch_clean_points, clean_points=True)
        if self.include_outliers:
            outlier = shape.outliers[center_point_ind]
        if self.use_pca:

            # compute pca of points in the patch:
            # center the patch around the mean:
            pts_mean = patch_pts[patch_pts_valid, :].mean(0)
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts[patch_pts_valid, :]))
            patch_pts[patch_pts_valid, :] = torch.mm(patch_pts[patch_pts_valid, :], trans)

            cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
            cp_new = torch.matmul(cp_new, trans)

            # re-center on original center point
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - cp_new

            if self.include_normals:
                patch_normal = torch.matmul(patch_normal, trans)
        else:
            trans = torch.eye(3).float()


        # get point tuples from the current patch
        if self.point_tuple > 1:
            patch_tuples = torch.FloatTensor(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple).zero_()
            for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
                start = scale_ind_range[s, 0]
                end = scale_ind_range[s, 1]
                point_count = end - start

                tuple_count = point_count**self.point_tuple

                # get linear indices of the tuples
                if tuple_count > self.points_per_patch:
                    patch_tuple_inds = self.rng.choice(tuple_count, self.points_per_patch, replace=False)
                    tuple_count = self.points_per_patch
                else:
                    patch_tuple_inds = np.arange(tuple_count)

                # linear tuple index to index for each tuple element
                patch_tuple_inds = np.unravel_index(patch_tuple_inds, (point_count,)*self.point_tuple)

                for t in range(self.point_tuple):
                    patch_tuples[start:start+tuple_count, t*3:(t+1)*3] = patch_pts[start+patch_tuple_inds[t], :]


            patch_pts = patch_tuples

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal,)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            elif pfeat == 'clean_points':
                patch_feats = patch_feats + (patch_clean_points,)
            elif pfeat == "original":
                patch_feats = patch_feats + (original,patch_radius)
            elif pfeat == "outliers":
                patch_feats = patch_feats + (outlier,)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))
        return (patch_pts,) + patch_feats + (trans,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        clean_points_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.clean_xyz') if self.include_clean_points else None
        outliers_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.outliers') if self.include_outliers else None
        return load_shape(point_filename, normals_filename, curv_filename, pidx_filename, clean_points_filename, outliers_filename)
