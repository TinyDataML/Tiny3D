from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import PointcloudPatchDataset, SequentialPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet import ResPCPNet
from shutil import copyfile

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='../data/pointCleanNetDataset', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', type=str, default='testset.txt', help='shape set file name')
    parser.add_argument('--modeldir', type=str, default='../models/denoisingModel', help='model folder')
    parser.add_argument('--model', type=str, default='PointCleanNet', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--n_neighbours', type=int, default=100, help='nearest neighbour used for inflation step')
    parser.add_argument('--sparse_patches', type=int, default=False, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--nrun', type=int, default=1, help='nrun')
    parser.add_argument('--shapename', type=str, default=1, help='shape to evaluate format : name{i}')

    return parser.parse_args()

def init_res_directory(opt):
    # copy input shape to results directory at first iteration
    if opt.nrun == 1:
        src = os.path.join(opt.indir, opt.shapename.rsplit("_", 1)[0] + ".xyz")
        dst = os.path.join(opt.outdir,opt.shapename.format(i = 0) + ".xyz" )
        copyfile(src, dst)


def eval_pcpnet(opt):
    # get a list of model names
    model_name = opt.model
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model_filename = os.path.join(opt.modeldir, opt.model + "_model.pth")
    param_filename = os.path.join(opt.modeldir, opt.model+opt.parmpostfix)

    # load model and training parameters
    trainopt = torch.load(param_filename)
    trainopt.outputs = ['clean_points']
    if opt.batchSize == 0:
        model_batchSize = trainopt.batchSize
    else:
        model_batchSize = opt.batchSize
    # get indices in targets and predictions corresponding to each output
    pred_dim = 0
    output_pred_ind = []
    for o in trainopt.outputs:
        if o in ['clean_points']:
            output_pred_ind.append(pred_dim)
            pred_dim += 3
        else:
            raise ValueError('Unknown output: %s' % (o))
    dataset = PointcloudPatchDataset(
        root=opt.outdir, shapes_list_file=opt.dataset,
        patch_radius=trainopt.patch_radius,
        points_per_patch=trainopt.points_per_patch,
        patch_features=['original'],
        seed=opt.seed,
        use_pca=trainopt.use_pca,
        center=trainopt.patch_center,
        point_tuple=trainopt.point_tuple,
        sparse_patches=opt.sparse_patches,
        cache_capacity=opt.cache_capacity, shape_names = [opt.shapename.format(i = opt.nrun-1)])
    if opt.sampling == 'full':
        datasampler = SequentialPointcloudPatchSampler(dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        datasampler = SequentialShapeRandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))

    regressor = ResPCPNet(
        num_points=trainopt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=trainopt.use_point_stn,
        use_feat_stn=trainopt.use_feat_stn,
        sym_op=trainopt.sym_op,
        point_tuple=trainopt.point_tuple)
    regressor.load_state_dict(torch.load(model_filename))
    regressor.cuda()

    shape_ind = 0
    shape_patch_offset = 0
    if opt.sampling == 'full':
        shape_patch_count = dataset.shape_patch_count[shape_ind]
    elif opt.sampling == 'sequential_shapes_random_patches':
        shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    shape_properties = torch.FloatTensor(shape_patch_count, pred_dim).zero_()

    # append model name to output directory and create directory if necessary
    model_outdir = os.path.join(opt.outdir, model_name)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    num_batch = len(dataloader)
    batch_enum = enumerate(dataloader, 0)


    regressor.eval()
    for batchind, data in batch_enum:

        # get batch, convert to variables and upload to GPU
        points,originals, patch_radiuses,data_trans = data
        points = Variable(points, volatile=True)
        points = points.transpose(2, 1)
        points = points.cuda()

        data_trans = data_trans.cuda()
        pred, trans, _, _ = regressor(points)
        pred = pred.data
        if trans is not None:
            trans = trans.data

        # post-processing of the prediction
        for oi, o in enumerate(trainopt.outputs):
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            if trainopt.use_point_stn:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)
            if trainopt.use_pca:
                # transform predictions with inverse pca rotation (back to world space)
                o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(1)
            n_points = patch_radiuses.shape[0]
            # new coordinates are : old coordiantes + displacement vector
            o_pred = torch.mul(o_pred, torch.t(patch_radiuses.expand(3, n_points)).float().cuda()) + originals.cuda()
            pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3] = o_pred

        print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))

        batch_offset = 0
        while batch_offset < pred.size(0):

            shape_patches_remaining = shape_patch_count-shape_patch_offset
            batch_patches_remaining = pred.size(0)-batch_offset
            # append estimated patch properties batch to properties for the current shape on the CPU
            shape_properties[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = pred[
                batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)
            if shape_patches_remaining <= batch_patches_remaining:

                # save shape properties to disk
                prop_saved = [False]*len(trainopt.outputs)

                # save clean points
                oi = [k for k, o in enumerate(trainopt.outputs) if o in ['clean_points']]
                if len(oi) > 1:
                    raise ValueError('Duplicate point output.')
                elif len(oi) == 1:
                    oi = oi[0]
                    normal_prop = shape_properties[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                    # Compute mean displacements, inspired from Taubin smoothing
                    normal_prop = get_meaned_displacements(dataset, normal_prop, opt.n_neighbours)
                    np.savetxt(os.path.join(opt.outdir,opt.shapename.format(i = opt.nrun) + '.xyz'), normal_prop.numpy())
                    prop_saved[oi] = True

                if not all(prop_saved):
                    raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')
                # start new shape
                if shape_ind + 1 < len(dataset.shape_names):
                    shape_patch_offset = 0
                    shape_ind = shape_ind + 1
                    if opt.sampling == 'full':
                        shape_patch_count = dataset.shape_patch_count[shape_ind]
                    elif opt.sampling == 'sequential_shapes_random_patches':
                        # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                        shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                    else:
                        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                    shape_properties = torch.FloatTensor(shape_patch_count, pred_dim).zero_()

def get_meaned_displacements(dataset, moved_points, n_neighbours):
    shp = dataset.shape_cache.get(0)
    nearest_neighbours = torch.tensor(shp.kdtree.query(shp.pts, n_neighbours)[1])
    displacement_vectors = moved_points - torch.tensor(shp.pts)
    new_displacement = displacement_vectors[nearest_neighbours]
    new_displacement = new_displacement.mean(1)
    new_points = moved_points - new_displacement
    return new_points

if __name__ == '__main__':
    eval_opt = parse_arguments()
    init_res_directory(eval_opt)
    eval_pcpnet(eval_opt)
