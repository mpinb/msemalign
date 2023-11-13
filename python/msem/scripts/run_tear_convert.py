#!/usr/bin/env python3

# script to convert a downloaded webknossos tear annotation file into stored dills
#   that are easily loadable in the msem package without the pain of dealing with
#   the webknossos dependency conflicts.

# script additionally does all the preprocessing required for the tear fixing,
#   up to the point of the actual image manipulation (done by run_regions).
# added option to do the actual image manipulation on downsampled images, i.e.,
#   image at the same resolution that the tear was annotated in (usually 256 nm).

import numpy as np

import scipy.interpolate as interp
import scipy.spatial.distance as scidist
import scipy.ndimage as nd

from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
# did not install sklearnex in wkw setup
#from sklearnex.neighbors import NearestNeighbors
import skfmm

import hdf5plugin
hdf5plugin.FILTERS # avoid linter warning
import h5py

# NOTE: except for def_common_params, do not rely on msem package in this script
#   as to avoid conflicts with the exact-version-specific and extensive webknossos dependencies.
import webknossos as wk

import dill
import argparse
import os
import time
import shutil

import tifffile
from matplotlib import pyplot as plt

# all parameters loaded from an experiment-specific import.
# NOTE: def_common_params should load without any other msem dependencies, so only need it in the PYTHONPATH,
#   that is, a (conda) environment with the rest of the msem dependencies need not be loaded.
from def_common_params import get_paths, total_nwafers, all_wafer_ids #, legacy_zen_format
from def_common_params import meta_folder, meta_dill_fn_str, wafer_region_prefix_str, order_txt_fn_str, region_suffix
from def_common_params import tears_subfolder, torn_regions, tear_annotation_ds
from def_common_params import region_manifest_cnts, region_include_cnts


### argparse
parser = argparse.ArgumentParser(description='run_regions')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
    help='specify wafer(s) for the regions to run')
parser.add_argument('--region_inds', nargs='+', type=int, default=[],
    help='list of region indices to run (default all torn regions in wafer)')
parser.add_argument('--all-wafers', dest='all_wafers', action='store_true',
    help='instead of specifying wafer id(s), include all wafers for dataset')
parser.add_argument('--run-type', nargs=1, type=str, default=['process'],
    choices=['process', 'nml-to-region', 'xregion-to-xnml', 'xnml-to-xregion', 'warp'],
    help='the type of run to choose')
parser.add_argument('--fit-ends-dist-um', nargs=1, type=float, default=[64.],
    help='Distance to fit at end of skeletons to project to edge of image')
parser.add_argument('--interpolation-sampling-um', nargs=1, type=float, default=[10.24],
    help='Distance to fit at end of skeletons to project to edge of image')
parser.add_argument('--nml-file', nargs=1, type=str, default=[''],
    help='specify the nml file containing the control points')
parser.add_argument('--nml-file-out', nargs=1, type=str, default=[''],
    help='specify the nml file to save transformed control points to')
parser.add_argument('--warp-image-in-out', nargs=2, type=str, default=['',''],
    help='specify input/output dirs with images to warp, uses "prefix" names')
parser.add_argument('--tear-annotation-ds', nargs=1, type=int, default=[-1],
    help='override the value from def_common_params')
parser.add_argument('--print-only', dest='print_only', action='store_true',
    help='just show the list of torn slices and unordered / ordered zinds')
parser.add_argument('--doplots', dest='doplots', action='store_true',
    help='show (debug) plots')
args = parser.parse_args()
args = vars(args)


### params that are set by command line arguments

# starting at 1 (Zeiss numbering)
region_inds = args['region_inds']

# wafer starting at 1, used internally for numbering processed outputs, does not map to any zeiss info
wafer_ids = args['wafer_ids']

# the name of the nml file downloaded from webknossos
nml_fn = args['nml_file'][0]

# the name of the nml file to export transformed control points to
nml_fn_out = args['nml_file_out'][0]

# for processing of endpoints in order to create a complete
#   cutting path for each side of the tear.
fit_ends_dist_um = args['fit_ends_dist_um'][0]

# how dense to interpolate points along tear and add points beyond tear
interpolation_sampling_um = args['interpolation_sampling_um'][0]

# optionally utilize pre-computed tear information to apply warping.
# image must be at the same downsampling level (resolution) as the tear was annotated in.
# specify input and output dirs (uses "prefix-standard" names for actual images)
warp_image_in_dn = args['warp_image_in_out'][0]
warp_image_out_dn = args['warp_image_in_out'][1]

# for (debug) plots
doplots = args['doplots']

# just show the list of torn slices and unordered / ordered zinds
print_only = args['print_only']

# option to override tear_annotation_ds
use_ds = args['tear_annotation_ds'][0] if args['tear_annotation_ds'][0] > 0 else tear_annotation_ds

# these specify what type of run this is (one of these must be set True)
run_type = args['run_type'][0]
run_process = run_type == 'process'
run_nml_to_region = run_type == 'nml-to-region'
run_xregion_to_xnml = run_type == 'xregion-to-xnml'
run_xnml_to_xregion = run_type == 'xnml-to-xregion'
run_warp = run_type == 'warp'

## fixed parameters not exposed in def_common_params

# xxx - hacky over-ride possbility
#torn_regions = [[], # no wafer 0, leave blank
#    [9, 53, 107, 113, 115, 116, 151, 196, 351, 361, 461, 545, 570, 678, 825, 942, 1333, 1918, 1944], # 19 wafer 1
#    # 19 wafer 2
#    [203, 278, 646, 681, 686, 758, 774, 809, 826, 1016, 1110, 1144, 1150, 1160, 1534, 1564, 1569, 1599, 1807],
#    [], # wafer 3
#    ] # 38 total

# xxx - hacky lookup region ids possibility
#tear_slices = [520, 1876, 1919, 1921, 1929, 1953, 785, 823, 825, 2151,
#               897, 2429, 1626, 2431, 2586, 2587, 2588, 2589, 2590, 2591, ]
tear_slices = None

## constants

# pre-generate the 2D structuring elements
conn4 = nd.generate_binary_structure(2,1)
conn8 = nd.generate_binary_structure(2,2)


## helper functions

# xxx - meh, maybe start a separate repo with MLS (and add TPS?)
#from msem.utils import mls_rigid_transform
# better grid point interpolation (and extrapolation) method than using normal interpolation.
# used for filling in outliers deltas as part of the fine alignment process.
# NOTE: this is a fully vectorized implementation, meaning that this will be very memory inefficient
#   if you try to apply it at too many points (i.e., all pixel locations in a large image).
def mls_rigid_transform(_v, p, q, alpha=1.0, tol=1e-6):
    ''' Rigid transform
    Image deformation using moving least squares
       see paper of same name by Schaefer et al
    ### Params:
        * v - ndarray: an array with size [m, 2], points to transform
        * p - ndarray: an array with size [n, 2], control points
        * q - ndarray: an array with size [n, 2], deformed control points
        * alpha - float: parameter used by weights
    ### Return:
        f_r - ndarray: an array with size [m, 2], deformed points from v
    '''

    # <<< avoid divide by zero in the weight calculation
    # use a tolerance (something less than 1), because points in v that are very close to those in p
    #   can also cause MLS to explode.
    knbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(p)
    kdist, knnbrs = knbrs.kneighbors(_v, return_distance=True)
    nsel_v = (kdist < tol).reshape(-1)
    any_v_in_p = nsel_v.any()
    if any_v_in_p:
        #print('{} points in query ({}) within {} of control ({})'.format(nsel_v.sum(), _v.shape[0], tol, p.shape[0]))
        closest_p_in_v = knnbrs.reshape(-1)[nsel_v]
        sel_v = np.logical_not(nsel_v)
        v = _v[sel_v,:]
    else:
        del nsel_v
        v = _v
    del knbrs, kdist, knnbrs
    # avoid divide by zero in the weight calculation >>>

    m, n = v.shape[0], p.shape[0]
    assert( q.shape[0] == n )

    # shape (n,m,2)
    p_ij = p[:,None,:]
    q_ij = q[:,None,:]
    v_ij = v[None,:,:]

    # weights, shape (n,m)
    w = 1.0 / np.sum((p_ij - v_ij)**2, axis=2)**alpha
    w_ij = w[:,:,None]
    sum_w = np.sum(w, axis=0)

    # centroids
    p_star = (w_ij * p_ij).sum(0) / sum_w[:,None] # shape (m,2)
    q_star = (w_ij * q_ij).sum(0) / sum_w[:,None] # shape (m,2)
    p_star_ij = p_star[None,:,:]
    q_star_ij = q_star[None,:,:]
    p_hat_ij = p_ij - p_star_ij
    q_hat_ij = q_ij - q_star_ij

    # similarity transform solved matrix
    p_hat_ij_uT = np.zeros_like(p_hat_ij)
    # from paper: (_uT) is an operator on 2D vectors such that (x, y) = (−y, x).
    p_hat_ij_uT[:,:,0] = -p_hat_ij[:,:,1]
    p_hat_ij_uT[:,:,1] = p_hat_ij[:,:,0]
    v_minus_p_star = v_ij - p_star_ij
    v_minus_p_star_uT = np.zeros_like(v_minus_p_star)
    v_minus_p_star_uT[:,:,0] = -v_minus_p_star[:,:,1]
    v_minus_p_star_uT[:,:,1] = v_minus_p_star[:,:,0]
    # transform matrix, shape (n,m,2,2)
    A_ij = w_ij[:,:,:,None] * np.matmul(\
            np.concatenate((p_hat_ij[:,:,:,None], -p_hat_ij_uT[:,:,:,None]), axis=3),
            np.concatenate((v_minus_p_star[:,:,:,None], -v_minus_p_star_uT[:,:,:,None]), axis=3).transpose(0,1,3,2))

    # f_r, shape (m,2)
    f_bar_r = np.matmul(q_hat_ij[:,:,None,:], A_ij).sum(0).reshape(m,2)
    v_minus_p_star = v_minus_p_star.reshape(m,2)
    _f_r = np.sqrt((v_minus_p_star**2).sum(1))[:,None] * f_bar_r / np.sqrt((f_bar_r**2).sum(1))[:,None] + q_star

    # <<< avoid divide by zero in the weight calculation
    if any_v_in_p:
        f_r = np.zeros_like(_v)
        f_r[sel_v,:] = _f_r
        # just the deformed control points for points in v that are very close to points in p
        f_r[nsel_v,:] = q[closest_p_in_v,:]
    else:
        f_r = _f_r
    # avoid divide by zero in the weight calculation >>>

    return f_r

def ANN(A):
    """All nearest neighbors algorithm.
    This is a greedy algorithm for solving the Traveling Salesman Problem.
    Takes the minimum path for the nearest neighbor greedy algorithm
    across all possible starting nodes.

    Args:
        A (ndarray shape (nnodes, nnodes)): pairwise distance matrix
            between nodes (locations / cities).

    Returns:
        min_path (list int): ordering of nodes corresponding to shortest TSP route.
        min_cost (float): corresponding cost (distance) of the min_path route.

    """
    N = A.shape[0]
    min_path = None
    min_cost = np.inf
    for i in range(N):
        path, cost = NN(A, i)
        if cost < min_cost:
            min_path = path
            min_cost = cost
    return min_path, min_cost

def NN(A, start):
    """Nearest neighbor algorithm.
    This is a greedy algorithm for solving the Traveling Salesman Problem.

    NOTE:
        https://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm

    Args:
        A (ndarray shape (nnodes, nnodes)): pairwise distance matrix
            between nodes (locations / cities).
        start (int): which node to start at.

    Returns:
        min_path (list int): ordering of nodes corresponding to shortest TSP route.
        min_cost (float): corresponding cost (distance) of the min_path route.

    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost

# https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
def interpolate_path(points, sampling=0.5):
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=-1 )), axis=0 )
    max_distance = distance[-1]
    distance = np.insert(distance, 0, 0, axis=0) / max_distance
    if distance.ndim > 1:
        distance = distance.reshape(points.shape[0], -1).max(1)

    alpha = np.linspace(0, 1, np.ceil(max_distance.max() / sampling).astype(np.int64))
    #interpolations_methods = ['slinear', 'quadratic', 'cubic']
    interpolator =  interp.interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)

def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return (P[4] and P.sum()==2)

# def lineBranches(P):
#     """Central pixel and three others must be set to be a line branch"""
#     return (P[4] and P.sum()==4)

def get_pts_labels(pts_bw):
    pts_lbls, npts = nd.label(pts_bw, structure=conn8)
    # get the points in the same order as the labeled points
    pts = np.zeros((npts,2), dtype=np.int64)
    for i in range(npts):
        pts[i,:] = np.transpose(np.nonzero(pts_lbls==(i+1)))[:,::-1].mean(0)
    return pts_lbls, npts, pts

def robust_line_fit(clf, pts, invert=False):
    xstd = pts[:,0].std(); ystd = pts[:,1].std()
    if invert and (xstd == 0 or ystd == 0):
        return None, None
    if ((xstd > ystd) and not invert) or ((xstd <= ystd) and invert):
        X = pts[:,0].reshape(-1,1); y = pts[:,1].reshape(-1,1)
        fit_x = False
    else:
        X = pts[:,1].reshape(-1,1); y = pts[:,0].reshape(-1,1)
        fit_x = True
    xi = [1,0] if fit_x else [0,1]
    clf.fit(X, y)
    return xi, fit_x

def dense_predict(connect_pts, clf, fit_x, xi, npts=None):
    if connect_pts[0][xi[0]] < connect_pts[1][xi[0]]:
        if npts is None:
            X = np.arange(connect_pts[0][xi[0]], connect_pts[1][xi[0]]+1)[:,None]
        else:
            X = np.linspace(connect_pts[0][xi[0]], connect_pts[1][xi[0]], npts)[:,None]
    else:
        if npts is None:
            X = np.arange(connect_pts[1][xi[0]], connect_pts[0][xi[0]]+1)[:,None]
        else:
            X = np.linspace(connect_pts[1][xi[0]], connect_pts[0][xi[0]], npts)[:,None]
    y = clf.predict(X)
    line_pts = np.concatenate((y,X), axis=1) if fit_x else np.concatenate((X,y), axis=1)
    return line_pts

def add_line_label(boundary, pts, img_shape, lbl):
    tmp = np.round(pts).astype(np.int64)
    tmp[tmp < 0] = 0
    tmp[tmp[:,0] >= img_shape[1],0] = img_shape[1]-1
    tmp[tmp[:,1] >= img_shape[0],1] = img_shape[0]-1
    boundary[tmp[:,1], tmp[:,0]] = lbl

def path_to_skel(path_points):
    tmp = np.round(path_points).astype(np.int64)
    bwskel = np.zeros(img_shape_ds, dtype=bool)
    bwskel[tmp[:,1], tmp[:,0]] = 1
    # SO - 56131466/detecting-start-and-end-point-of-line-in-image-numpy-array
    endpts_bw = nd.generic_filter(bwskel, lineEnds, (3,3))
    endpts_lbls, nendpts, endpts = get_pts_labels(endpts_bw)
    assert( nendpts == 2 ) # something is wrong
    return bwskel, endpts_lbls, nendpts, endpts

def tear_skeleton_to_segments(bwskel, endpts, endpts_lbls, figbase=0):
    nendpts = endpts.shape[0]
    dilated_skel = nd.binary_dilation(bwskel, structure=conn8)

    ## do linear fits at each end of the skeleton
    print('Linear fit {} um {} pix from skeleton ends'.format(fit_ends_dist_um, fit_ends_dist))
    t = time.time()
    fit_x = [None]*nendpts; intersect = np.zeros((nendpts,2), dtype=np.double)
    line_pts = [None]*nendpts
    max_line_dist = 0.
    for i in range(nendpts):
        # get the distance transform along the skeleton
        # https://stackoverflow.com/questions/28187867/geodesic-distance-transform-in-python
        m = np.ones(img_shape_ds, dtype=bool); m[endpts_lbls==(i+1)] = 0
        d = skfmm.distance(np.ma.masked_array(nd.distance_transform_edt(m, return_indices=False,
                return_distances=True), np.logical_not(dilated_skel)))
        dmax = d.max()
        if dmax > max_line_dist: max_line_dist = dmax

        # fit line to some distance along end of skeleton
        pts = np.transpose(np.nonzero(d < fit_ends_dist))[:,::-1]
        xi, fit_x[i] = robust_line_fit(clf[i], pts)

        # NOTE: this does not work because linear regression is not the same
        #   when you flip the independent / dependent variables.
        # # use the invert fits just to find region image boundary intersection
        # ixi, ifit_x = robust_line_fit(iclf, pts, invert=True)

        # find the closest point where the line intersects with the boundary of the region image.
        # if there are two endpoints (xxx - currently all that is supported),
        #   then only take the closest intersection point that is also further away from the other endpoint.
        X = np.array([0,img_shape_ds[xi[1]]-1]); y = clf[i].predict(X.reshape(2,1))
        intersects = np.concatenate((y,X[:,None]), axis=1) \
                if fit_x[i] else np.concatenate((X[:,None],y), axis=1)
        if np.abs(clf[i].coef_[0]) > 1e-10:
            X = np.array([0,img_shape_ds[xi[0]]-1])
            y = (X.reshape(2,1) - clf[i].intercept_) / clf[i].coef_[0]
            tmp = np.concatenate((y,X[:,None]), axis=1) \
                    if not fit_x[i] else np.concatenate((X[:,None],y), axis=1)
            intersects = np.concatenate((intersects, tmp), axis=0)
        dcur = ((intersects - endpts[i,:])**2).sum(1)
        if nendpts == 2:
            dother = ((intersects - endpts[(i + 1) % 2,:])**2).sum(1)
            dcur[dother < dcur] = np.inf
        intersect[i,:] = intersects[np.argmin(dcur),:]

        # create a line segment from the image boundary intersect to the furthest point away
        #   from the intersect that was used in the fit.
        j = np.argmax(((intersect[i,:][None,:] - pts)**2).sum(1))
        line_pts[i] = dense_predict((intersect[i,:], pts[j,:]), clf[i], fit_x[i], xi)

    if doplots:
        plt.figure(1237 + figbase); plt.gcf().clf(); plt.title('skel end fits'); plt.imshow(bwskel)
        plt.scatter(endpts[:,0], endpts[:,1], c='r', s=36, marker='x')
        plt.scatter(intersect[:,0], intersect[:,1], c='r', s=36, marker='o')
        for i in range(nendpts):
            plt.plot(line_pts[i][:,0], line_pts[i][:,1], c='g', linewidth=2)

    print('\tdone in %.4f s' % (time.time() - t, ))

    # complete the boundary by connecting the tear end points with the line segments
    print('Complete boundary and make image segments'); t = time.time()
    boundary = np.zeros(img_shape_ds, dtype=np.int8); boundary[bwskel] = nendpts+1
    clf2 = linear_model.LinearRegression(fit_intercept=True)
    for i in range(nendpts):
        # find closet point in line segment to the endpoint
        j = np.argmin(((line_pts[i] - endpts[i,:][None,:])**2).sum(1))

        # if the line does not intersect the endpoint, then connect at the closest point
        closest_pt = np.round(line_pts[i][j,:])
        if (closest_pt != endpts[i,:]).any():
            pts = np.concatenate((closest_pt[None,:], endpts[i,:][None,:]), axis=0)
            xi, fit_x2 = robust_line_fit(clf2, pts)
            line_pts2 = dense_predict((pts[0,:], pts[1,:]), clf2, fit_x2, xi)
            add_line_label(boundary, line_pts2, img_shape_ds, i+1)

        xi = [1,0] if fit_x[i] else [0,1]
        line_pts2 = dense_predict((intersect[i,:], closest_pt), clf[i], fit_x[i], xi)
        add_line_label(boundary, line_pts2, img_shape_ds, i+1)

    # remove any objects created inside of the boundary
    tmp = np.ones(img_shape_ds, dtype=bool); tmp[boundary.astype(bool)] = 0
    labels, nlbls = nd.label(np.pad(tmp,2,constant_values=1), structure=conn4)
    if nlbls > 0:
        add = np.arange(1,nlbls+1); add = add[add != labels[0,0]]
        if add.size > 0:
            boundary[np.isin(labels[2:-2,2:-2], add)] = nendpts+1

    tmp = np.ones(img_shape_ds, dtype=bool); tmp[boundary.astype(bool)] = 0
    tear_segments, ntear_segments = nd.label(tmp, structure=conn4)
    tear_segments = tear_segments.astype(np.int8) # save memory, should never be more than a few tears
    print('\tdone in %.4f s' % (time.time() - t, ))

    if doplots:
        plt.figure(1238 + figbase); plt.gcf().clf(); plt.title('boundary'); plt.imshow(boundary)
        plt.scatter(endpts[:,0], endpts[:,1], c='r', s=36, marker='x')
        plt.scatter(intersect[:,0], intersect[:,1], c='r', s=36, marker='o')
        plt.figure(1239 + figbase); plt.gcf().clf(); plt.title('tear segments'); plt.imshow(tear_segments)
        if ntear_segments != 2:
            # for debug
            for i in range(ntear_segments):
                print(np.transpose(np.nonzero(tear_segments==i+1))[0,:])
            plt.show() # for debug
    assert( ntear_segments==2 ) # boundary is messed up, common error, keep this after plot for debug

    print('Get intersection between control point segments and boundary'); t = time.time()
    boundary_pts = np.transpose(np.nonzero(boundary))[:,::-1]
    clf2 = linear_model.LinearRegression(fit_intercept=True)
    knbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(boundary_pts)
    nodes_middles = np.empty((nskels,3,2), dtype=np.double); nodes_middles.fill(np.nan)
    for i in range(nskels):
        # put the corespondence points in order depending on which label they are in
        for j in range(2):
            inode = np.round(cnodes[i,j,:]).astype(np.int64)
            assert(tear_segments[inode[1], inode[0]] != 0) # correspondence point on boundary
            im = tear_segments[inode[1], inode[0]] - 1
            assert( not np.isfinite(nodes_middles[i,im,:]).any() ) # nodes both on same side of tear
            nodes_middles[i,im,:] = cnodes[i,j,:]
        # put the intersecting point between the boundary and the line segment last
        xi, fit_x2 = robust_line_fit(clf2, cnodes[i,:,:])
        line_pts2 = dense_predict((cnodes[i,0,:], cnodes[i,1,:]), clf2, fit_x2, xi)
        kdist, knnbrs = knbrs.kneighbors(line_pts2, return_distance=True)
        nodes_middles[i,2,:] = line_pts2[np.argmin(kdist.reshape(-1)),:]
    print('\tdone in %.4f s' % (time.time() - t, ))

    return boundary, tear_segments, nodes_middles, intersect, fit_x
#def tear_skeleton_to_segments(

def big_img_info(fn, dataset='image'):
    fh = h5py.File(fn, 'r')
    image = fh[dataset]
    img_shape = image.shape; img_dtype = image.dtype
    fh.close()
    return img_shape, img_dtype

# super-light save/load relative to msem utils

def big_img_save(fn, data, dataset_name):
    fh = h5py.File(fn, 'a')
    if dataset_name in fh: del fh[dataset_name]
    dataset = fh.create_dataset(dataset_name, data.shape, dtype=data.dtype)
    dataset[:] = data
    fh.close()

def big_img_load(fn, dataset_name):
    fh = h5py.File(fn, 'r')
    dataset = fh[dataset_name]
    data = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(data)
    fh.close()
    return data



### inits based on params

# set wafer_ids to contain all wafers, if specified
if args['all_wafers']:
    wafer_ids = list(all_wafer_ids)

if tear_slices is None:
    assert( torn_regions is not None and all([x is not None for x in wafer_ids]) ) # torn_regions not defined
#assert( not legacy_zen_format ) # did not implement, problematic with region_strs

cum_manifest_cnts = np.concatenate(([0], np.cumsum(region_manifest_cnts[1:])))
cum_include_cnts = np.concatenate(([0], np.cumsum(region_include_cnts[1:])))

meta_dill_fn = os.path.join(meta_folder, meta_dill_fn_str)
with open(meta_dill_fn, 'rb') as f: meta_dict = dill.load(f)
fit_ends_dist = fit_ends_dist_um*meta_dict['scale_um_to_pix'] / use_ds
interpolation_sampling = interpolation_sampling_um*meta_dict['scale_um_to_pix'] / use_ds

if warp_image_out_dn:
    warp_image_out_orig_dn = os.path.join(warp_image_out_dn,'orig')
    os.makedirs(warp_image_out_orig_dn, exist_ok=True)

if nml_fn:
    print('Loading tear nml'); t = time.time()
    # open the nml and retrieve all the groups.
    # NOTE: annotations should be in groups, where the groups are for each slice,
    #   and the group for a single slice is named with the slice prefix (see below).
    nml = wk.Skeleton.load(nml_fn)
    groups = list(nml.children)
else:
    nml = groups = None

if nml_fn_out:
    annotation = wk.Annotation(
        name="xformed_tear_control_points", dataset_name=nml.dataset_name, voxel_size=nml.voxel_size
    )

# iterate over all the torn slices in the specified wafers
total_nskels = 0
for wafer_id, wafer_ind in zip(all_wafer_ids, range(total_nwafers)):
    if wafer_id not in wafer_ids: continue
    _, _, _, alignment_folder, _, region_strs = get_paths(wafer_id)
    nregions = sum([len(x) for x in region_strs])
    # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
    region_strs_flat = [item for sublist in region_strs for item in sublist]
    order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))
    solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based

    use_region_inds = tear_slices if tear_slices is not None else torn_regions[wafer_id]
    if tear_slices is None and len(region_inds) > 0:
        use_region_inds = np.intersect1d(region_inds, use_region_inds)
    for slice_or_region_ind in use_region_inds:
        if tear_slices is not None:
            wafer_zind = slice_or_region_ind - cum_include_cnts[wafer_ind]
            if wafer_zind >= 0 and wafer_zind < region_include_cnts[wafer_id]:
                region_ind = solved_order[wafer_zind] + 1
            else:
                continue
        else:
            region_ind = slice_or_region_ind
        region_str = region_strs_flat[region_ind-1]
        zind = cum_manifest_cnts[wafer_ind] + region_ind - 1

        prefix = wafer_region_prefix_str.format(wafer_id, region_str)
        solved_zind = np.nonzero(solved_order == region_ind-1)[0][0] + cum_include_cnts[wafer_ind]
        print('{: <25} {: <6} {: <6} {}'.format(prefix, region_ind, zind, solved_zind))
        if print_only: continue
        slice_image_fn = os.path.join(alignment_folder, prefix + region_suffix + '.h5')
        annotation_out_fn = os.path.join(alignment_folder, tears_subfolder, prefix + '_annotation.dill')
        export_prefix = 'wafer{:02d}_manifest{:05d}_{}'.format(wafer_id, region_ind-1, region_str)
        #image_fn = export_prefix + region_suffix + '.tif'
        image_fn = export_prefix + region_suffix + '.tiff'
        print(slice_image_fn)

        # get the shape of the region image (should already be saved at this point)
        img_shape, img_dtype = big_img_info(slice_image_fn)
        img_shape_ds = np.ceil(np.array(img_shape, dtype=np.double) / use_ds).astype(np.int64)
        print('\tregion image is {} x {}'.format(img_shape[1], img_shape[0]))

        if groups is not None:
            # groups are named with the slice prefix
            try:
                igroup = [x.name.strip() for x in groups].index(prefix)
            except ValueError:
                print('No annotation info for ' + prefix)
                continue
            skels = list(groups[igroup].children); cnt = len(skels); nskels = 0
            cnodes = np.zeros((cnt,2,2), dtype=np.double)
            tear_conns = np.zeros((cnt,2,2), dtype=np.double); ntconn = 0
            for skel in skels:
                nodes = skel.get_node_positions()
                if nodes.shape[0] < 1: continue # skip empty trees
                if run_xnml_to_xregion:
                    assert( all([x[2] == solved_zind for x in nodes]) ) # node not on expected slice
                else:
                    assert( all([x[2] == zind for x in nodes]) ) # node not on expected slice
                if skel.name.strip().startswith('tear_connect'):
                    tear_conns[ntconn,:,:] = [x[:2] for x in nodes]; ntconn += 1
                else:
                    cnodes[nskels,:,:] = [x[:2] for x in nodes]; nskels += 1
            cnodes = cnodes[:nskels,:,:]
            tear_conns = tear_conns[:ntconn,:,:]
            print('\t{} correspondence points and {} tear connections'.format(nskels, ntconn))
        elif run_process:
            control_points = big_img_load(slice_image_fn, 'control_points')
            cnodes = control_points.reshape(-1,2,2) / use_ds
            nskels = cnodes.shape[0]
        else:
            control_points = cnodes = None; nskels = 0

        if cnodes is not None and doplots:
            plt.figure(1235); plt.gcf().clf(); plt.title('loaded')
            for i in range(cnodes.shape[0]):
                plt.scatter(cnodes[i,0,0], cnodes[i,0,1], c='r', s=36, marker='x')
                plt.scatter(cnodes[i,1,0], cnodes[i,1,1], c='r', s=36, marker='x')
                plt.plot(cnodes[i,:2,0], cnodes[i,:2,1], c='g', linewidth=2)
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', 'datalim')

        if run_process:
            print('Converting control point midpoints into tear skeleton'); t = time.time()

            # order the points along the path defined by the control point midpoints
            #   by using approximate TSP solver.
            mcnodes = cnodes.mean(axis=1)
            node_order = ANN(scidist.squareform(scidist.pdist(mcnodes)))[0]
            mcnodes = mcnodes[node_order,:]
            imcnodes = interpolate_path(mcnodes)
            bwskel, endpts_lbls, nendpts, endpts = path_to_skel(imcnodes)
            del mcnodes, imcnodes
            print('\tdone in %.4f s' % (time.time() - t, ))

            if doplots:
                plt.figure(1236); plt.gcf().clf(); plt.title('skel'); plt.imshow(bwskel)
                plt.scatter(endpts[:,0], endpts[:,1], c='r', s=36, marker='x')

            clf = [linear_model.LinearRegression(fit_intercept=True) for x in range(nendpts)]
            iclf = linear_model.LinearRegression(fit_intercept=True)
            boundary, tear_segments, nodes_middles, intersect, fit_x = \
                tear_skeleton_to_segments(bwskel, endpts, endpts_lbls)
            onodes = nodes_middles[:,:2,:]

            # now that the control points are ordered by which side of the tear they are on,
            #   interpolate the paths on either side of the tear separately and then use the
            #   midpoints of the interpolated control points to estimate the tear centerline,
            #   i.e., the boundary between the tear segments. this should give the most accurate
            #   estimate of the center of the tear, but without having to manually label the tear.

            print('Converting interpolated control points into tear skeleton'); t = time.time()
            onodes = onodes[node_order,:,:]
            ionodes = interpolate_path(onodes)
            imonodes = ionodes.mean(axis=1)
            bwskel, endpts_lbls, nendpts, endpts = path_to_skel(imonodes)
            del ionodes, imonodes
            print('\tdone in %.4f s' % (time.time() - t, ))

            if doplots:
                plt.figure(2236); plt.gcf().clf(); plt.title('skel'); plt.imshow(bwskel)
                plt.scatter(endpts[:,0], endpts[:,1], c='r', s=36, marker='x')

            clf = [linear_model.LinearRegression(fit_intercept=True) for x in range(nendpts)]
            iclf = linear_model.LinearRegression(fit_intercept=True)
            boundary, tear_segments, nodes_middles, intersect, fit_x = \
                tear_skeleton_to_segments(bwskel, endpts, endpts_lbls, figbase=1000)

            print('Add interpolated control points along and beyond tear'); t = time.time()
            add_nodes_middles = interpolate_path(onodes, sampling=interpolation_sampling)
            add_nodes_middles = np.concatenate((add_nodes_middles,
                    add_nodes_middles.mean(axis=1, keepdims=True)), axis=1)
            nodes_middles = np.concatenate((nodes_middles, add_nodes_middles), axis=0)

            ## automatically add mutual correspondence points above and below tear
            mean_interval = interpolation_sampling
            max_npts = np.round(np.sqrt((img_shape_ds*img_shape_ds).sum()) / mean_interval).astype(np.int64)
            add_nodes_middles = np.zeros((max_npts,3,2), dtype=np.double)
            cnpts = 0
            for i in range(nendpts):
                d = np.sqrt(((endpts[i,:] - intersect[i,:])**2).sum())
                npts = np.round(d / mean_interval).astype(np.int64) + 1
                if npts > 2:
                    xi = [1,0] if fit_x[i] else [0,1]
                    line_pts2 = dense_predict((intersect[i,:], endpts[i,:]), clf[i], fit_x[i], xi, npts=npts)[1:-1]
                    add_nodes_middles[cnpts:cnpts+npts-2,:,:] = line_pts2[:,None,:]
                    cnpts += npts-2
            add_nodes_middles = add_nodes_middles[:cnpts,:,:]
            nodes_middles = np.concatenate((nodes_middles, add_nodes_middles), axis=0)
            print('\tdone in %.4f s' % (time.time() - t, ))

            if doplots:
                tmp = tear_segments.copy(); tmp[boundary.astype(bool)] = 0 #; tmp[bw] = 3
                plt.figure(1240); plt.gcf().clf(); plt.title('tear segments points'); plt.imshow(tmp)
                plt.scatter(endpts[:,0], endpts[:,1], c='r', s=36, marker='o')
                plt.scatter(intersect[:,0], intersect[:,1], c='r', s=36, marker='o')
                for i in range(nodes_middles.shape[0]):
                    plt.scatter(nodes_middles[i,0,0], nodes_middles[i,0,1], c='r', s=36, marker='x')
                    if (nodes_middles[i,0,:] != nodes_middles[i,1,:]).any():
                        plt.scatter(nodes_middles[i,1,0], nodes_middles[i,1,1], c='r', s=36, marker='x')
                        plt.scatter(nodes_middles[i,2,0], nodes_middles[i,2,1], c='r', s=36, marker='x')
                        plt.plot(nodes_middles[i,:2,0], nodes_middles[i,:2,1], c='g', linewidth=2)
                plt.show()

            # write out the correspondence points and tear mask to a slice-specific dill file
            d = {'correspondence':nodes_middles, 'tear_segments_labels':tear_segments}
            with open(annotation_out_fn, 'wb') as f: dill.dump(d, f)

        elif run_nml_to_region:
            big_img_save(slice_image_fn, cnodes.reshape(-1,2)*use_ds, 'control_points')

        elif run_xnml_to_xregion:
            big_img_save(slice_image_fn, cnodes.reshape(-1,2)*use_ds, 'xcontrol_points')

        elif run_xregion_to_xnml:
            try:
                control_points = big_img_load(slice_image_fn, 'xcontrol_points')
            except:
                continue
            cnodes = control_points.reshape(-1,2,2) / use_ds

            group = annotation.skeleton.add_group(prefix)
            icnodes = np.round(cnodes).astype(np.int64)
            for i in range(cnodes.shape[0]):
                tree = group.add_tree("explorative_tear_convert_{}".format(total_nskels))
                node_1 = tree.add_node(position=(icnodes[i,0,0], icnodes[i,0,1], solved_zind))
                node_2 = tree.add_node(position=(icnodes[i,1,0], icnodes[i,1,1], solved_zind))
                tree.add_edge(node_1, node_2)
                total_nskels += 1

        elif run_warp:
            try:
                with open(annotation_out_fn, 'rb') as f: cdict = dill.load(f)
            except FileNotFoundError:
                print('No annotation info for ' + prefix)
                continue
            #d = {'correspondence':nodes_middles, 'tear_segments_labels':tear_segments}
            warp_image_in_fn = os.path.join(warp_image_in_dn, image_fn)
            print(warp_image_in_fn)
            img = tifffile.imread(warp_image_in_fn)
            img_shape = img.shape
            print(img_shape, cdict['tear_segments_labels'].shape, cdict['correspondence'].shape)
            # might have to crop the label image down
            tear_segments = cdict['tear_segments_labels'][:img_shape[0],:img_shape[1]]

            #grid_y, grid_x = np.indices((img_shape[0], img_shape[1]), dtype=np.double)
            grid_pts = np.mgrid[:img_shape[0],:img_shape[1]]
            grid_pts = grid_pts[::-1,:,:]
            #grid_x = grid_pts[0,:,:]; grid_y = grid_pts[1,:,:]
            grid_pts_flat = grid_pts.reshape(2,-1)
            coords = np.zeros_like(grid_pts_flat)

            nsegments = tear_segments.max()
            for i in range(nsegments):
                print('Interpolating segment {}'.format(i+1)); t = time.time()
                pts = cdict['correspondence'][:,i,:]
                dst_pts = cdict['correspondence'][:,2,:]

                sel_pts = np.logical_or(tear_segments == i+1, tear_segments == 0)
                sel_pts_flat = sel_pts.reshape(-1)
                sel_grid_pts_flat = grid_pts_flat[:,sel_pts_flat].T
                # this is for heavily downsampled images, so can get away with using MLS pixel dense method.
                coords[:,sel_pts_flat] = mls_rigid_transform(sel_grid_pts_flat, dst_pts, pts).T
                print('\tdone in %.4f s' % (time.time() - t, ))
            coords = coords.reshape([2] + list(img_shape)); coords = [coords[1,:,:], coords[0,:,:]]
            print('Applying remap'); t = time.time()
            img = nd.map_coordinates(img, coords, order=1, mode='constant', cval=0.0, prefilter=False)
            print('\tdone in %.4f s' % (time.time() - t, ))

            warp_image_out_fn = os.path.join(warp_image_out_dn, image_fn)
            print(warp_image_out_fn)
            tifffile.imwrite(warp_image_out_fn, img)

            # for comparison, also copy source file to a subfolder of output dir
            shutil.copy(warp_image_in_fn, warp_image_out_orig_dn)

            if doplots:
                dx = coords[1]-grid_pts[0,:,:]; dy = coords[0]-grid_pts[1,:,:]
                print(dx.min(), dx.max(), dy.min(), dy.max())
                plt.figure(1234); plt.gcf().clf(); plt.title('delta x'); plt.imshow(dx); plt.colorbar()
                plt.figure(1235); plt.gcf().clf(); plt.title('delta y'); plt.imshow(dy); plt.colorbar()
        # run_type if/elif

        if doplots: plt.show()
    #for region_ind in torn_regions[wafer_id]:
#for wafer_id, wafer_ind in zip(all_wafer_ids, range(total_nwafers)):

if nml_fn_out: annotation.save(nml_fn_out)

print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
