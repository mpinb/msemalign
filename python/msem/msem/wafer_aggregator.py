"""wafer_aggregator.py

Helper class for wafer class that aggregates solved rough and fine alignments.
Alignments are computed between neighboring slices, so they need to be
  aggregated to create single "global" transforms for each slice.

Copyright (C) 2018-2023 Max Planck Institute for Neurobiology of Behavior

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import time
import os
#import sys
#import warnings
import dill

import scipy.sparse as sp
#import scipy.spatial.distance as scidist
import scipy.spatial as spatial
#import scipy.interpolate as interp
import scipy.ndimage as nd

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing
from sklearn.neighbors import NearestNeighbors
# xxx - sometimes intel version completely crashes or hangs, not worth the modest speedup
#from sklearnex.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

from .mfov import mfov
from .zimages import zimages
from .procrustes import RigidRegression, RigidRegression_types
from .AffineRANSACRegressor import AffineRANSACRegressor, _ransac_repeat

from .utils import get_num_threads, tile_nblks_to_ranges
from .utils import mad_zscore, mad_angle_zscore
from .utils import get_voronoi_circumcenters_from_delaunay_tri, get_voronoi_to_simplex_mapping
from .utils import voronoi_vectors_to_point_vectors, mls_rigid_transform
from .utils import dill_lock_and_load, dill_lock_and_dump, get_process_uuid
from .utils import delta_interp_methods
from .utils import make_delta_plot #, make_grid_plot
from .utils import big_img_load, big_img_save, big_img_info


import multiprocessing as mp
import queue


# calls into "kevin's formula" least squares solver independently per each grid voronoi vertex.
#   in the documentation this is now referred to as the "emalign solver".
# shared between rough and fine reconciler.
# this is the main function for a single worker thread, so each voronoi vertex can be parallized independetly.
def reconcile_deltas_job(n, ind, inds, simplices, neighbor_rng, nneighbors, pts_to_grid_pts, deltas, weights,
        valid_comparison, L1_norm, L2_norm, min_adj, regr_bias, voronoi_to_simplex, scale, voronoi_neighbors,
        neighbors_std_pixels, neighbors_W, neighbors_expected, vertices, z_nnbrs_rad, result_queue, verbose):
    if verbose: print('\tworker%d started' % (ind,))
    neighbors_var_pixels = neighbors_std_pixels*neighbors_std_pixels

    # option to only use local neighborhood for solving deltas
    # first array value is the radius size of the overlapping neighborhood
    z_nnbrs = 2*z_nnbrs_rad[0] + 1
    assert(z_nnbrs < n) # z neighborhood can not be bigger than the number of slices
    if z_nnbrs > 1:
        z_nearest = NearestNeighbors(n_neighbors=z_nnbrs, algorithm='kd_tree').fit(np.arange(n).reshape(-1,1))
        nlocal = z_nnbrs; nlocal_nbrs = z_nnbrs + nneighbors
        min_nnbrs = -min(neighbor_rng)

        if z_nnbrs_rad[1] > 1:
            assert(z_nnbrs_rad[0] > z_nnbrs_rad[1] + 1) # overlap radius must be bigger than chunk radius
            # second array value is the radius size of the chunk to store in each iteration (ziters).
            # the overlap that is not part of the chunk but part of the neighborhood is discarded.
            nchunks = max([np.round(n/(2*z_nnbrs_rad[1])).astype(np.int64), 1])
        else:
            # this method of only storing one delta at a time but solved with context is relatively very slow
            nchunks = n
        ichunks = np.array_split(np.arange(n), nchunks)
        ziters = len(ichunks)
    else:
        nlocal = nlocal_nbrs = n; ziters = 1
        zrng = [0, nlocal]; min_nnbrs = 0

    # solve for deltas at each voronoi vertex, treat each independently.
    for g in range(inds.size):
        #vrdelta = vcomps_sel = 0 # default to zero, broadcasts out when results assembled in main thread
        # originally just set these as zeros and let it broadcast.
        # with the option for a z neighborhood range (z_nnbrs > 0), this was no longer possible.
        vrdelta_default = np.zeros((nlocal_nbrs,2), dtype=np.double)
        vcomps_sel_default = np.zeros((nlocal_nbrs,), dtype=bool)
        vrdelta = vrdelta_default; vcomps_sel = vcomps_sel_default

        # see http://qhull.org/html/qvoronoi.htm option Qz, point "at infinity"
        # ignore voronoi vertices with no simplex mapping. they do not define any voronoi regions.
        if voronoi_to_simplex[inds[g]].size > 0:
            # option to only use local neighborhood for solving deltas
            if z_nnbrs > 1:
                vrdeltas = np.zeros((n,2), dtype=np.double)
                vcomps_sels = np.zeros((n,), dtype=bool)

            # if the 2d smoothing feature is not enabled then nbrs contains only the vertex being solved.
            nbrs = voronoi_neighbors[inds[g]]
            # put the vertex being solved as the first neighbor
            nbrs = np.concatenate(([inds[g]], np.setdiff1d(nbrs, [inds[g]])))
            nnbrs = nbrs.size
            # m is now the total solver array sizes. the first n-entries are the deltas for the vertex
            #   that is currently being solved. neighborhood must be kept relatively small for this to work.
            m = nlocal_nbrs*nnbrs

            # outermost iterates chunks if using local z neighborhood
            for iz in range(ziters):
                # get total weightage for each z-slice and each neighbor, so that z-slice comparisons weighting
                #   can be balanced against 2D smoothness comparisons weighting.
                cum_z_weights = np.zeros((nlocal,nnbrs), dtype=np.double)

                # the matrices being populated here that will be sent to the solver.
                adj = sp.dok_matrix((m,m), dtype=bool)
                dx = sp.dok_matrix((m,m), dtype=np.double)
                dy = sp.dok_matrix((m,m), dtype=np.double)
                W = sp.dok_matrix((m,m), dtype=np.double)

                # option to only use local neighborhood for solving deltas
                if z_nnbrs > 1:
                    vrdelta = vrdelta_default; vcomps_sel = vcomps_sel_default
                    zinds = np.sort(z_nearest.kneighbors(np.array([ichunks[iz].mean(dtype=np.int64)]).reshape((1,1)),
                            return_distance=False).reshape(-1))
                    # range to be solved in original slice indices
                    zrng = [zinds.min(), zinds.max()+1]
                    # range to be stored in original slice indices (not including overlap)
                    crng = [ichunks[iz].min(), ichunks[iz].max()+1]
                    # range to be stored in adjacency matrix indices (not including overlap)
                    lcrng = [np.nonzero(zinds == crng[0])[0][0] + min_nnbrs,
                             np.nonzero(zinds == crng[1] - 1)[0][0] + min_nnbrs + 1]

                # outer loop over slices to be solved
                for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):
                    # inner loop over number of neighboring slices to use
                    for k,ik in zip(neighbor_rng, range(nneighbors)):
                        j = i+k; jl = il+k
                        ilo = il + min_nnbrs; jlo = jl + min_nnbrs
                        # do not add adjacency for missing or invalid comparisons
                        if not valid_comparison[i,ik]: continue
                        assert(j >= 0 and j < n) # you screwed up, comparison index outside image range

                        # loop over 2d neighbors. first index of nbrs is current vertex being solved.
                        for inbr in range(nnbrs):
                            # these are the simplices that correpsond to this voronoi vertex.
                            # in the case of nvertices==nsimplices, this is a single simplex.
                            # xxx - this code should theoretically still work in the case of multiple simplices.
                            #   this has not been tested as msem package always uses equally spaced point grids (hex).
                            csimplices = simplices[voronoi_to_simplex[nbrs[inbr]],:]

                            # additional mapping required into weights and deltas if blocking is enabled
                            if pts_to_grid_pts is not None:
                                csimplices = pts_to_grid_pts[csimplices]
                                assert( (csimplices >= 0).all() ) # block point mapping fail

                            # if weights are per slice, simplices do not matter.
                            # if weights are per point, take the min weight over the simplex vertices.
                            curW = weights[i][ik] if weights[i].ndim==1 else weights[i][ik,csimplices].min()
                            # do not add adjacency for zero-weighted comparisons
                            if curW==0: continue
                            cum_z_weights[il,inbr] += curW

                            # take the mean delta over the simplex vertices.
                            # the reason this is not weighted by distance is because we are using
                            #   the voronoi points which are the circumcenters of the simplices.
                            cdelta = scale*deltas[i][ik,csimplices,:].reshape((-1,2)).mean(0)

                            # update the matrices that are sent to the solver.
                            ii = inbr*nlocal_nbrs+ilo; jj = inbr*nlocal_nbrs+jlo
                            W[jj,ii] = curW; adj[jj,ii] = 1; dx[jj,ii] = cdelta[0]; dy[jj,ii] = cdelta[1]
                        #for inbr in range(nnbrs):
                    #for k,ik in zip(neighbor_rng, range(nneighbors)):
                #for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):

                # now add pairwise connections between the neighboring 2D vertices.
                # term neighbor is confusing here. we have neighboring z-slices (3D) and neighboring vertices (2D).
                if nnbrs > 1 and all([x > 0 for x in neighbors_expected]):
                    # scale the baseline weighting by the total number of pairwise comparisons.
                    nbrsW = neighbors_W / nnbrs / (nnbrs-1)
                    # optionally only select a subset of the pairwise connections
                    #   that do not involved the current vertex (neighbor at index zero).
                    # specify different expected number of neighbors for the center vertex
                    #   that is being solved and the rest.
                    expected_nnbrs = neighbors_expected
                    expected_nnbrs = [0 if x < 0 else x for x in expected_nnbrs]
                    expected_nnbrs = [nnbrs-1 if x > nnbrs-1 else x for x in expected_nnbrs]

                    # double for loop over all pairwise combinations of vertices.
                    for inbr in range(nnbrs):
                        jsel = (np.random.rand(nnbrs-1) < expected_nnbrs[inbr > 0]/(nnbrs-1))
                        for jnbr in range(nnbrs):
                            if inbr==jnbr: continue # no self-comparisons
                            # optional sparsity in pairwise comparisons
                            if not jsel[jnbr-1]: continue

                            # delta between voronoi vertices in the same z-plane
                            vdelta = vertices[nbrs[inbr],:] - vertices[nbrs[jnbr],:]

                            # scale the baseline weight of the comparisons based on the distance between the points.
                            if np.isfinite(neighbors_var_pixels) and neighbors_var_pixels > 0:
                                curW = nbrsW*np.exp(-(vdelta*vdelta).sum(1)/neighbors_var_pixels/2)
                            else:
                                curW = nbrsW

                            # force similarity between the 2D vertices by setting delta to zero
                            cdelta = [0,0]
                            ## tie vertices together at the actual voronoi spacing
                            #cdelta = scale * vdelta

                            # add connections between neighboring vertices in each z-slice.
                            for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):
                                ilo = il + min_nnbrs
                                # adjust for the total z-weighting at each z-index, accumulated above.
                                ccurW = curW * cum_z_weights[il,inbr]
                                # do not add adjacency for zero-weighted comparisons
                                if ccurW == 0: continue

                                # update the matrices that are sent to the solver.
                                ii = inbr*nlocal_nbrs+ilo; jj = jnbr*nlocal_nbrs+ilo
                                W[ii,jj] = curW; adj[ii,jj] = 1; dx[ii,jj] = cdelta[0]; dy[ii,jj] = cdelta[1]
                            # for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):
                        #for jnbr in range(nnbrs):
                    #for inbr in range(nnbrs):
                #if nnbrs > 1:

                #print(adj.shape, adj.nnz, nnbrs)
                if adj.nnz >= min_adj:
                    try:
                        vrdelta, _, vcomps_sel = mfov.solve_stitching(adj, dx, Dy=dy, W=W, label_adj_min=min_adj,
                                return_comps_sel=True, l1_alpha=L1_norm, l2_alpha=L2_norm, regr_bias=regr_bias)
                        # slice out just the vertex being solved, discard neighborhood vertices
                        vrdelta = vrdelta[:nlocal_nbrs,:]; vcomps_sel = vcomps_sel[:nlocal_nbrs]
                    except np.linalg.LinAlgError:
                        print('WARNING: solver did not converge at voronoi vertex ind {}'.format(inds[g]))

                # option to only use local neighborhood for solving deltas
                if z_nnbrs > 1:
                    vrdeltas[crng[0]:crng[1],:] = vrdelta[lcrng[0]:lcrng[1],:]
                    vcomps_sels[crng[0]:crng[1]] = vcomps_sel[lcrng[0]:lcrng[1]]
            #for iz in range(ziters):

            # option to only use local neighborhood for solving deltas
            if z_nnbrs > 1:
                vrdelta = vrdeltas; vcomps_sel = vcomps_sels
        #if voronoi_to_simplex[inds[g]].size > 0:

        result = {'ind':inds[g], 'vrdelta':vrdelta, 'vcomps_sel':vcomps_sel, 'iworker':ind}
        result_queue.put(result)
    #for g in range(inds.size):

    if verbose: print('\tworker%d completed' % (ind, ))
#def reconcile_deltas_job

def filter_solved_deltas_job(ind, inds, inliers, deltas, grid_pts, shape_pixels, affine_degree, nthreads,
        result_queue, verbose):
    if verbose: print('\tworker%d started' % (ind,))

    # iterate the fine deltas and fit local regions with affines.
    # use the local affine fitted deltas as the new delta at each point, an "affine filter"
    n = inds.size
    ngrid = grid_pts.shape[0]
    poly = preprocessing.PolynomialFeatures(degree=affine_degree)
    poly.fit_transform(np.random.rand(3,2)) # just so features are populated for 2D
    clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=nthreads)

    # outer loop over slices, fits are done independently over slices and skips
    for i in range(n):
        #if verbose: print('worker{} processing total order ind {}'.format(ind,inds[i],))

        filtered_deltas = np.zeros((ngrid,2), dtype=np.double)
        for g in range(ngrid):
            if not inliers[i,g]: continue # do not do filter outlier points
            sel_pts = np.logical_and(grid_pts >= grid_pts[g,:]-shape_pixels/2,
                                     grid_pts <= grid_pts[g,:]+shape_pixels/2).all(1)
            sel_pts = np.logical_and(sel_pts, inliers[i,:])

            # if there are not enough points to fit the affine
            if sel_pts.sum() < poly.n_output_features_ + 1:
                print('WARNING: at grid point {}, only {} inliers'.format(g,sel_pts.sum()))
                #assert(False) # do not do this, can still happen for very small number of inliers
                # leaving the deltas at zero and continue.
                # outlier detection should have already flagged this as an invalid comparison.
                continue

            # fit the affine and use it to estimate the current point ("affine-filter")
            cgrid_pts = grid_pts[sel_pts,:]
            cgrid_pts_dst = cgrid_pts + deltas[i,sel_pts,:]
            cgrid_pts_src = poly.fit_transform(cgrid_pts)
            clf.fit(cgrid_pts_src, cgrid_pts_dst)
            fit_pt = clf.predict(poly.fit_transform(grid_pts[g,:][None,:]))
            filtered_deltas[g,:] = fit_pt - grid_pts[g,:]
        #for g in range(self.ngrid):

        result = {'ind':inds[i], 'filtered_deltas':filtered_deltas, 'iworker':ind}
        result_queue.put(result)
    #for i in range(self.img_range[0],self.img_range[1]):

    if verbose: print('\tworker%d completed' % (ind, ))
#def filter_solved_deltas_job


#     ### Params to mls_rigid_transform:
#         * v - ndarray: an array with size [m, 2], points to transform
#         * p - ndarray: an array with size [n, 2], control points
#         * q - ndarray: an array with size [n, 2], deformed control points
def mls_rigid_transform_nbhd_job(ind, v, p, q, nneighbors, result_queue, verbose):
    if verbose: print('\tworker%d started' % (ind,))

    n = v.shape[0]
    f_v = np.zeros_like(v)
    nbrs = NearestNeighbors(n_neighbors=nneighbors, algorithm='kd_tree').fit(p)
    for i in range(n):
        iv = v[i,:][None,:]
        pinds = nbrs.kneighbors(iv, return_distance=False).reshape(-1)
        f_v[i] = mls_rigid_transform(iv, p[pinds,:], q[pinds,:])
    #for i in range(n):

    # return all the results at once. single result puts/gets through the result queue are slow.
    result = {'f_v':f_v, 'iworker':ind}
    result_queue.put(result)

    if verbose: print('\tworker%d completed' % (ind, ))
#def mls_rigid_transform_nbhd_job

def get_block_info_job(ind, iblocks, nblocks, block_overlap_pix, grid_locations_pixels, result_queue, verbose):
    if verbose: print('\tworker%d started' % (ind,))

    n = iblocks.shape[0]
    blk_ngrid = np.zeros(n, dtype=np.int64)
    for i in range(n):
        igrid_min = np.floor(grid_locations_pixels.min(0)).astype(np.int64)
        igrid_max = np.ceil(grid_locations_pixels.max(0)).astype(np.int64)
        _, _, _, rng = tile_nblks_to_ranges(igrid_max - igrid_min, nblocks, block_overlap_pix, iblocks[i,:],
            ignore_bounds=True)
        blk_min = np.array([rng[0][0], rng[1][0]]) + igrid_min
        blk_max = np.array([rng[0][1], rng[1][1]]) + igrid_min
        blk_ngrid[i] = np.logical_and(grid_locations_pixels >= blk_min, grid_locations_pixels <= blk_max).all(1).sum()

    # return all the results at once. single result puts/gets through the result queue are slow.
    result = {'blk_ngrid':blk_ngrid, 'iworker':ind}
    result_queue.put(result)

    if verbose: print('\tworker%d completed' % (ind, ))
#def get_block_info_job


class wafer_aggregator(zimages):
    """msem wafer helper class.

    Accumulates transformations between neighboring slices to give global transforms for rough and fine alignment.

    .. note::


    """

    ### fixed parameters not exposed

    # so that this can be consistent across whole msem package using env variable.
    nthreads = get_num_threads()

    def __init__(self, wafer_ids, region_strs, solved_orders, wafers_nregions, meta_dill_fn='', order_range=None,
            rough_bounding_box_pixels=None, min_percent_inliers_C_cutoff=0.5, outlier_affine_degree=1,
            residual_threshold_pixels=50, inlier_min_neighbors=2, inlier_min_component_size_edge_pixels=2.5,
            C_hard_cutoff=None, ninlier_neighhbors_cmp=[7,7], ok_outlier_zscore=2., not_ok_inlier_zscore=2.,
            fine_nearby_points_pixels=None, region_interp_type_deltas='cubic', ransac_repeats=1, ransac_max=1000,
            rigid_type=RigidRegression_types.affine, verbose_iterations=False, verbose=False):
        self.wafer_ids = wafer_ids
        self.nwafer_ids = len(wafer_ids)
        self.mwafer_ids = max(wafer_ids)
        self.wafer_aggregator_verbose = verbose

        # array of the solved order for each region being accumulated
        self.solved_orders = solved_orders

        # list of the region strings that is indexed by the solved orders
        self.region_strs = region_strs

        # NOTE: this must match those used by wafer.py and wafer_solver.py
        # this has to match the code in wafer.py so that the rough bounding box area is always the same pixel size.
        self.rough_bounding_box_size = np.round(rough_bounding_box_pixels[1]-rough_bounding_box_pixels[0])

        # prints messages after each accumulation iteration (slice)
        self.verbose_iterations = verbose_iterations

        # meta dill is used for coordinating processes
        self.meta_dill_fn = meta_dill_fn

        # inits related to slice ordering and total number of slices over all wafers
        self.wafers_nimgs = np.zeros((self.nwafer_ids,), dtype=np.int32)
        self.wafers_nregions = wafers_nregions
        # initialize per wafer
        cum_nsolved_order = 0
        self.region_ind_to_solved_order = [None]*self.nwafer_ids
        self.missing_region_inds = [None]*self.nwafer_ids
        for i in range(self.nwafer_ids):
            self.wafers_nimgs[i] = solved_orders[i].size
            # get the inverse lookup for the solved order, put -1's for missing regions
            imap = -np.ones(wafers_nregions[i], np.int64)
            imap[solved_orders[i]] = np.arange(solved_orders[i].size)
            sel = (imap >= 0); imap[sel] += cum_nsolved_order
            cum_nsolved_order += solved_orders[i].size
            self.region_ind_to_solved_order[i] = imap
            self.missing_region_inds[i] = np.nonzero(np.logical_not(sel))[0]
        self.total_nimgs = self.wafers_nimgs.sum()
        self.cum_wafers_nimgs = np.concatenate(([0], np.cumsum(self.wafers_nimgs)), axis=0)

        # this is for only accumulating a subset of the specified solved order slices.
        # default of None or range less than zero specifies to accumulate all order solved slices in specified wafers.
        if order_range is not None and all([x >=0 for x in order_range]):
            assert(all([x >= 0 for x in order_range]))
            assert(all([x <  self.total_nimgs-1 for x in order_range]))
            self.order_rng = np.array(order_range)
        else:
            self.order_rng = np.array([0, self.total_nimgs])

        # <<< parameters for the fine alignment outlier detection

        # polynomial degree for the affine fit for detecting outliers, over 3 definitely not recommended
        self.outlier_affine_degree = outlier_affine_degree

        # residual threshold for the ransac affine fit
        self.residual_threshold_pixels = residual_threshold_pixels

        # minimum number of non-outlying or excluded neighbors to keep as inlier
        self.inlier_min_neighbors = inlier_min_neighbors

        # minimum size of connected components square edge length.
        # used to compute smallest connected compent size (in number of grid points).
        self.inlier_min_component_size_edge_pixels = inlier_min_component_size_edge_pixels

        # hard cutoffs for xcorr values.
        # indexed by n-1 where n is the distance away from current slice being compared.
        self.C_hard_cutoff = C_hard_cutoff

        # iterate hard cutoffs in decreasing order until less than this percentage of included points
        #   are flagged as outliers by the C_cutoff.
        self.min_percent_inliers_C_cutoff = float(min_percent_inliers_C_cutoff)

        # number of inlier neighbors to calculate z-scores over to decide it "outlier deltas" should be
        #   included with a lesser weight or not.
        self.ninlier_neighhbors_cmp = [abs(int(x)) for x in ninlier_neighhbors_cmp]
        self.inlier_neighbors_cmp_deltas = any([x < 0 for x in ninlier_neighhbors_cmp])

        # zscore to include "outlier deltas" with a lesser weight
        self.ok_outlier_zscore = float(ok_outlier_zscore)

        # zscore to reject some inliers but include in solver with a lesser weight
        self.not_ok_inlier_zscore = float(not_ok_inlier_zscore)

        # create another optional outlier select that only includes points with specified distance
        #   of the current inlier points.
        self.fine_nearby_points_pixels = fine_nearby_points_pixels

        # method of interpolating the deltas to fill in outliers.
        self.region_interp_type_deltas = region_interp_type_deltas

        # parameters for the fine alignment outlier detection >>>

        # other inits
        self.forward_affines = None
        self.wafers_imaged_order_rough_affines = None

        # number of times to repeat ransac fit, parallized by workers
        self.ransac_repeats = ransac_repeats

        # number of individual ransac max iterations
        self.ransac_max = ransac_max

        # this is the type of the rigid transform to fit for rough alignments
        self.rigid_type = rigid_type

    def _triangulate_grid(self, neighbors_radius_pixels):
        if hasattr(self, 'grid_delaunay'): return

        if self.wafer_aggregator_verbose:
            print('Computing simplex and circumcenter information for grid'); t = time.time()

        pts = self.grid_locations_pixels # just for convenience
        dln = spatial.Delaunay(pts)
        vor = spatial.Voronoi(pts)

        # nsimplices and nvertices usually are the same, except on some regular-spaced grids,
        #   a single voronoi point can belong to multiple simplices.
        # in a cartesian grid, for example, most of the voronoi vertices correspond to two circumcenters,
        #   one for each of the two triangles that bisect each square (at the center of each square).
        # this is because the 4-corners of a square are always on the same circle (circumscribed circle).
        #nsimplices = dln.simplices.shape[0]
        self.grid_nvertices = vor.vertices.shape[0]

        # create a mapping from the simplices to voronoi vertices
        vertices = get_voronoi_circumcenters_from_delaunay_tri(pts, dln.simplices)
        voronoi_to_simplex = get_voronoi_to_simplex_mapping(vertices, vor.vertices)

        # use specified neighborhood radius for optional 2d smoothing constraint in solver.
        nbrs = NearestNeighbors(radius=neighbors_radius_pixels, algorithm='ball_tree').fit(vor.vertices)
        voronoi_neighbors = nbrs.radius_neighbors(vor.vertices, return_distance=False)

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        if not self.single_block:
            # have to iterate the voronoi points here in order to create voronoi select for the block.
            if self.wafer_aggregator_verbose:
                print('Iterating voronoi vertices to get those corresponding to block'); t = time.time()
            # see fuller comments in corresponding loop in reconcile_deltas_job.
            self.sel_vertices = np.zeros((self.grid_nvertices,), dtype=bool)
            for g in range(self.grid_nvertices):
                if voronoi_to_simplex[g].size > 0:
                    csimplices = dln.simplices[voronoi_to_simplex[g],:]
                    # process all vertices that correspond to any point in the block
                    self.sel_vertices[g] = self.blk_grid_pts_novlp_sel[csimplices].any()
            if self.wafer_aggregator_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))
        else:
            self.sel_vertices = None

        return dln, vor, voronoi_to_simplex, voronoi_neighbors

    # calls into "kevin's formula" least squares solver independently per each grid voronoi vertex.
    # shared between rough and fine reconciler. both thread and process parallelized across grid points.
    # NOTE: this can NOT be parallelized by images, as all the deltas in the z-direction are used
    #   to reconcile the final deltas at each grid point.
    def _reconcile_grid_deltas(self, deltas, weights, valid_comparison, L1_norm=0., L2_norm=0., min_adj=0,
            regr_bias=False, idw_p=2., scale=1., neighbors_radius_pixels=0., neighbors_std_pixels=np.inf,
            neighbors_W=0., neighbors_expected=0, z_nnbrs_rad=[0,0], nworkers=1, iprocess=0, nprocesses=1):
        # make sure grid is within rough bounding box
        #assert( (grid_points.min(0) > [0,0]).all() ) # bad grid points
        #assert( (grid_points.max(0) < self.rough_bounding_box_size).all() ) # bad grid points

        # inits
        dln, vor, voronoi_to_simplex, voronoi_neighbors = self._triangulate_grid(neighbors_radius_pixels)
        n = self.total_nimgs
        pts = self.grid_locations_pixels # just for convenience
        workers = [None]*nworkers
        result_queue = mp.Queue(self.grid_nvertices)

        # this is to coordinate the same set of permuted indices across all processes
        # xxx - probably best to functionize this and move to utils
        assert(iprocess < nprocesses) # don't be a fool
        #inds = np.arange(self.grid_nvertices) # poor balancing for large number of processes / workers
        if self.single_block:
            uuid_str = get_process_uuid()
            if iprocess == 0:
                inds = np.random.permutation(self.grid_nvertices) # to try for better balancing
                if nprocesses > 0:
                    d, f1, f2 = dill_lock_and_load(self.meta_dill_fn, keep_locks=True)
                    d['process_uuids'][uuid_str] = {'inds':inds, 'cnt':1}
                    # instead of this can also re-export meta dill properly with run_wafer
                    #d['process_uuids']= {} # hack to remove, sometimes needed after debugging
                    dill_lock_and_dump(self.meta_dill_fn, d, f1, f2)
            else:
                if self.wafer_aggregator_verbose:
                    print('Waiting for uuid from process 0'); t = time.time()
                loaded_uuid = False
                while not loaded_uuid:
                    d = dill_lock_and_load(self.meta_dill_fn)
                    if uuid_str in d['process_uuids']:
                        loaded_uuid = True
                        inds = d['process_uuids'][uuid_str]['inds']
                    else:
                        time.sleep(2)
                if self.wafer_aggregator_verbose:
                    print('\tdone in %.4f s' % (time.time() - t, ))

                # increment a count, then delete the count key after all processes are done
                d, f1, f2 = dill_lock_and_load(self.meta_dill_fn, keep_locks=True)
                d['process_uuids'][uuid_str]['cnt'] += 1
                if d['process_uuids'][uuid_str]['cnt'] == nprocesses:
                    print('Deleting uuid from meta for process sync'); t = time.time()
                    del d['process_uuids'][uuid_str]
                dill_lock_and_dump(self.meta_dill_fn, d, f1, f2)

            # no additional point mapping without blocking
            pts_to_grid_pts = None
        else: # if self.single_block:
            assert(nprocesses==1) # xxx - did not see the use case for both multiple blocks and processes
            inds = np.random.permutation(np.nonzero(self.sel_vertices)[0]) # to try for better balancing
            # create a mapping from grid point indices to block grid point indices
            pts_to_grid_pts = -np.ones((self.ngrid,), dtype=np.int64)
            pts_to_grid_pts[self.blk_grid_pts_sel] = np.arange(self.blk_ngrid)

        # split the inds by total number of workers (nworkers * nprocesses)
        inds = np.array_split(inds, nworkers*nprocesses)
        inds = inds[iprocess*nworkers:(iprocess+1)*nworkers]
        #inds_proc = np.concatenate(inds); ninds_proc = inds_proc.size
        ninds_proc = sum([x.size for x in inds])

        if self.wafer_aggregator_verbose:
            print('Reconciling deltas at {} of {} voronoi vertices'.format(ninds_proc, self.grid_nvertices,))
            print('\tiprocess {} using {} independent workers'.format(iprocess,nworkers,))
            if not self.single_block:
                print('\tiblock {} {} of {} {}'.format(self.iblock[0], self.iblock[1],
                    self.nblocks[0], self.nblocks[1]))
            t = time.time()

        for i in range(nworkers):
            workers[i] = mp.Process(target=reconcile_deltas_job, daemon=True,
                    args=(n, i, inds[i], dln.simplices, self.neighbor_rng, self.nneighbors,
                        pts_to_grid_pts, deltas, weights, valid_comparison, L1_norm, L2_norm, min_adj, regr_bias,
                        voronoi_to_simplex, scale, voronoi_neighbors, neighbors_std_pixels, neighbors_W,
                        neighbors_expected, vor.vertices, z_nnbrs_rad, result_queue, self.wafer_aggregator_verbose))
            workers[i].start()
        # NOTE: only call join after queue is emptied
        # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

        # for multiple processes save memory by only allocating vertices for this process.
        sproc = (nprocesses==1 and self.single_block)
        # vrdeltas = np.zeros((n,self.grid_nvertices,2), dtype=np.double)
        # vcomps_sel = np.zeros((n,self.grid_nvertices), dtype=bool)
        vrdeltas = np.zeros((n,ninds_proc,2), dtype=np.double)
        vcomps_sel = np.zeros((n,ninds_proc), dtype=bool)
        inds_proc = None if sproc else np.zeros((ninds_proc,), dtype=np.int64)
        nprint = 200
        dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for g in range(ninds_proc):
        g = 0
        while g < ninds_proc:
            if self.wafer_aggregator_verbose and g>0 and g%nprint==0:
                print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
                print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            if sproc:
                vrdeltas[:,res['ind'],:] = res['vrdelta']
                vcomps_sel[:,res['ind']] = res['vcomps_sel']
            else:
                vrdeltas[:,g,:] = res['vrdelta']
                vcomps_sel[:,g] = res['vcomps_sel']
                inds_proc[g] = res['ind']
            worker_cnts[res['iworker']] += 1
            g += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        if sproc:
            rdeltas = self.voronoi_vectors_to_point_vectors(vrdeltas=vrdeltas, pts=pts, vor=vor, idw_p=idw_p)
            comps_sel = (self.voronoi_vectors_to_point_vectors(vrdeltas=vcomps_sel[:,:,None], pts=pts, vor=vor,
                    idw_p=idw_p) > 0.).any(2)
        else:
            rdeltas = vrdeltas
            comps_sel = vcomps_sel

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        return rdeltas, comps_sel, inds_proc
    #def _reconcile_grid_deltas(self,

    def voronoi_vectors_to_point_vectors(self, vrdeltas=None, pts=None, vor=None, idw_p=2.):
        if vrdeltas is None: vrdeltas = self.cum_deltas
        if pts is None: pts = self.grid_locations_pixels
        if vor is None: vor = spatial.Voronoi(pts)

        return voronoi_vectors_to_point_vectors(vrdeltas, pts, vor, idw_p=idw_p, bcast=True)

    # <<< new rough alignment "reconciler"

    # initialize member variables for rough affines and rough affined fitted points from solver.
    def init_rough(self, rough_alignment_grid_pixels,
                   forward_affines, forward_pts_src, forward_pts_dst,
                   reverse_affines, reverse_pts_src, reverse_pts_dst,
                   affine_percent_matches=None):
        self.neighbor_range = self.max_neighbor_range = len(forward_pts_src[0])
        self.neighbor_rng = [x for x in range(-self.neighbor_range,self.neighbor_range+1) if x != 0]
        self.nneighbors = len(self.neighbor_rng)
        self.single_block = True # xxx - do we need block processing for rough alignment?

        # rough alignment grid locations.
        # need some minimum number of points on a grid that covers the rough_bounding_box.
        # xxx - can the points here be chosen automatically?
        self.ngrid = rough_alignment_grid_pixels.shape[0]
        self.grid_locations_pixels = rough_alignment_grid_pixels + self.rough_bounding_box_size[None,:]/2
        self.ngrid = self.grid_locations_pixels.shape[0]

        self._compute_skip_lens()

        self.forward_affines = forward_affines; self.reverse_affines = reverse_affines
        self.forward_pts_src = forward_pts_src; self.forward_pts_dst = forward_pts_dst
        self.reverse_pts_src = reverse_pts_src; self.reverse_pts_dst = reverse_pts_dst

        self.affine_percent_matches = affine_percent_matches


    # this is an artifact of how the skips are done in wafer_solver.
    def _compute_skip_lens(self):
        # pre-compute the starting offsets for each skip "chunk". they are arranged in chunks by the modulo.
        # maybe this can also work by splitting arrays or using modulo math.
        # it was written this way to match the way it is written in wafer_solver.
        self.skip_lens = [[None]*self.max_neighbor_range for x in range(self.nwafer_ids)]
        self.skip_cumlens = [[None]*self.max_neighbor_range for x in range(self.nwafer_ids)]
        for i in range(self.nwafer_ids):
            for j in range(self.max_neighbor_range):
                self.skip_lens[i][j] = np.zeros((j+1,), dtype=np.int64)
                for k in range(j+1):
                    self.skip_lens[i][j][k] = self.solved_orders[i][k::(j+1)].size
                self.skip_cumlens[i][j] = np.concatenate(([0], np.cumsum(self.skip_lens[i][j])))


    def reconcile_rough_alignments(self, rough_distance_cutoff_pixels, L1_norm=0., L2_norm=0., regr_bias=False,
            min_valid_slice_comparisons=0, idw_p=2., neighbor_dist_scale=None, neighbors2D_radius_pixels=0.,
            neighbors2D_std_pixels=np.inf, neighbors2D_W=0., neighbors2D_expected=0, z_neighbors_radius=0,
            nworkers=1, iprocess=0, nprocesses=1):
        if self.wafer_aggregator_verbose:
            print('Reconcile rough, ngrid points {}, L1 norm {}, L2 norm {}, iprocess {}, nprocesses {}'.\
                format(self.ngrid, L1_norm, L2_norm, iprocess, nprocesses))
        sproc = (nprocesses==1 and self.single_block)

        # inits
        poly = preprocessing.PolynomialFeatures(degree=1)
        grid_pts_src = poly.fit_transform(self.grid_locations_pixels)
        all_pts_src = [[None]*self.nneighbors for x in range(self.total_nimgs)]
        #all_pts_dst = [[None]*self.nneighbors for x in range(self.total_nimgs)]
        all_deltas = np.zeros((self.total_nimgs,self.nneighbors,self.ngrid,2), dtype=np.double)
        all_deltas_weight = np.zeros((self.total_nimgs,self.nneighbors), dtype=np.double)
        valid_comparison = np.zeros((self.total_nimgs,self.nneighbors), dtype=bool)

        # outer loop over slices
        for i in range(self.total_nimgs):
            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]

            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                if k < 0:
                    wafer_ind = j_wafer_ind; ind = j_ind
                    pts_src = self.reverse_pts_src[wafer_ind][ak-1]
                    pts_dst = self.reverse_pts_dst[wafer_ind][ak-1]
                    aff = self.reverse_affines[wafer_ind][ak-1]
                else:
                    wafer_ind = i_wafer_ind; ind = i_ind
                    pts_src = self.forward_pts_src[wafer_ind][ak-1]
                    pts_dst = self.forward_pts_dst[wafer_ind][ak-1]
                    aff = self.forward_affines[wafer_ind][ak-1]

                # the alignments are calculated in the solver by dividing into groups modulo by skip amount.
                div_ind = ind // ak; mod_ind = ind % ak
                skip_ind = self.skip_cumlens[wafer_ind][ak-1][mod_ind] + div_ind
                pts_src = pts_src[skip_ind]; pts_dst = pts_dst[skip_ind]; aff = aff[skip_ind]

                if pts_src is not None:
                    all_pts_src[i][ik] = pts_src #; all_pts_dst[i][ik] = pts_dst

                    # scikit learn puts constant terms on the left, remove augment and flip
                    aff = np.concatenate( (aff[:2,2][:,None], aff[:2,:2], ), axis=1 )
                    all_deltas[i,ik,:,:] = np.dot(grid_pts_src, aff.T) - self.grid_locations_pixels
                    all_deltas_weight[i,ik] = 1
                    valid_comparison[i,ik] = 1
                    #make_delta_plot(grid, deltas=all_deltas[i][ik]); plt.show()

        # scale weights so slices with less valid comparisons contribute less.
        valid_compare_cnt = valid_comparison.sum(1)
        assert((valid_compare_cnt <= self.nneighbors).all())
        all_deltas_weight = all_deltas_weight * valid_compare_cnt[:,None]
        # normalize weights to [0,1] so that L1 norm values are relative to max weight of 1
        all_deltas_weight = all_deltas_weight / all_deltas_weight.max()

        # option to scale the weights as a function of the comparison distance in the slice ordering.
        if neighbor_dist_scale is not None:
            assert(neighbor_dist_scale[0] == 1 and all(x <= 1 for x in neighbor_dist_scale))
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                ak = abs(k)
                #all_weights = np.zeros((self.total_nimgs,self.nneighbors,self.ngrid), dtype=np.double)
                all_deltas_weight[:,ik] = all_deltas_weight[:,ik] * neighbor_dist_scale[ak-1]

        # run the solver on all the deltas
        solved_deltas, _, inds = self._reconcile_grid_deltas(all_deltas, all_deltas_weight,
                valid_comparison, L1_norm=L1_norm, L2_norm=L2_norm, idw_p=idw_p, min_adj=min_valid_slice_comparisons,
                neighbors_radius_pixels=neighbors2D_radius_pixels, neighbors_std_pixels=neighbors2D_std_pixels,
                neighbors_W=neighbors2D_W, neighbors_expected=neighbors2D_expected, regr_bias=regr_bias,
                z_nnbrs_rad=z_neighbors_radius, nworkers=nworkers, iprocess=iprocess, nprocesses=nprocesses)

        # store the deltas before fitting affines to support reconcile parallelization
        self.cum_deltas = solved_deltas
        self.cum_deltas_inds = inds
        self.all_pts_src = all_pts_src

        if sproc:
            self.cum_deltas_inds = None # not needed for single process, only for merge
            self.refit_affines_rough_alignments(rough_distance_cutoff_pixels)

    # this function is broken off of reconcile_rough_alignments
    #   in order to support parallelization of the delta reconciler.
    def init_refit_affines_rough_alignments(self):
        # init vars per wafer
        if self.wafers_imaged_order_rough_affines is None:
            self.wafers_imaged_order_rough_affines = [None]*self.nwafer_ids
            self.wafers_imaged_order_rough_rigid_affines = [None]*self.nwafer_ids
            for i in range(self.nwafer_ids):
                self.wafers_imaged_order_rough_affines[i] = [None]*self.wafers_nregions[i]
                self.wafers_imaged_order_rough_rigid_affines[i] = [None]*self.wafers_nregions[i]
        else:
            # this is to apply the solved affine on top of an existing one
            self.wafers_imaged_order_rough_rigid_affines = self.wafers_imaged_order_rough_affines

        # init vars unrolled over all wafers
        self.cum_affines = [None]*self.total_nimgs
        self.cum_rigid_affines = [None]*self.total_nimgs

    # this function is broken off of reconcile_rough_alignments
    #   in order to support parallelization of the delta reconciler.
    def refit_affines_rough_alignments(self, rough_distance_cutoff_pixels, img_range=None):
        if img_range is None:
            self.img_range = [0, self.total_nimgs]
        else:
            self.img_range = img_range

        if self.wafer_aggregator_verbose:
            print('Refitting affines ({} total slices), img range {}-{}'.format(self.total_nimgs,
                self.img_range[0],self.img_range[1]))
            t = dt = time.time()

        self.init_refit_affines_rough_alignments()
        poly = preprocessing.PolynomialFeatures(degree=1)

        # turn the solved slices back into affine transformations to be applied to each slice.
        clf = linear_model.LinearRegression(fit_intercept=False, n_jobs=self.nthreads)
        grid_pts_src = poly.fit_transform(self.grid_locations_pixels)
        rigid_type = self.rigid_type
        # keep the rigid fit as rigid if we are fitting a full affine xform
        if rigid_type == RigidRegression_types.affine: rigid_type = RigidRegression_types.rigid
        clf_rigid = RigidRegression(rigid_type=rigid_type)
        nprint = 2000
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            if self.wafer_aggregator_verbose and (i+1)%nprint == 0:
                print('\t%d done in %.4f s' % (nprint, time.time() - dt, )); dt = time.time()
            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice = self.solved_orders[i_wafer_ind][i_ind]
            #i_wafer_id = self.wafer_ids[i_wafer_ind]

            # remove any grid points that were not nearby any original sift points before fitting affine.
            pts_sel = np.zeros((self.ngrid,), dtype=bool)
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                if self.all_pts_src[i][ik] is None: continue
                knbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.all_pts_src[i][ik])
                kdist, _ = knbrs.kneighbors(self.grid_locations_pixels, return_distance=True)
                pts_sel = np.logical_or(pts_sel, kdist.reshape(-1) < rough_distance_cutoff_pixels)

            assert(pts_sel.sum() > 12) # not enough grid points nearby
            #print('at img ind {}, {} / {} pts selected'.format(i,pts_sel.sum(),self.ngrid))
            #make_delta_plot(solved_centers,deltas=self.cum_deltas[i,:,:], grid_sel=pts_sel); plt.show()

            grid_pts_dst = self.grid_locations_pixels[pts_sel,:] + self.cum_deltas[i,pts_sel,:]
            clf.fit(grid_pts_src[pts_sel,:], grid_pts_dst)
            # scikit learn puts constant terms on the left, flip and augment
            caffine = clf.coef_
            caffine = np.concatenate( (np.concatenate( (caffine[:,1:], caffine[:,0][:,None]), axis=1 ),
                                       np.zeros((1,3), dtype=caffine.dtype)), axis=0 )
            caffine[2,2] = 1

            # also fit rigid affines, translation and rotation only
            clf_rigid.fit(self.grid_locations_pixels[pts_sel,:], grid_pts_dst)

            # save current affine for the wafer and over all wafers.
            A = self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice]
            if A is None:
                self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice] = caffine
                self.cum_affines[i] = caffine
                self.wafers_imaged_order_rough_rigid_affines[i_wafer_ind][i_slice] = clf_rigid.coef_.copy()
                self.cum_rigid_affines[i] = clf_rigid.coef_.copy()
            else:
                # apply the affine on top of an existing affine
                B = np.dot(caffine, A)
                self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice] = B
                self.cum_affines[i] = B
                B = np.dot(clf_rigid.coef_, A)
                self.wafers_imaged_order_rough_rigid_affines[i_wafer_ind][i_slice] = B
                self.cum_rigid_affines[i] = B

        # save memory, these are no longer needed after the affine fits
        self.cum_deltas = None
        self.all_pts_src = None

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

    # new rough alignment "reconciler" >>>

    # <<< new fine alignment "reconciler"

    def _set_block_info(self, iblock):
        self.iblock = iblock
        self.itblock = np.ravel_multi_index(np.array(iblock)[:,None], self.nblocks)[0]
        #self.first_block = all([x==0 for x in iblock])
        igrid_min = np.floor(self.grid_locations_pixels.min(0)).astype(np.int64)
        igrid_max = np.ceil(self.grid_locations_pixels.max(0)).astype(np.int64)
        _, _, _, rng = tile_nblks_to_ranges(igrid_max - igrid_min, self.nblocks, self.block_overlap_pix, iblock,
                ignore_bounds=True)
        blk_min = np.array([rng[0][0], rng[1][0]]) + igrid_min
        blk_max = np.array([rng[0][1], rng[1][1]]) + igrid_min
        self.blk_grid_pts_sel = np.logical_and(self.grid_locations_pixels >= blk_min,
                self.grid_locations_pixels <= blk_max).all(1)
        self.blk_grid_pts = self.grid_locations_pixels[self.blk_grid_pts_sel,:]
        self.blk_ngrid = self.blk_grid_pts.shape[0]
        self.blk_grid_pts_novlp_sel = np.logical_and(\
                self.grid_locations_pixels >= blk_min + self.block_overlap_pix,
                self.grid_locations_pixels <= blk_max - self.block_overlap_pix).all(1)
        self.blk_grid_pts_blk_novlp_sel = np.logical_and(\
                self.blk_grid_pts >= blk_min + self.block_overlap_pix,
                self.blk_grid_pts <= blk_max - self.block_overlap_pix).all(1)

    #     ### Params to mls_rigid_transform:
    #         * v - ndarray: an array with size [m, 2], points to transform
    #         * p - ndarray: an array with size [n, 2], control points
    #         * q - ndarray: an array with size [n, 2], deformed control points
    def _mls_rigid_transform_nbhd(self, v, p, q, inlier_nneighbors, nworkers):
        # inits
        inproc = v.shape[0]
        # NOTE: this is different from most of the other MP looks, in that it does not return each
        #   result individually. since the results are just single points this is slow with the queue.
        #   additionally, the blocks should be small enough that the returned results are not over the
        #   max size allowable for passing through the queue (pickling limitations).
        nproc = nworkers
        workers = [None]*nworkers
        result_queue = mp.Queue(nproc)
        inds = np.arange(inproc)
        inds = np.array_split(inds, nworkers)

        for i in range(nworkers):
            workers[i] = mp.Process(target=mls_rigid_transform_nbhd_job, daemon=True,
                    args=(i, v[inds[i][0]:inds[i][-1]+1], p, q, inlier_nneighbors, result_queue, False))
            workers[i].start()
        # NOTE: only call join after queue is emptied
        # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

        # output of this function, interpolated results from mls
        f_v = np.zeros_like(v)

        # collect the worker results.
        # nprint = 1
        # dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        i = 0
        while i < nproc:
            #if self.wafer_aggregator_verbose and i>0 and i%nprint==0:
            #    print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
            #    print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            j = res['iworker']
            f_v[inds[j][0]:inds[j][-1]+1] = res['f_v']
            worker_cnts[j] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        return f_v
    # def _mls_rigid_transform_nbhd

    def init_fine(self, grid_locations_pixels, alignment_folders, delta_dill_fn_str, neighbor_range, crops_um,
            griddist_pixels, img_range=None, load_type='outliers', load_filtered_deltas=False, run_str='none',
            nblocks=[1,1], iblock=[0,0], block_overlap_pix=[0,0], load_interpolated_deltas=False, keep_xcorrs=False,
            fine_interp_weight=1., fine_interp_neighbor_dist_scale=None, zero_deltas_indices=[]):
        self.ngrid = grid_locations_pixels.shape[0]
        # since the addition of always appplying the rough bounding box, center the grid points on it.
        self.grid_locations_pixels = grid_locations_pixels + self.rough_bounding_box_size[None,:]/2
        self.neighbor_range = neighbor_range
        self.max_neighbor_range = max(neighbor_range)
        # self.nneighbors and self.neighbor_rngs are everywhere, so decided to set it to the max
        #   and only use the list that is indexed by the crop index where it is necessary.
        self.neighbor_rng = [x for x in range(-self.max_neighbor_range,self.max_neighbor_range+1) if x != 0]
        self.nneighbors = len(self.neighbor_rng)
        # for the neighbor range indexed by crop index
        self.neighbor_rngs = [[x for x in range(-y,y+1) if x != 0] for y in self.neighbor_range]
        self.nneighbors_all = [len(x) for x in self.neighbor_rngs]

        assert( load_type in ['outliers', 'outliers-merge', 'reconcile', 'process_fine', 'solved'] )

        # setup for blockwise processing
        self.nblocks = nblocks; self.block_overlap_pix = block_overlap_pix
        self.ntblocks = nblocks[0]*nblocks[1]
        self.single_block = all([x==1 for x in nblocks])
        self._set_block_info(iblock)

        # these are only preserved for updating the fine dill files with outliers in update_fine
        self.alignment_folders = alignment_folders
        self.delta_dill_fn_str = delta_dill_fn_str
        self.delta_run_str = run_str
        self.crops_um = crops_um
        self.range_crops = len(crops_um)

        if self.C_hard_cutoff is not None:
            assert(len(self.C_hard_cutoff) >= self.max_neighbor_range) # need number of C_hard_cutoffs == max_skips

        # option to only process subset of the total image range
        if img_range is None:
            self.img_range = [0, self.total_nimgs]
        else:
            self.img_range = img_range

        # the distance to the "immediate" nearest neighbors, for hex grids the spacing of the points.
        # compute an adjacancy matrix of the immediate neighbors, used in outlier detection.
        self.neighbors_grid_dist_pixels = griddist_pixels

        if load_type == 'outliers':
            # pre-compute grid points related values need for outlier detection only
            # NOTE: previously several values were computed based on full distance matrix of the grid points.
            #   this works for grids up to maybe about 10000 points, but then this becomes prohibitive.
            #   so do not do this anymore for the grid points, instead compute using nearest neighbor searches.
            if self.single_block:
                grid_pts = self.grid_locations_pixels
                ngrid = self.ngrid
            else:
                grid_pts = self.blk_grid_pts
                ngrid = self.blk_ngrid

            # precompute the "immediate" nearest neighbors using the grid distance.
            nbrs = NearestNeighbors(radius=1.5*self.neighbors_grid_dist_pixels,
                    algorithm='ball_tree').fit(grid_pts)
            edges_i = nbrs.radius_neighbors(grid_pts, return_distance=False)
            cnts = np.array([x.size for x in edges_i]); edges_i = np.concatenate(edges_i)
            edges_j = np.concatenate([x*np.ones(y, dtype=np.int64) for x,y in zip(range(ngrid), cnts)])
            adj = sp.coo_matrix((np.ones(2*cnts.sum(), dtype=bool), (np.concatenate((edges_i, edges_j)),
                    np.concatenate((edges_j, edges_i)))), shape=(ngrid,ngrid)).tolil()
            adj.setdiag(0); self.grid_neighbors_adj = adj

            # this is used in outlier detection for "lesser weighted outliers"
            nbrs = NearestNeighbors(n_neighbors=self.ninlier_neighhbors_cmp[1],
                    algorithm='kd_tree').fit(grid_pts)
            self.inlier_neighbors_argsort = nbrs.kneighbors(grid_pts, return_distance=False)

            # this is an optimization that is used in wafer fine alignment on the next crop iteration,
            #   so that xcorrs are only computed for outliers that are close to current inliers.
            if self.fine_nearby_points_pixels is not None:
                nbrs = NearestNeighbors(radius=self.fine_nearby_points_pixels,
                        algorithm='ball_tree').fit(grid_pts)
                # should not matter that the same point is included in the nearest neighbor lists.
                self.grid_points_nearby = nbrs.radius_neighbors(grid_pts, return_distance=False)

        # allocate anything with grid points as python arrays to avoid wasting memory
        #   when loading limited image range and particularly with lots of grid points.
        # old unrolled method for reference, very wasteful for memory
        # self.deltas = np.zeros((self.total_nimgs,self.nneighbors,self.ngrid,2), dtype=np.double)
        # self.xcorrs = np.zeros((self.total_nimgs,self.nneighbors,self.ngrid), dtype=np.double)
        self.deltas = [None]*self.total_nimgs
        self.xcorrs = [None]*self.total_nimgs
        self.fine_weights = [None]*self.total_nimgs
        self.sel_inliers = [None]*self.total_nimgs
        self.deltas_inliers = [None]*self.total_nimgs
        if load_type == 'reconcile' or load_type == 'process_fine':
            self.all_fine_outliers = [None]*self.total_nimgs
            self.fine_valid_comparison = np.zeros((self.total_nimgs,self.nneighbors), dtype=bool)
            # self.all_fine_valid_comparisons = \
            #         np.zeros((self.total_nimgs,self.nneighbors,self.range_crops), dtype=bool)
            self.fine_weights_maxes = np.zeros((self.total_nimgs,), dtype=np.double)
            if load_type == 'reconcile':
                # this is only used for plotting
                self.count_included = np.zeros((self.total_nimgs,self.nneighbors), dtype=np.int64)

        self._compute_skip_lens()

        # for this load type, do not load any info from the delta dills
        if load_type == 'solved': return

        # this is an optimization to not have to reload dill files more than once.
        # this was easier to implement than having this loop driven by the delta dill indices.
        dill_dict_cache = {}

        # outer loop over slices
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            self.deltas[i] = np.zeros((self.nneighbors,self.ngrid,2), dtype=np.double)
            self.xcorrs[i] = np.zeros((self.nneighbors,self.ngrid), dtype=np.double)
            self.fine_weights[i] = np.zeros((self.nneighbors,self.ngrid), dtype=np.double)

            if load_type == 'reconcile':
                # because of the reslice, it's much easier to unroll this for the reconcile mode
                self.all_fine_outliers[i] = np.zeros((self.range_crops,self.nneighbors,self.ngrid), dtype=bool)
            elif load_type == 'process_fine':
                # leave the process_fine mode as it was, except with crops/images dims inverted
                self.all_fine_outliers[i] = [[None]*self.nneighbors for x in range(self.range_crops)]

            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            #i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            #i_wafer_id = self.wafer_ids[i_wafer_ind]

            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                #j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                #j_wafer_id = self.wafer_ids[j_wafer_ind]
                #print('\nProcessing total order ind %d' % (i,))
                #print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                #      (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))
                # every comparison is always stored in the lesser indexed dill file by run_wafer.py
                # this part is consistent with the rough aligment.
                if k < 0:
                    wafer_ind = j_wafer_ind; ind = j_ind
                    # this depends on which way the comparison is made in wafer.py.
                    direction_str = 'forward' # for cur->next images as template->image
                    #direction_str = 'reverse' # for cur->next images as image->template
                else:
                    wafer_ind = i_wafer_ind; ind = i_ind
                    direction_str = 'reverse' # for cur->next images as template->image
                    #direction_str = 'forward' # for cur->next images as image->template

                # cross-wafer comparison dills are named specially. the order index is negative
                #   and denotes the size of the skip.
                # xxx - this special file name for the cross wafer skips was not a great idea
                if i_wafer_ind == j_wafer_ind: # normal case
                    fn = os.path.join(self.alignment_folders[wafer_ind], self.delta_dill_fn_str.format(\
                            self.delta_run_str, self.wafer_ids[wafer_ind], ind))
                else: # cross-wafer
                    # NOTE: the cross-wafer alignemnts are ALWAYS stored in the negative index dill
                    #   files of the previous wafer (lesser indexed wafer). The less indexed wafer
                    #   is always selected as the index/wafer index above.
                    fn = os.path.join(self.alignment_folders[wafer_ind], self.delta_dill_fn_str.format(\
                            self.delta_run_str, self.wafer_ids[wafer_ind], ind-self.wafers_nimgs[wafer_ind]))
                # possibility of concurrent read/write dill access here when running process parallized
                #   outlier detection. the dills are updated during outlier detection in update_fine.
                if fn in dill_dict_cache:
                    # this is an optimization to not have to reload dill files more than once.
                    d = dill_dict_cache[fn]
                else:
                    if load_type == 'reconcile' or (load_type == 'outliers' and not self.single_block):
                        with open(fn, 'rb') as f: d = dill.load(f)
                    else:
                        d = dill_lock_and_load(fn)
                    dill_dict_cache[fn] = d
                    dill_dict_cache[fn]['count'] = 0
                dill_dict_cache[fn]['count'] = dill_dict_cache[fn]['count'] + 1
                if dill_dict_cache[fn]['count'] == self.nneighbors:
                    # if all the keys from this dict have been loaded, remove from the cache.
                    del dill_dict_cache[fn]

                # check if comparison bridges the specified "zero out" indices
                izero = -1
                for ii in range(len(zero_deltas_indices)):
                    if any([(zero_deltas_indices[ii][0] - x) >= 0 for x in [i,j]]) and \
                            any([(x - zero_deltas_indices[ii][1]) >= 0 for x in [i,j]]):
                        izero = ii; break
                #if izero > -1:
                #    print('\tcomparison at zorders {} and {}, crosses blur indices {} to {}'.format(i, j,
                #        zero_deltas_indices[izero][0], zero_deltas_indices[izero][1]))

                # loop over crops, do this loop nested inside outer loops.
                # the crop iteration as the outmost loop is very inefficient because
                #   it would have to reload the dills for each crop iteration.
                grid_selects = None
                for c in range(self.range_crops):

                    # create a single set of deltas and xcorrs from runs at different crops sizes.
                    # the workflow is only to run the next higher crop size if deltas from
                    #   the previous crop size were detected as outliers.
                    key_str = 'crop{}-skip{}-{}'.format(c, ak-1, direction_str)

                    # neighbor_rngs is to support less neighbors for larger crops.
                    if k in self.neighbor_rngs[c]:
                        # the first index from deltas and cvals is because wafer still supports aligning
                        #   more than 2 regions at once, even though this is not the normal workflow.
                        if grid_selects is None:
                        #if grid_selects[i][ik] is None:
                            self.deltas[i][ik,:,:] = d[key_str]['wafer_grid_deltas'][1,:,:]
                            self.xcorrs[i][ik,:] = d[key_str]['wafer_grid_Cvals'][1,:]
                        else:
                            self.deltas[i][ik,grid_selects,:] = d[key_str]['wafer_grid_deltas'][1,grid_selects,:]
                            self.xcorrs[i][ik,grid_selects] = d[key_str]['wafer_grid_Cvals'][1,grid_selects]

                    if load_type == 'reconcile' or load_type == 'process_fine':
                        # get the outliers select for loading the next crop size
                        #grid_selects = d[key_str]['outliers']
                        grid_selects = d[key_str]['nearby_outliers']

                        if load_type == 'reconcile':
                            # this is only used for generating the percent outlier plots, meh
                            self.all_fine_outliers[i][c,ik,d[key_str]['outliers']] = 1
                        elif load_type == 'process_fine':
                            self.all_fine_outliers[i][c][ik] = d[key_str]['outliers']
                        # self.all_fine_valid_comparisons[i,ik,c] = d[key_str]['valid_comparison']

                        # only load the weights and optionally set interpolated values
                        #   from those calculated/stored in the last crop (the final set of outliers).
                        if c == self.range_crops - 1:
                            self.fine_weights[i][ik,:] = d[key_str]['weights']
                            self.fine_valid_comparison[i,ik] = d[key_str]['valid_comparison']

                            if load_interpolated_deltas:
                                sel = np.isfinite(d[key_str]['wafer_grid_interp_deltas']).all(1)
                                self.deltas[i][ik,sel,:] = d[key_str]['wafer_grid_interp_deltas'][sel,:]
                                # NOTE: this final weighting then determines the weight of the interpolated deltas
                                #   relative to the rest of the deltas (set during outlier detection).
                                w = fine_interp_weight
                                if fine_interp_neighbor_dist_scale is not None:
                                    w *= fine_interp_neighbor_dist_scale[ak-1]
                                self.fine_weights[i][ik,sel] = w*d[key_str]['interp_weights'][sel]
                                # xxx - hacky, method to remove more outlier deltas based on interpolation
                                sel = (self.fine_weights[i][ik,:] < 0); self.fine_weights[i][ik,sel] = 0.

                            if load_filtered_deltas:
                                self.deltas[i][ik,:,:] = d[key_str]['wafer_grid_filtered_deltas']

                            if izero > -1:
                                self.deltas[i][ik,:,:] = 0.
                                sel = np.isfinite(self.xcorrs[i][ik,:])
                                # xxx - for now decided to just zero the direct neighbors, parameterize?
                                if ak == 1:
                                    #print('\tzero deltas')
                                    self.xcorrs[i][ik,sel] = 0.999
                                    # should only need this for the reconcile pathway anyways
                                    self.all_fine_outliers[i][c,ik,d[key_str]['outliers']] = 0
                                    self.fine_weights[i][ik,:] = 1.
                                    self.fine_valid_comparison[i,ik] = 1
                                else:
                                    #print('\tinvalidate comparison')
                                    self.xcorrs[i][ik,sel] = 0.
                                    # should only need this for the reconcile pathway anyways
                                    self.all_fine_outliers[i][c,ik,sel] = 0
                                    self.fine_weights[i][ik,:] = 0.
                                    self.fine_valid_comparison[i,ik] = 0
                        #if c == self.range_crops - 1:

                    # for outliers run, do not load the outliers select
                    #   for the last crop, as these are what we will be calculating.
                    elif (load_type == 'outliers' or load_type == 'outliers-merge') and  c < self.range_crops - 1:
                        #grid_selects = d[key_str]['outliers']
                        grid_selects = d[key_str]['nearby_outliers']
                    #if/elif load_type
                #for c in range(self.range_crops):
            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):

            if load_type == 'reconcile' or load_type == 'process_fine':
                valid_compare_cnt = self.fine_valid_comparison[i,:].sum()
                assert( valid_compare_cnt <= self.nneighbors ) # sanity check failed
                # scale weights so slices with less valid comparisons contribute less.
                self.fine_weights[i] = self.fine_weights[i] * valid_compare_cnt
                # the maxes are used to compute an overall max which is normalized by in reconcile
                self.fine_weights_maxes[i] = self.fine_weights[i].max()

            if load_type != 'outliers-merge':
                if load_type == 'process_fine':
                    # xxx - basically this is the interpolation pathway
                    #   the others may be dead code paths and hopefully do not need block support anyways.
                    # for block interpolation we still want to keep around all the inliers for the whole slice.
                    self.sel_inliers[i] = (self.fine_weights[i] > 0)
                    self.deltas_inliers[i] = [None]*self.nneighbors
                    # have to unroll this so we can still index over neighbors
                    for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                        self.deltas_inliers[i][ik] = self.deltas[i][ik,self.sel_inliers[i][ik,:],:]
                    # save memory by already turning this into a bool, actual xcorr values are not used
                    self.xcorrs[i] = np.isfinite(self.xcorrs[i])
                # for block processing, save memory by selecting out only the block
                if not self.single_block:
                    self.fine_weights[i] = self.fine_weights[i][:,self.blk_grid_pts_sel]
                    self.deltas[i] = self.deltas[i][:,self.blk_grid_pts_sel,:]
                if load_type == 'reconcile':
                    # collect the count included information for plotting / detecting missing slices
                    self.count_included[i,:] = np.isfinite(self.xcorrs[i]).sum(1)
                    if not keep_xcorrs:
                        self.xcorrs[i] = None # to save memory, not needed for fine accumulation
                elif not self.single_block:
                    self.xcorrs[i] = self.xcorrs[i][:,self.blk_grid_pts_sel]
            #if load_type != 'outliers-merge':
        #for i in range(self.img_range[0],self.img_range[1]):
    #def init_fine(self,

    # currently this only saves the outliers from the last crop run so that only the cross-correlations
    #   for the outlier points are rerun with a larger crop size.
    # xxx - most of this code is shared with init_fine, merge somehow? create an iterator, use examples in emdrp
    def update_fine(self, update_type='outliers', merge_inliers_blk_cutoff=0, merge_inliers_min_comp=0,
            merge_inliers_min_hole_comp=0):
        assert( update_type in ['outliers', 'outliers-merge', 'block-init', 'filtered_deltas',
                'interpolated_deltas'] )

        #ngrid, 2, nwafer_ids, range_skips
        # outer loop over slices
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            i_wafer_id = self.wafer_ids[i_wafer_ind]

            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                j_wafer_id = self.wafer_ids[j_wafer_ind]
                # every comparison is always stored in the lesser indexed dill file by run_wafer.py
                # this part is consistent with the rough aligment.
                if k < 0:
                    wafer_ind = j_wafer_ind; ind = j_ind
                    # this depends on which way the comparison is made in wafer.py.
                    direction_str = 'forward' # for cur->next images as template->image
                    #direction_str = 'reverse' # for cur->next images as image->template
                else:
                    wafer_ind = i_wafer_ind; ind = i_ind
                    direction_str = 'reverse' # for cur->next images as template->image
                    #direction_str = 'forward' # for cur->next images as image->template

                # cross-wafer comparison dills are named specially. the order index is negative
                #   and denotes the size of the skip.
                # xxx - this special file name for the cross wafer skips was not a great idea
                if i_wafer_ind == j_wafer_ind: # normal case
                    fn = os.path.join(self.alignment_folders[wafer_ind],
                            self.delta_dill_fn_str.format(self.delta_run_str, self.wafer_ids[wafer_ind], ind))
                else: # cross-wafer
                    # NOTE: the cross-wafer alignemnts are ALWAYS stored in the negative index dill
                    #   files of the previous wafer (lesser indexed wafer). The less indexed wafer
                    #   is always selected as the index/wafer index above.
                    fn = os.path.join(self.alignment_folders[wafer_ind], self.delta_dill_fn_str.format(\
                            self.delta_run_str, self.wafer_ids[wafer_ind], ind-self.wafers_nimgs[wafer_ind]))

                if (update_type == 'outliers' and not self.single_block) or update_type == 'block-init' or \
                        update_type == 'outliers-merge':
                    # storing the outlier block information into the same alignment dills was horrendously slow.
                    # store information separately by index (as before) and by first index of the block.
                    # using both indices with explode total dill files into millions (also inefficient).
                    dfn, pfn = os.path.split(fn)
                    dfn = os.path.join(dfn, 'blocks')
                    if update_type != 'outliers-merge':
                        dfn = os.path.join(dfn, '{:03d}'.format(self.iblock[0]))
                        #pfn += '.{}'.format(self.iblock[0]) # store in separate dirs instead
                        fn = os.path.join(dfn, pfn)
                    if update_type == 'block-init' and not os.path.isfile(fn):
                        # block-init should not be process parallelized, so only one process per dill file
                        os.makedirs(dfn, exist_ok=True)
                        with open(fn, 'wb') as f: dill.dump({}, f)

                # dills can be updated by multiple processes
                d, f1, f2 = dill_lock_and_load(fn, keep_locks=True)

                # save the current outlier/processed deltas in the last crop index specified.
                key_str = 'crop{}-skip{}-{}'.format(self.range_crops-1, ak-1, direction_str)

                if update_type == 'block-init' and key_str in d:
                    print(fn, key_str)
                    assert(False) # safety mechanism, delete the block dills by hand for overwrite

                if update_type != 'block-init':
                    # this key will not exist if the skips were reduced at a higher crop setting.
                    #   create the key so that the rest of the fine agg can go on as before, using the
                    #     last crop iteration to store the outliers.
                    if self.nneighbors_all[self.range_crops-1] < self.nneighbors and key_str not in d:
                        print('adding key {} to dill {}'.format(key_str,fn))
                        d[key_str] = {}

                if update_type=='outliers':
                    subd = d[key_str]
                    if not self.single_block:
                        if self.iblock[1] not in subd: subd[self.iblock[1]] = {}
                        subd = subd[self.iblock[1]]
                    subd['outliers'] = self.fine_outliers[i][ik]
                    subd['nearby_outliers'] = self.fine_nearby_outliers[i][ik]
                    subd['weights'] = self.fine_weights[i][ik,:]
                    subd['valid_comparison'] = self.fine_valid_comparison[i,ik]
                elif update_type=='block-init':
                    d[key_str] = {}
                elif update_type=='outliers-merge':
                    if self.verbose_iterations:
                        print('\nProcessing total order ind %d' % (i,))
                        print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                              (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))
                    all_outliers, all_nearby_outliers, all_weights, valid_comparison = \
                        self._outliers_block_merge(pfn, dfn, key_str, self.deltas[i][ik,:,:],
                            np.isfinite(self.xcorrs[i][ik,:]), merge_inliers_blk_cutoff,
                            merge_inliers_min_comp, merge_inliers_min_hole_comp)

                    subd = d[key_str]
                    subd['outliers'] = all_outliers
                    subd['nearby_outliers'] = all_nearby_outliers
                    subd['weights'] = all_weights
                    subd['valid_comparison'] = valid_comparison
                elif update_type=='filtered_deltas':
                    d[key_str]['wafer_grid_filtered_deltas'] = self.filtered_deltas[i,ik,:,:]
                elif update_type=='interpolated_deltas':
                    if self.single_block:
                        d[key_str]['wafer_grid_interp_deltas'] = self.fine_interp_deltas[i][ik,:,:]
                        d[key_str]['interp_weights'] = self.fine_interp_weights[i][ik,:]
                    else:
                        # xxx - or part is a hook for fixing when we had a bug here, remove
                        if 'wafer_grid_interp_deltas' not in d[key_str] or \
                                d[key_str]['wafer_grid_interp_deltas'].shape[0] != self.ngrid:
                            # initialize for the entire grid if nothing has been written yet
                            deltas = np.empty((self.ngrid,2), dtype=np.double); deltas.fill(np.nan)
                            d[key_str]['wafer_grid_interp_deltas'] = deltas
                            weights = np.zeros((self.ngrid,), dtype=np.double)
                            d[key_str]['interp_weights'] = weights
                        else:
                            deltas = d[key_str]['wafer_grid_interp_deltas']
                            weights = d[key_str]['interp_weights']

                        # update the block without any overlap points
                        sel = self.blk_grid_pts_novlp_sel; bsel = self.blk_grid_pts_blk_novlp_sel
                        deltas[sel,:] = self.fine_interp_deltas[i][ik,bsel,:]
                        weights[sel] = self.fine_interp_weights[i][ik,bsel]
                    #else: # if self.single_block:
                #update_type select

                # dills can be updated by multiple processes
                dill_lock_and_dump(fn, d, f1, f2)
            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
        #for i in range(self.img_range[0],self.img_range[1]):
    #def update_fine(self,

    # this is just because this code block was rolling too deep in update_fine.
    # NOTE: the plotting or suspension here will hang other runs of this script, because this is run
    #   inside of the delta dill lock.
    def _outliers_block_merge(self, pfn, dfn, key_str, deltas, sel_included, merge_inliers_blk_cutoff,
            merge_inliers_min_comp, merge_inliers_min_hole_comp):
        # need another outlier detection at the merge level.
        # the problem is that in a pariticular tile that should really be all outliers,
        #   a few spurious points will sneak through that randomly fit the affine.
        inliers_blks_pts = np.zeros((self.ntblocks,2), dtype=np.double)
        inliers_blks_deltas = np.zeros((self.ntblocks,2), dtype=np.double)
        inliers_blks_sel = np.ones(self.ntblocks, dtype=bool)
        npts = np.empty(self.ntblocks, dtype=np.int64)
        all_subd = [None]*self.ntblocks
        # iterate over all the blocks that the grid was tiled into
        for x in range(self.nblocks[0]):
            for y in range(self.nblocks[1]):
                self._set_block_info([x,y])
                fn = os.path.join(dfn, os.path.join(dfn, '{:03d}'.format(self.iblock[0])), pfn)
                #fn = os.path.join(dfn, pfn + '.{}'.format(self.iblock[0])) # store in separate dirs instead
                with open(fn, 'rb') as f: d = dill.load(f)
                assert(len(d[key_str]) == self.nblocks[1])
                subd = d[key_str][self.iblock[1]]

                # cache all the data from the dills so they do not have to be loaded again below
                all_subd[self.itblock] = subd

                # get the inliers points for this block
                sel = self.blk_grid_pts_novlp_sel; bsel = self.blk_grid_pts_blk_novlp_sel
                csel = np.zeros((self.blk_ngrid,), dtype=bool); csel[subd['outliers']] = 1
                csel = np.logical_not(csel[bsel]) # change to inlier select and remove overlap
                csel = np.logical_and(csel, sel_included[sel]) # also discard excluded points
                pts = self.blk_grid_pts[bsel,:][csel,:]; npts[self.itblock] = pts.shape[0]
                # very gross outlier block detection, remove blocks with less than a cutoff of inliers.
                #cutoff = np.round(bsel.sum() * merge_inliers_blk_perc)
                if pts.shape[0] < merge_inliers_blk_cutoff:
                    inliers_blks_sel[self.itblock] = 0
                    #p = self.blk_grid_pts
                    #if (p[0,:] > [43000,38250]).all() and (p[0,:] < [61250,53250]).all():
                    #    print(x,y)
                    #    csel2 = np.zeros((self.blk_ngrid,), dtype=bool); csel2[subd['outliers']] = 1
                    #    print(np.logical_not(csel2).sum())
                    #    make_delta_plot(self.blk_grid_pts, deltas=deltas[self.blk_grid_pts_sel,:],
                    #        figno=10, grid_sel_r=csel2); plt.show()
                else:
                    # get the closest inlier to the center of the block
                    ctr = (self.blk_grid_pts.max(0) + self.blk_grid_pts.min(0))/2
                    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts)
                    ind = nbrs.kneighbors(ctr[None,:], return_distance=False)[0][0]
                    inliers_blks_pts[self.itblock,:] = pts[ind,:]
                    inliers_blks_deltas[self.itblock,:] = deltas[sel,:][csel,:][ind,:]
        inliers_blks_pts = inliers_blks_pts[inliers_blks_sel,:]
        inliers_blks_deltas = inliers_blks_deltas[inliers_blks_sel,:]

        sel_blks_inliers = np.zeros(self.ntblocks, dtype=bool)
        sel_blks_inliers[inliers_blks_sel] = np.ones(inliers_blks_pts.shape[0], dtype=bool)

        # optionally run connected components on the blocks and threshold on component size.
        sel_remove = None
        if merge_inliers_min_comp > 0:
            bw = sel_blks_inliers.reshape(self.nblocks)
            labels, nlbls = nd.label(bw, structure=nd.generate_binary_structure(2,2))
            if nlbls > 0:
                sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                rmv = np.nonzero(sizes < merge_inliers_min_comp)[0] + 1
                if rmv.size > 0:
                    sel_remove = np.in1d(labels.reshape(-1), rmv)
                    sel_blks_inliers[sel_remove] = 0
                    sel_remove = sel_remove[inliers_blks_sel]

        # optionally run connected components on the block holes and threshold on component size.
        sel_add = None
        if merge_inliers_min_hole_comp > 0:
            bw = sel_blks_inliers.reshape(self.nblocks)
            labels, nlbls = nd.label(np.logical_not(bw), structure=nd.generate_binary_structure(2,1))
            if nlbls > 0:
                sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                add = np.nonzero(sizes < merge_inliers_min_hole_comp)[0] + 1
                if add.size > 0:
                    sel_add = np.in1d(labels.reshape(-1), add)
                    sel_blks_inliers[sel_add] = 1
                    sel_add = sel_add[inliers_blks_sel]

        if self.wafer_aggregator_verbose:
            print('\t{} of {} inlier blocks'.format(sel_blks_inliers.sum(), self.ntblocks))

        doplots = False # for debug only, do not expose b/c this code runs inside dill lock
        if doplots:
            # histograms
            #step = 0.025; Cbins = np.arange(0,1,step); Ccbins = Cbins[:-1] + step/2
            Chist,Cbins = np.histogram(npts[npts > 0], 50)
            #Chist,Cbins = np.histogram(nzC, Cbins)
            Ccbins = Cbins[:-1] + (Cbins[1]-Cbins[0])/2
            plt.figure(1); plt.gcf().clf()
            plt.plot(Ccbins, Chist)
            grid_sel_r = None
            make_delta_plot(inliers_blks_pts, deltas=inliers_blks_deltas,
                grid_sel_r=grid_sel_r, grid_sel_b=sel_remove, figno=10)
            make_delta_plot(inliers_blks_pts, deltas=inliers_blks_deltas,
                grid_sel_r=grid_sel_r, grid_sel_b=sel_add, figno=11)
            plt.show()

        # this is the actual block outlier merging loop,
        #   but uses the "meta-block-outliers" result from above.
        sel_outliers = np.zeros((self.ngrid,), dtype=bool)
        sel_nearby_outliers = np.zeros((self.ngrid,), dtype=bool)
        all_weights = np.zeros((self.ngrid,), dtype=np.double)
        valid_comparison = False
        # iterate over all the blocks that the grid was tiled into
        for x in range(self.nblocks[0]):
            for y in range(self.nblocks[1]):
                self._set_block_info([x,y])
                subd = all_subd[self.itblock]
                sel = self.blk_grid_pts_novlp_sel; bsel = self.blk_grid_pts_blk_novlp_sel

                if sel_blks_inliers[self.itblock]:
                    # inlier block, assimilate the results
                    tmp = np.zeros((self.blk_ngrid,), dtype=bool); tmp[subd['outliers']] = 1
                    sel_outliers[sel] = tmp[bsel]
                    tmp = np.zeros((self.blk_ngrid,), dtype=bool); tmp[subd['nearby_outliers']] = 1
                    sel_nearby_outliers[sel] = tmp[bsel]
                    all_weights[sel] = subd['weights'][bsel]
                    valid_comparison = valid_comparison or subd['valid_comparison']
                else:
                    # outlier block, mark all included points as outliers
                    sel_outliers[np.logical_and(sel, sel_included)] = 1
        all_outliers = np.nonzero(sel_outliers)[0]
        all_nearby_outliers = np.nonzero(sel_nearby_outliers)[0]

        return all_outliers, all_nearby_outliers, all_weights, valid_comparison
    #def _outliers_block_merge(self

    def _get_all_nblks(self, nworkers):
        # inits
        inproc = self.ntblocks
        # NOTE: this is different from most of the other MP looks, in that it does not return each
        #   result individually. since the results are just single points this is slow with the queue.
        #   additionally, the complete results should be small enough that the returned results are not
        #   over the max size allowable for passing through the queue (pickling limitation).
        nproc = nworkers
        workers = [None]*nworkers
        result_queue = mp.Queue(nproc)
        inds = np.arange(inproc)
        inds = np.array_split(inds, nworkers)

        # create unrolled iblocks
        tmp = np.meshgrid(np.arange(self.nblocks[0]), np.arange(self.nblocks[1]), indexing='ij')
        iblocks = np.concatenate([x.flat[:][:,None] for x in tmp], axis=1)
        assert(iblocks.shape[0] == self.ntblocks)

        for i in range(nworkers):
            workers[i] = mp.Process(target=get_block_info_job, daemon=True,
                    args=(i, iblocks[inds[i][0]:inds[i][-1]+1,:], self.nblocks, self.block_overlap_pix,
                        self.grid_locations_pixels, result_queue, False))
            workers[i].start()
        # NOTE: only call join after queue is emptied
        # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

        # collect the worker results.
        # nprint = 1
        # dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        blk_ngrid = np.zeros(self.ntblocks, dtype=np.int64)
        i = 0
        while i < nproc:
            #if self.wafer_aggregator_verbose and i>0 and i%nprint==0:
            #    print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
            #    print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            j = res['iworker']
            blk_ngrid[inds[j][0]:inds[j][-1]+1] = res['blk_ngrid']
            worker_cnts[j] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        return blk_ngrid
    # def _get_all_nblks

    def fine_deltas_reslice_init(self, fn, keep_xcorrs=False, nworkers=1, nprocesses=1):
        #assert(not self.single_block) # fine reslice not necessary for single block processing

        if self.single_block:
            blk_ngrid = max_blk_ngrid = self.ngrid
        else:
            print('Getting all block sizes with {} workers'.format(nworkers)); t=time.time()
            blk_ngrid = self._get_all_nblks(nworkers)
            max_blk_ngrid = blk_ngrid.max()
            print('\tmax block size (npts) is {}'.format(max_blk_ngrid))
            print('\tdone in %.4f s' % (time.time() - t, ))

        # the allocations for the attributes that need to be saved, for reference
        # self.fine_valid_comparison = np.zeros((self.total_nimgs,self.nneighbors), dtype=bool)
        # self.fine_weights_maxes = np.zeros((self.total_nimgs,), dtype=np.double)
        # self.count_included = np.zeros((self.total_nimgs,self.nneighbors), dtype=np.int64)
        # self.deltas = [None]*self.total_nimgs
        # self.deltas[i] = np.zeros((self.nneighbors,self.ngrid,2), dtype=np.double)
        # self.fine_weights = [None]*self.total_nimgs
        # self.fine_weights[i] = np.zeros((self.nneighbors,self.ngrid), dtype=np.double)
        # self.all_fine_outliers = [None]*self.total_nimgs
        # self.all_fine_outliers[i] = np.zeros((self.range_crops,self.nneighbors,self.ngrid), dtype=bool)

        # create the full shapes of the various datasets
        ishape = [self.total_nimgs]
        gshape = self.nblocks + [max_blk_ngrid]
        cshape = ishape + [self.range_crops] + [self.nneighbors] + gshape
        nshape = ishape + [self.nneighbors]
        wshape = nshape + gshape
        dshape = wshape + [2]
        bshape = ishape + gshape

        datasets = ['count_included', 'fine_valid_comparison', 'fine_weights_maxes', 'fine_weights',
                'deltas', 'all_fine_outliers', 'blk_grid_pts_blk_novlp_sel']
        custom_slcs = [np.s_[:1,:1], np.s_[:1,:1], (np.s_[:1],),
                np.s_[:1,:1,:1,:1,:1], np.s_[:1,:1,:1,:1,:1,:1], np.s_[:1,:1,:1,:1,:1,:1], np.s_[:1,:1,:1,:1]]
        shapes = [nshape, nshape, ishape, wshape, dshape, cshape, bshape]
        dtypes = [np.int64, bool, np.double, np.double, np.double, bool, bool]
        attrs = [{}, {}, {}, {}, {}, {'nprocesses':nprocesses}, {}]

        if keep_xcorrs:
            datasets += ['xcorrs']
            custom_slcs += [np.s_[:1,:1,:1,:1,:1]]
            shapes += [wshape]
            dtypes += [np.double]
            attrs += [{}]

        # initialize all the datasets with the correct shapes based on the number of blocks
        for dataset, custom_slc, shape, dtype, attr, i in zip(datasets,
                custom_slcs, shapes, dtypes, attrs, range(len(datasets))):
            data = np.zeros(np.ones(len(custom_slc), dtype=np.int64), dtype=dtype)
            big_img_save(fn, data, shape, dataset=dataset, compression=True, recreate=True,
                custom_slc=custom_slc, truncate=(i==0), attrs=attr)

        # # save the point counts in each block
        # data = blk_ngrid
        # big_img_save(fn, data, self.nblocks, dataset='blk_ngrid', compression=True, recreate=True)
    #def fine_deltas_reslice_init(self,

    def fine_deltas_reslice(self, fn, keep_xcorrs=False, nblocks=[1,1], block_overlap_pix=[0,0]):
        # setup for blockwise processing
        self.nblocks = nblocks; self.block_overlap_pix = block_overlap_pix
        self.ntblocks = nblocks[0]*nblocks[1]
        self.single_block = all([x==1 for x in nblocks])
        #assert(not self.single_block) # fine reslice not necessary for single block processing

        # fine_weights_maxes, fine_valid_comparison, count_included, all_fine_outliers, fine_weights, deltas
        ir = self.img_range

        datasets = ['count_included', 'fine_valid_comparison', 'fine_weights_maxes']
        custom_slcs = [np.s_[ir[0]:ir[1],:], np.s_[ir[0]:ir[1],:], np.s_[ir[0]:ir[1]]]
        # save the required data that is not blocked
        for dataset, custom_slc in zip(datasets, custom_slcs):
            data = getattr(self, dataset)[custom_slc]
            big_img_save(fn, data, dataset=dataset, custom_slc=custom_slc)

        # xxx - do not see a straight forward way to loop over dataset strings here
        # iterate the blocks and save the blocked data
        dt = time.time()
        for x in range(self.nblocks[0]):
            for y in range(self.nblocks[1]):
                self._set_block_info([x,y])
                data = np.concatenate([x[:,self.blk_grid_pts_sel][None,:,None,None,:] \
                        for x in self.fine_weights[ir[0]:ir[1]]], axis=0)
                custom_slc = np.s_[ir[0]:ir[1],:,x:x+1,y:y+1,:self.blk_ngrid]
                big_img_save(fn, data, dataset='fine_weights', custom_slc=custom_slc)
                data = np.concatenate([x[:,self.blk_grid_pts_sel,:][None,:,None,None,:,:] \
                        for x in self.deltas[ir[0]:ir[1]]], axis=0)
                custom_slc = np.s_[ir[0]:ir[1],:,x:x+1,y:y+1,:self.blk_ngrid,:]
                big_img_save(fn, data, dataset='deltas', custom_slc=custom_slc)
                data = np.concatenate([x[:,:,self.blk_grid_pts_sel][None,:,:,None,None,:] \
                        for x in self.all_fine_outliers[ir[0]:ir[1]]], axis=0)
                custom_slc = np.s_[ir[0]:ir[1],:,:,x:x+1,y:y+1,:self.blk_ngrid]
                big_img_save(fn, data, dataset='all_fine_outliers', custom_slc=custom_slc)
                data = self.blk_grid_pts_blk_novlp_sel
                custom_slc = np.s_[ir[0]:ir[1],x:x+1,y:y+1,:self.blk_ngrid]
                big_img_save(fn, data, dataset='blk_grid_pts_blk_novlp_sel', custom_slc=custom_slc)

                if keep_xcorrs:
                    data = np.concatenate([x[:,self.blk_grid_pts_sel][None,:,None,None,:] \
                            for x in self.xcorrs[ir[0]:ir[1]]], axis=0)
                    custom_slc = np.s_[ir[0]:ir[1],:,x:x+1,y:y+1,:self.blk_ngrid]
                    big_img_save(fn, data, dataset='xcorrs', custom_slc=custom_slc)
            #for y in range(self.nblocks[1]):

            print('\tx slice {} of {} done in {:.4f} s'.format(x+1, self.nblocks[0], time.time()-dt))
            dt = time.time()
        #for x in range(self.nblocks[0]):
    #def fine_deltas_reslice(self

    def fine_deltas_reslice_load(self, fn, img_ranges):
        self.deltas = [None]*self.total_nimgs
        self.fine_weights = [None]*self.total_nimgs
        self.all_fine_outliers = [None]*self.total_nimgs

        nproc = len(img_ranges)
        #dt = time.time()
        datasets = ['count_included', 'fine_valid_comparison', 'fine_weights_maxes']
        bfn, ext = os.path.splitext(fn)
        for i in range(nproc):
            cfn = bfn + '.{}'.format(i) + ext
            ir = img_ranges[i]
            nir = ir[1] - ir[0]
            custom_slcs = [np.s_[ir[0]:ir[1],:], np.s_[ir[0]:ir[1],:], np.s_[ir[0]:ir[1]]]

            # load this process blocked for datasets that are not x/y blocked
            for dataset, custom_slc, j in zip(datasets, custom_slcs, range(len(datasets))):
                shp, dtype = big_img_info(cfn, dataset=dataset)
                if i==0: setattr(self, dataset, np.empty(shp, dtype=dtype))
                data = getattr(self, dataset)
                big_img_load(cfn, img_blk=data, dataset=dataset, custom_slc=custom_slc, custom_dslc=custom_slc)

            # xxx - did not see a straight forward way to loop over dataset strings here
            x,y = self.iblock
            shp, dtype = big_img_info(cfn, dataset='fine_weights')
            shp = np.array(shp); shp[0] = nir; shp[2:4] = 1; shp[4] = self.blk_ngrid
            fine_weights = np.empty(shp, dtype=dtype)
            custom_slc = np.s_[ir[0]:ir[1],:,x:x+1,y:y+1,:self.blk_ngrid]
            big_img_load(cfn, img_blk=fine_weights, dataset='fine_weights', custom_slc=custom_slc)
            shp, dtype = big_img_info(cfn, dataset='deltas')
            shp = np.array(shp); shp[0] = nir; shp[2:4] = 1; shp[4] = self.blk_ngrid
            deltas = np.empty(shp, dtype=dtype)
            custom_slc = np.s_[ir[0]:ir[1],:,x:x+1,y:y+1,:self.blk_ngrid,:]
            big_img_load(cfn, img_blk=deltas, dataset='deltas', custom_slc=custom_slc)
            shp, dtype = big_img_info(cfn, dataset='all_fine_outliers')
            shp = np.array(shp); shp[0] = nir; shp[3:5] = 1; shp[5] = self.blk_ngrid
            all_fine_outliers = np.empty(shp, dtype=dtype)
            custom_slc = np.s_[ir[0]:ir[1],:,:,x:x+1,y:y+1,:self.blk_ngrid]
            big_img_load(cfn, img_blk=all_fine_outliers, dataset='all_fine_outliers', custom_slc=custom_slc)

            # unroll the image dimension into python lists
            for j,k in zip(range(ir[0], ir[1]), range(nir)):
                self.fine_weights[j] = fine_weights[k,:,:,:,:].reshape((self.nneighbors,self.blk_ngrid))
                self.deltas[j] = deltas[k,:,:,:,:,:].reshape((self.nneighbors,self.blk_ngrid,2))
                self.all_fine_outliers[j] = all_fine_outliers[k,:,:,:,:,:].\
                        reshape((self.range_crops,self.nneighbors,self.blk_ngrid))
            #print('\tchunk {} of {} done in {:.4f} s'.format(i+1, nproc, time.time()-dt)); dt = time.time()
        #for i in range(nproc):
    #def fine_deltas_reslice_load(self,

    def fine_deltas_outlier_detection(self, min_inliers=0, nworkers=1, doplots=False):
        if self.wafer_aggregator_verbose:
            print('Outlier detection fine, ngrid points {}, img range {}-{}'.format(self.ngrid,
                self.img_range[0],self.img_range[1]))

        # setup for blockwise processing
        if self.single_block:
            grid_pts = self.grid_locations_pixels
            ngrid = self.ngrid
        else:
            grid_pts = self.blk_grid_pts
            ngrid = self.blk_ngrid
            if self.wafer_aggregator_verbose:
                print('\tiblock {} {} of {} {}, ngrid points block {}'.format(self.iblock[0], self.iblock[1],
                    self.nblocks[0], self.nblocks[1], ngrid))

        # inits for the affine fits
        poly = preprocessing.PolynomialFeatures(degree=self.outlier_affine_degree)
        #clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)
        #ransac = linear_model.RANSACRegressor(base_estimator=clf, stop_probability=1-1e-6, max_trials=self.ransac_max,
        #                                      loss=lambda true,pred: np.sqrt(((true - pred)**2).sum(1)),
        #                                      residual_threshold=self.residual_threshold_pixels)
        ransac = AffineRANSACRegressor(stop_probability=1-1e-6, max_trials=self.ransac_max,
                                      # loss=lambda true,pred: np.sqrt(((true - pred)**2).sum(1)),
                                      # residual_threshold=self.residual_threshold_pixels)
                                      loss=lambda true,pred: ((true - pred)**2).sum(1),
                                      residual_threshold=self.residual_threshold_pixels**2)
        pts_src = poly.fit_transform(grid_pts)

        # old unrolled method for reference
        #all_weights = np.zeros((self.total_nimgs,self.nneighbors,ngrid), dtype=np.double)
        all_weights = [None]*self.total_nimgs
        all_outliers = [[None]*self.nneighbors for x in range(self.total_nimgs)]
        all_nearby_outliers = [[None]*self.nneighbors for x in range(self.total_nimgs)]
        # valid_comparison is the same as for rough alignment, same for all slice comparisons.
        # this is just a very rough flag as to whether a particular slice is valid,
        #   i.e., it's an inbounds neighbor comparison, and also if there is some minimum number
        #   of valid, i.e. inlier, deltas. false basically means a slice can not be aligned at all.
        valid_comparison = np.zeros((self.total_nimgs,self.nneighbors), dtype=bool)

        # use the grid dist to convert component size edge length to number of grid points
        inlier_min_component_size = \
            int((self.inlier_min_component_size_edge_pixels/self.neighbors_grid_dist_pixels)**2)

        # outer loop over slices, outlier detection is done independently over slices and skips
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            all_weights[i] = np.zeros((self.nneighbors,ngrid), dtype=np.double)

            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            i_wafer_id = self.wafer_ids[i_wafer_ind]
            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                j_wafer_id = self.wafer_ids[j_wafer_ind]

                if self.verbose_iterations:
                    print('\nProcessing total order ind %d' % (i,))
                    print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                          (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))

                cdeltas = self.deltas[i][ik,:,:]
                cxcorrs = self.xcorrs[i][ik,:]
                # mask of points included or not because they are outside of the roi polygons.
                # cross correlations were not run on excluded points.
                sel_included = np.isfinite(cxcorrs)

                sel_excluded = np.logical_not(sel_included)
                nincluded = sel_included.sum()
                nexcluded = sel_excluded.sum()
                if self.verbose_iterations:
                    print('\t%d excluded (outside polygon), %d included' % (nexcluded,nincluded))

                # initialize outliers mask and previous
                sel_outliers = np.zeros_like(sel_included)
                cnt_old = 0

                if self.C_hard_cutoff is not None:
                    if doplots:
                        # histograms
                        #step = 0.025; Cbins = np.arange(0,1,step); Ccbins = Cbins[:-1] + step/2
                        #nzC = xcorrs[i,ik,sel_included]
                        nzC = cxcorrs[sel_included]
                        Chist,Cbins = np.histogram(nzC, 50)
                        #Chist,Cbins = np.histogram(nzC, Cbins)
                        Ccbins = Cbins[:-1] + (Cbins[1]-Cbins[0])/2
                        plt.figure(1); plt.gcf().clf()
                        plt.plot(Ccbins, Chist)

                    for cak in range(ak-1,len(self.C_hard_cutoff)): # start at the neighbor index
                    #for cak in range(0,len(self.C_hard_cutoff)): # always start at 0, rely on perc inliers cutoff
                        # add any points as outliers that are below a hard xcorr cutoff.
                        sel_remove = (cxcorrs < self.C_hard_cutoff[cak])
                        # add new outliers, but do not include excluded points in the outlier mask
                        csel_outliers = np.logical_or(sel_outliers, sel_remove); csel_outliers[sel_excluded] = 0
                        # continue if the C-cutoff removes more than some percentage of the included points.
                        #print(nincluded-csel_outliers.sum())
                        if (nincluded-csel_outliers.sum()) >= int(self.min_percent_inliers_C_cutoff * nincluded): break
                    sel_outliers = csel_outliers
                    if self.verbose_iterations:
                        cnt = sel_outliers.sum()
                        print('\t%d outliers, C cutoff %g' % (cnt-cnt_old, self.C_hard_cutoff[cak]))
                        cnt_old = cnt

                pts_dst = grid_pts + cdeltas
                sel_not_missing = np.logical_not(np.logical_or(sel_excluded, sel_outliers))
                # do not even run ransac outliers detection if the number of included points
                #   is less than the required number of params for a linear model. this is mostly
                #   for outliers in blocks where most or all of the points are not included.
                if pts_dst.shape[1]+1 < sel_not_missing.sum():
                    _, mask = _ransac_repeat(pts_src[sel_not_missing,:], pts_dst[sel_not_missing,:], ransac,
                            self.ransac_repeats, verbose=self.wafer_aggregator_verbose, nworkers=nworkers)
                else:
                    mask = None
                sel_remove = np.zeros_like(sel_included)
                sel_remove[sel_not_missing] = np.logical_not(mask) if mask is not None else 0
                # add new outliers, but do not include excluded points in the outlier mask
                sel_outliers = np.logical_or(sel_outliers, sel_remove); sel_outliers[sel_excluded] = 0
                if self.verbose_iterations:
                    cnt = sel_outliers.sum()
                    print('\t%d outliers, affine fit' % (cnt-cnt_old,))
                    cnt_old = cnt

                if self.inlier_min_neighbors > 0:
                    # add any points as outliers that have less then some cutoff of neighbors remaining.
                    neighbors = self.grid_neighbors_adj.copy()
                    sel_missing = np.logical_or(sel_excluded, sel_outliers)
                    neighbors[sel_missing,:] = 0; neighbors[:,sel_missing] = 0
                    neighbors_cnt = neighbors.sum(1).A1
                    sel_remove = (neighbors_cnt < self.inlier_min_neighbors)
                    # add new outliers, but do not include excluded points in the outlier mask
                    sel_outliers = np.logical_or(sel_outliers, sel_remove); sel_outliers[sel_excluded] = 0
                    if self.verbose_iterations:
                        cnt = sel_outliers.sum()
                        print('\t%d outliers, neighbors cutoff' % (cnt-cnt_old,))
                        cnt_old = cnt

                if inlier_min_component_size > 0:
                    # run graph connected components on the remaining neighbors.
                    # remove components below some threshold size.
                    neighbors = self.grid_neighbors_adj.copy()
                    sel_missing = np.logical_or(sel_excluded, sel_outliers)
                    neighbors[sel_missing,:] = 0; neighbors[:,sel_missing] = 0
                    nlabels, labels = sp.csgraph.connected_components(neighbors, directed=False)
                    sizes = np.bincount(labels)
                    sel_remove = np.in1d(labels, np.nonzero(sizes < inlier_min_component_size)[0])
                    # add new outliers, but do not include excluded points in the outlier mask
                    sel_outliers = np.logical_or(sel_outliers, sel_remove); sel_outliers[sel_excluded] = 0
                    if self.verbose_iterations:
                        cnt = sel_outliers.sum()
                        print('\t%d outliers, components cutoff' % (cnt-cnt_old,))
                        cnt_old = cnt

                # save indices of outlier points. this is so these points can be re-run with a larger
                #   context around the template.
                all_outliers[i][ik] = np.nonzero(sel_outliers)[0]

                # number of points remaining at end of outlier detection.
                ninliers = (ngrid - sel_outliers.sum() - nexcluded)

                # instead of weighting based on outlier type (did not work), put in a lesser weighting
                #   for any outliers that are not too far off from the remaining nearest neighbors.
                sel_ok_outliers = np.zeros_like(sel_outliers)
                if self.ok_outlier_zscore != 0:
                    excluded_inds = np.nonzero(sel_excluded)[0]
                    for y,iy in zip(all_outliers[i][ik], range(all_outliers[i][ik].size)):
                        outliers_except_cur = all_outliers[i][ik][all_outliers[i][ik] != y]
                        not_inliers_inds = np.concatenate((outliers_except_cur, excluded_inds))
                        inds = np.setdiff1d(self.inlier_neighbors_argsort[y,:], not_inliers_inds, assume_unique=True)

                        # the first index in inds is that of the actual current outlier point,
                        #   based on argsort of the distance matrix (zero distance to same point).
                        # i.e., the first point index in inds is the current outlier (inds[0]==y).
                        assert(inds[0] == y) # sanity check

                        # instead of a full argsort, ninlier_neighhbors_cmp[1] controls the total argsort size.
                        # only consider a point as an ok outlier if there enough inliers within this neighborhood.
                        if inds.size >= self.ninlier_neighhbors_cmp[0]:
                            sel_ok_outliers[y] = self._neighborhood_based_inliers(inds, cdeltas,
                                    self.ok_outlier_zscore)

                # optionally iterate again over inliers and demote any that do not fit similar criteria
                #   as the ok outliers. this can help weed out a few remaining spurious deltas in the case
                #   that a large ransac residual was used (to deal with big splits for example).
                sel_not_ok_inliers = np.zeros_like(sel_included)
                if self.not_ok_inlier_zscore != 0:
                    cinliers = np.nonzero(np.logical_or(np.logical_not(np.logical_or(sel_excluded, sel_outliers)),
                            sel_ok_outliers))[0]
                    for y,iy in zip(cinliers, range(cinliers.size)):
                        inds = np.intersect1d(self.inlier_neighbors_argsort[y,:], cinliers, assume_unique=True)

                        # the first index in inds is that of the actual current outlier point,
                        #   based on argsort of the distance matrix (zero distance to same point).
                        # i.e., the first point index in inds is the current outlier (inds[0]==y).
                        # intersect1d sorts the output, put y back as the first element.
                        inds = np.concatenate(([y], inds[inds != y]))

                        # instead of a full argsort, ninlier_neighhbors_cmp[1] controls the total argsort size.
                        if inds.size >= self.ninlier_neighhbors_cmp[0]:
                            sel_not_ok_inliers[y] = not self._neighborhood_based_inliers(inds, cdeltas,
                                self.not_ok_inlier_zscore, verbose=False)
                        else:
                            # xxx - not enough inliers in the neighborhood, then reject this inlier?
                            sel_not_ok_inliers[y] = 1
                            #pass

                    # add the not ok inliers to the outliers
                    sel_outliers[sel_not_ok_inliers] = 1
                    cnt = sel_outliers.sum()

                    # remove any ok outliers that were not ok inliers
                    sel_ok_outliers[sel_not_ok_inliers] = 0

                    # save indices of outlier points. this is so these points can be re-run with a larger
                    #   context around the template.
                    all_outliers[i][ik] = np.nonzero(sel_outliers)[0]

                    # number of points remaining at end of outlier detection.
                    ninliers_orig = ninliers
                    ninliers = (ngrid - sel_outliers.sum() - nexcluded)
                else: #if self.not_ok_inlier_zscore != 0:
                    ninliers_orig = ninliers

                # xxx - with the tissue masks enabled now, this can probably be completely removed
                #   was old feature to only run xcorrs at next crop for outliers that are within
                #     some distance of the current inlier set.
                nearby_outliers = np.ones(all_outliers[i][ik].size, dtype=bool)
                if self.fine_nearby_points_pixels is not None:
                    missing = np.nonzero(np.logical_or(sel_outliers, sel_excluded))[0]
                    for y,iy in zip(all_outliers[i][ik], range(all_outliers[i][ik].size)):
                        # optionally create another outliers select that is only the outliers that are
                        #   within a certain distance of any inlier.
                        if np.in1d(self.grid_points_nearby[y], missing, assume_unique=True).all():
                            nearby_outliers[iy] = 0
                all_nearby_outliers[i][ik] = all_outliers[i][ik][nearby_outliers]

                # set the weightings to one for inliers, and to a small value for "ok outliers"
                sel_inliers = np.logical_not(np.logical_or(sel_excluded, sel_outliers))
                all_weights[i][ik,sel_inliers] = 1.
                # xxx - parameterize the relative weighting?
                #all_weights[i][ik,sel_ok_outliers] = 0.05
                all_weights[i][ik,sel_ok_outliers] = 1.
                # probably better to totally reject these, or if not, disable this feature
                #all_weights[i][ik,sel_not_ok_inliers] = 0.01

                # make this as a valid comparison as long as there is at least one included delta remaining.
                # this is more of a "gross-check". there needs to be at least one valid simplex remaining,
                #   but this is handled before calling the solver by skipping adjacencies if any of the weights
                #   for a given simplex are zero.
                # changed this for downstream usage by already rejecting this comparison as invalid if the slice
                #   can not be interpolated. with the MLS interpolation, this value is low but not zero.
                valid_comparison[i,ik] = (ninliers > min_inliers)

                if self.verbose_iterations:
                    print('\t%d / %d included' % (ninliers, nincluded))
                    print('\t%d / %d outliers are lesser weighted outliers' % (sel_ok_outliers.sum(),cnt))
                    print('\t%d / %d inliers are lesser weighted rejected inliers' % \
                        (sel_not_ok_inliers.sum(),ninliers_orig))
                    print('\t%d / %d outliers are nearby outliers' % (all_nearby_outliers[i][ik].size,cnt))

                if doplots:
                    # histograms
                    dmags = np.sqrt((cdeltas*cdeltas).sum(1))
                    dhist,dbins = np.histogram(dmags, 50)
                    dcbins = dbins[:-1] + (dbins[1]-dbins[0])/2
                    plt.figure(2); plt.gcf().clf()
                    plt.plot(dcbins, dhist)

                    # make_delta_plot(grid_pts, deltas=deltas[i,ik,:,:],
                    #         figno=10, grid_sel_b=sel_excluded, grid_sel_r=sel_outliers)
                    # tmp = deltas[i,ik,:,:].copy()
                    make_delta_plot(grid_pts, deltas=cdeltas,
                            figno=10, grid_sel_b=sel_excluded, grid_sel_r=sel_outliers)
                    tmp = cdeltas.copy()
                    # show ok outliers as red points that still have deltas
                    tmp[np.logical_and(sel_outliers, np.logical_not(sel_ok_outliers)),:] = 0
                    #tmp[sel_outliers,:] = 0 # do not show ok outliers at all
                    make_delta_plot(grid_pts, deltas=tmp,
                            figno=11, grid_sel_b=sel_excluded, grid_sel_r=sel_outliers)
                    # color the nearby outliers instead of excluded
                    sel_nearby = np.zeros_like(sel_excluded)
                    sel_nearby[all_nearby_outliers[i][ik]] = 1
                    make_delta_plot(grid_pts, deltas=tmp,
                            figno=12, grid_sel_b=sel_nearby, grid_sel_r=sel_outliers)
                    plt.show()

            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
        #for i in range(self.total_nimgs):

        self.fine_weights = all_weights
        self.fine_outliers = all_outliers
        self.fine_nearby_outliers = all_nearby_outliers
        self.fine_valid_comparison = valid_comparison
    #def fine_deltas_outlier_detection(self,

    # helper function for fine_deltas_outlier_detection
    def _neighborhood_based_inliers(self, inds, cdeltas, zscore, verbose=False):
        if self.inlier_neighbors_cmp_deltas:
            # optionally use the other inliers in the neighborhood with the closest deltas.
            cmp_nbrs = NearestNeighbors(n_neighbors=self.ninlier_neighhbors_cmp[0],
                    algorithm='kd_tree').fit(cdeltas[inds,:])
            inds = inds[cmp_nbrs.kneighbors(cdeltas[inds[0],:].reshape((1,2)), return_distance=False).reshape(-1)]
        else:
            # default to use the closest other inliers in the neighborhood.
            inds = inds[:self.ninlier_neighhbors_cmp[0]]

        vecs = cdeltas[inds,:]
        # do not fall for changing this again, comparing stats vs mean/median delta
        #   for the selected neighborhood points works the best.
        vecs_minus_med = vecs - np.median(vecs, axis=0)
        vecs_minus_med_mag = np.sqrt(((vecs_minus_med)**2).sum(1))

        if zscore < 0:
            # interpret this as a residual value in pixels, not a zscore.
            sel_ok_point = (vecs_minus_med_mag[0] < -zscore)
            sel_ok_nbhd = (vecs_minus_med_mag[1:].mean() < -zscore)
            #if verbose and not (sel_ok_point and sel_ok_nbhd):
            #    print(inds[0], vecs, sel_ok_point, sel_ok_nbhd)
        else:
            vecs_minus_med_ang = np.arctan2(vecs_minus_med[:,1], vecs_minus_med[:,0])
            # z-score of distance of vectors from median vector
            zd = mad_zscore(vecs_minus_med_mag)
            # z-score of angles of vectors from median vector
            za = mad_angle_zscore(vecs_minus_med_ang)
            # we want both the z-score of the current outlier,
            #   and the mean z-score of the neighborhood to be low.
            sel_ok_point = np.logical_and(np.abs(zd[0]) < zscore, np.abs(za[0]) < zscore)
            sel_ok_nbhd = np.logical_and(np.abs(zd).mean() < zscore, np.abs(za).mean() < zscore)

        return (sel_ok_point and sel_ok_nbhd)
    #def _neighborhood_based_inliers

    def interpolate_fine_outliers(self, interp_inliers=0., inlier_nneighbors=0, nworkers=1, doplots=False):
        if self.wafer_aggregator_verbose:
            print('Outlier interpolation fine, ngrid points {}, img range {}-{}'.format(self.ngrid,
                self.img_range[0],self.img_range[1]))
            t = time.time()

        # setup for blockwise processing
        if self.single_block:
            grid_pts = self.grid_locations_pixels
            ngrid = self.ngrid
        else:
            grid_pts = self.blk_grid_pts
            ngrid = self.blk_ngrid
            if self.wafer_aggregator_verbose:
                print('\tiblock {} {} of {} {}, ngrid points block {}'.format(self.iblock[0], self.iblock[1],
                    self.nblocks[0], self.nblocks[1], ngrid))

        # old unrolled method for reference, very memory wasteful
        # self.fine_interp_deltas = np.empty((self.total_nimgs,self.nneighbors,ngrid,2), dtype=np.double)
        # self.fine_interp_deltas.fill(np.nan)

        self.fine_interp_deltas = [None]*self.total_nimgs
        self.fine_interp_weights = [None]*self.total_nimgs

        # outer loop over slices, outlier detection is done independently over slices and skips
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            self.fine_interp_deltas[i] = np.empty((self.nneighbors,ngrid,2), dtype=np.double)
            self.fine_interp_deltas[i].fill(np.nan)
            self.fine_interp_weights[i] = np.zeros((self.nneighbors,ngrid), dtype=np.double)

            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            i_wafer_id = self.wafer_ids[i_wafer_ind]
            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k #; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                j_wafer_id = self.wafer_ids[j_wafer_ind]

                if not self.fine_valid_comparison[i,ik]: continue

                if self.verbose_iterations:
                    print('\nProcessing total order ind %d' % (i,))
                    print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                          (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))

                cdeltas = self.deltas[i][ik,:,:]
                cxcorrs = self.xcorrs[i][ik,:]
                cfine_weights = self.fine_weights[i][ik,:]

                #sel_included = np.isfinite(cxcorrs) # converted to bool already in init_fine
                sel_included = cxcorrs
                nincluded = sel_included.sum()
                # # interpolate anything that is zero weight and included
                #sel_interp = np.logical_and(cfine_weights == 0, sel_included)
                # interpolate anything that is zero weight, with the rigid MLS interpolation
                #   and the tissues masks for exclusion, better to also "interpolate" excluded points.
                sel_interp = (cfine_weights == 0)
                #sel_deltas = (cfine_weights > 0)
                if interp_inliers >= 1:
                    cutoff = int(interp_inliers)
                else:
                    cutoff = np.round(nincluded * interp_inliers)

                # try to interpolate the existing grid points in order to replace
                #   all the three types of bad points (see above).
                # for block support, keep all the inliers for the whole slice.
                #   i.e., interpoate on the block, but use all the inliers on the whole slice as the control points.
                #p = grid_pts[sel_deltas,:]; v = cdeltas[sel_deltas,:]
                p = self.grid_locations_pixels[self.sel_inliers[i][ik,:],:]
                v = self.deltas_inliers[i][ik]
                ip = grid_pts[sel_interp,:]; npts = p.shape[0]
                deltas = np.empty((ngrid,2), dtype=np.double); deltas.fill(np.nan)
                if npts < cutoff:
                    if self.verbose_iterations:
                        print('Not interpolating, npoints {} (of {}) < cutoff {}'.format(npts, nincluded, cutoff))
                elif ip.shape[0] < 1:
                    if self.verbose_iterations:
                        print('No points to interpolate, all {} points are included inliers'.format(npts))
                else:
                    if self.verbose_iterations:
                        print('Interpolating based on {} points'.format(npts,))
                    # original method
                    # vx = interp.griddata(p, v[:,0], ip, fill_value=np.nan, method=self.region_interp_type_deltas)
                    # vy = interp.griddata(p, v[:,1], ip, fill_value=np.nan, method=self.region_interp_type_deltas)
                    # deltas = np.concatenate((vx[:,None],vy[:,None]), axis=1)
                    if self.delta_interp_method == delta_interp_methods.MLS:
                        # Interpolate using the moving least square method
                        if inlier_nneighbors == 0 or p.shape[0] <= inlier_nneighbors:
                            f_v = mls_rigid_transform(ip, p, p+v)
                        else:
                            f_v = self._mls_rigid_transform_nbhd(ip, p, p+v, inlier_nneighbors, nworkers)
                        deltas[sel_interp,:] = f_v - ip
                    elif self.delta_interp_method == delta_interp_methods.TPS:
                        # Interpolate using thin plate splines
                        # xxx - realized sklearn method with two dims independently is not sufficient
                        assert(False) # not implemented

                if doplots:
                    d = cdeltas.copy(); d[sel_interp,:] = deltas[sel_interp,:]
                    make_delta_plot(grid_pts, deltas=d,
                        grid_sel_r=sel_interp, grid_sel_b=np.logical_not(sel_included), figno=10)
                    plt.show()

                self.fine_interp_deltas[i][ik,:,:] = deltas
            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):

            sel_valid = np.isfinite(self.fine_interp_deltas[i]).all(2)
            # sanity check, both interpolated x/y should either be nan or not nan
            assert( np.logical_not(np.isfinite(self.fine_interp_deltas[i][np.logical_not(sel_valid),:])).all() )
            self.fine_interp_weights[i][sel_valid] = 1.
        #for i in range(self.img_range[0],self.img_range[1]):

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def interpolate_fine_outliers(self):

    def fine_deltas_to_rough_deltas(self):
        if self.wafer_aggregator_verbose:
            print('Fine inlier affine fits, ngrid points {}, img range {}-{}'.format(self.ngrid,
                self.img_range[0],self.img_range[1]))
            t = time.time()

        # inits affines and points
        self.forward_affines = [None]*self.nwafer_ids; self.reverse_affines = [None]*self.nwafer_ids
        self.rforward_affines = [None]*self.nwafer_ids; self.rreverse_affines = [None]*self.nwafer_ids
        self.forward_pts_src = [None]*self.nwafer_ids; self.reverse_pts_src = [None]*self.nwafer_ids
        self.forward_pts_dst = [None]*self.nwafer_ids; self.reverse_pts_dst = [None]*self.nwafer_ids
        for i in range(self.nwafer_ids):
            self.forward_affines[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.reverse_affines[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.rforward_affines[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.rreverse_affines[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.forward_pts_src[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.reverse_pts_src[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.forward_pts_dst[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]
            self.reverse_pts_dst[i] = [[None]*self.wafers_nimgs[i] for x in range(self.max_neighbor_range)]

        # select for current inlier points
        cinliers = np.empty((self.ngrid,), dtype=bool)
        #all_sel_excluded = np.logical_not(np.isfinite(self.xcorrs))

        # turn the fine delta inliers into affine transformations to be applied to each slice.
        # these can be exported in the same format as rough deltas
        #   and then applied using the rough alignment mechanisms.
        poly = preprocessing.PolynomialFeatures(degree=1)
        clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)
        clf_rigid = RigidRegression()

        # outer loop over slices, fits are done independently over slices and skips
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            i_wafer_id = self.wafer_ids[i_wafer_ind]
            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                j_wafer_id = self.wafer_ids[j_wafer_ind]

                if not self.fine_valid_comparison[i,ik]: continue

                if self.verbose_iterations:
                    print('\nProcessing total order ind %d' % (i,))
                    print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                          (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))

                # get the current source and dest points
                cinliers.fill(1)
                cinliers[self.all_fine_outliers[i][-1][ik]] = 0
                #cinliers[np.logical_not(np.isfinite(self.xcorrs[i][ik,:]))] = 0 # converted to bool in init_fine
                cinliers[np.logical_not(self.xcorrs[i][ik,:])] = 0
                assert(cinliers.sum() > 12) # not enough inlier grid points to fit affine
                # affines calculated on the points are the inverse of those for the images.
                cgrid_pts_dst = self.grid_locations_pixels[cinliers,:]
                cgrid_pts = cgrid_pts_dst + self.deltas[i][ik,cinliers,:]
                cgrid_pts_src = poly.fit_transform(cgrid_pts)

                # fit the affines
                clf.fit(cgrid_pts_src, cgrid_pts_dst)
                # scikit learn puts constant terms on the left, flip and augment
                caffine = clf.coef_
                caffine = np.concatenate( (np.concatenate( (caffine[:,1:], caffine[:,0][:,None]), axis=1 ),
                                           np.zeros((1,3), dtype=caffine.dtype)), axis=0 )
                caffine[2,2] = 1

                # also fit rigid affines, translation and rotation only
                clf_rigid.fit(cgrid_pts, cgrid_pts_dst)

                # "re-waferize" the outputs so they can be fed back into rough reconciler.
                if k < 0:
                    wafer_ind = j_wafer_ind; ind = j_ind
                else:
                    wafer_ind = i_wafer_ind; ind = i_ind
                # the alignments are calculated in the solver by dividing into groups modulo by skip amount.
                div_ind = ind // ak; mod_ind = ind % ak
                skip_ind = self.skip_cumlens[wafer_ind][ak-1][mod_ind] + div_ind
                if k < 0:
                    self.reverse_pts_src[wafer_ind][ak-1][skip_ind] = cgrid_pts
                    self.reverse_pts_dst[wafer_ind][ak-1][skip_ind] = cgrid_pts_dst
                    self.reverse_affines[wafer_ind][ak-1][skip_ind] = caffine
                    self.rreverse_affines[wafer_ind][ak-1][skip_ind] = clf_rigid.coef_.copy()
                else:
                    self.forward_pts_src[wafer_ind][ak-1][skip_ind] = cgrid_pts
                    self.forward_pts_dst[wafer_ind][ak-1][skip_ind] = cgrid_pts_dst
                    self.forward_affines[wafer_ind][ak-1][skip_ind] = caffine
                    self.rforward_affines[wafer_ind][ak-1][skip_ind] = clf_rigid.coef_.copy()

            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
        #for i in range(self.img_range[0],self.img_range[1]):

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def fine_deltas_to_rough_deltas(self):


    def reconcile_fine_alignments(self, L1_norm=0., L2_norm=0., min_valid_slice_comparisons=0, regr_bias=False,
            idw_p=2., solve_reverse=False, neighbor_dist_scale=None, neighbors2D_radius_pixels=0.,
            neighbors2D_expected=0, neighbors2D_std_pixels=np.inf, neighbors2D_W=0., z_neighbors_radius=0,
            nworkers=1, iprocess=0, nprocesses=1):
        if self.wafer_aggregator_verbose:
            print('Reconcile fine, ngrid points {}, L1 norm {}, L2 norm {}, iprocess {}, nprocesses {}'.\
                format(self.ngrid, L1_norm, L2_norm, iprocess, nprocesses))

        weight_max = self.fine_weights_maxes.max()
        assert(weight_max > 0) # all the fine weights are zero?!?!?
        for i in range(self.img_range[0],self.img_range[1]):
            if self.deltas[i] is None: continue

            # normalize weights to [0,1] so that L1 norm values are relative to max weight of 1
            self.fine_weights[i] = self.fine_weights[i] / weight_max

            # option to scale the weights as a function of the comparison distance in the slice ordering.
            if neighbor_dist_scale is not None:
                assert(neighbor_dist_scale[0] == 1 and all(x <= 1 for x in neighbor_dist_scale))
                for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                    ak = abs(k)
                    self.fine_weights[i][ik,:] = self.fine_weights[i][ik,:] * neighbor_dist_scale[ak-1]

            # set the deltas to zero for all zero weights (outliers and excluded points).
            sel = (self.fine_weights[i] == 0); self.deltas[i][sel, :] = 0

        # run the solver on all the deltas.
        # The scale factor of -1 inverts the deltas before the solver step
        #   Purpose is to compute the dst->src instead of src->dst coordinate remap.
        scale = -1. if solve_reverse else 1.
        solved_deltas, comps_sel, inds = self._reconcile_grid_deltas(self.deltas, self.fine_weights,
                self.fine_valid_comparison, nworkers=nworkers, L1_norm=L1_norm, L2_norm=L2_norm, idw_p=idw_p,
                scale=scale, min_adj=min_valid_slice_comparisons, iprocess=iprocess, nprocesses=nprocesses,
                neighbors_radius_pixels=neighbors2D_radius_pixels, neighbors_std_pixels=neighbors2D_std_pixels,
                neighbors_W=neighbors2D_W, neighbors_expected=neighbors2D_expected, regr_bias=regr_bias,
                z_nnbrs_rad=z_neighbors_radius)

        # for single process, these are the solved deltas for the whole dataset in solved order.
        # for multiple processes, these are only the currently solved deltas. they are put into
        #   solved order as part of the merge processes using inds.
        self.cum_deltas = solved_deltas
        self.cum_comps_sel = comps_sel
        self.cum_deltas_inds = inds

    # this function is broken off of reconcile_fine_alignments
    #   in order to support parallelization of the delta reconciler.
    def waferize_reconcile_fine_alignments(self):
        # save the deltas by wafer in imaged order.
        self.imaged_order_deltas = [None]*self.nwafer_ids
        #self.imaged_order_reverse_deltas = [None]*self.nwafer_ids
        for i in range(self.nwafer_ids):
            # old method, roll the deltas into python list
            #self.imaged_order_deltas[i] = [None]*self.wafers_nregions[i]
            # new method, re-order the existing numpy array
            self.imaged_order_deltas[i] = self.cum_deltas[self.region_ind_to_solved_order[i],:,:]
            self.imaged_order_deltas[i][self.missing_region_inds[i]] = np.nan

        # old method, roll the deltas into python list
        # #for i in range(self.total_nimgs):
        # for i in range(self.order_rng[0],self.order_rng[1]):
        #     i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
        #     i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
        #     i_slice = self.solved_orders[i_wafer_ind][i_ind]
        #     #i_wafer_id = self.wafer_ids[i_wafer_ind]
        #
        #     self.imaged_order_deltas[i_wafer_ind][i_slice] = self.cum_deltas[i,:,:]

    def fine_deltas_affine_filter(self, shape_pixels, affine_degree=1, use_interp_points=False,
            affine_interpolation=False, output_features_scale=None, doplots=False):
        assert( not use_interp_points or not affine_interpolation ) # do not use these features together
        if self.wafer_aggregator_verbose:
            print('{} fine deltas with affine filter, ngrid points {}, img range {}-{}'.format(\
                'Interpolating' if affine_interpolation else 'Filtering', self.ngrid,
                self.img_range[0],self.img_range[1]))
            print('Filter pixel shape {} x {}'.format(shape_pixels[0], shape_pixels[1]))
            t = time.time()

        # this is a scale on the number of output features used a a threshold to not fit
        #   if there are less inliers than this in the filtered area.
        if output_features_scale is None:
            # default to 2 which scales the number of output features by ndims,
            #   so is equal to the number of parameters that are being fit.
            output_features_scale = 2.

        # setup for blockwise processing
        if self.single_block:
            grid_pts = self.grid_locations_pixels
            ngrid = self.ngrid
        else:
            grid_pts = self.blk_grid_pts
            ngrid = self.blk_ngrid
            if self.wafer_aggregator_verbose:
                print('\tiblock {} {} of {} {}, ngrid points block {}'.format(self.iblock[0], self.iblock[1],
                    self.nblocks[0], self.nblocks[1], ngrid))

        if not use_interp_points:
            # select for current inlier points
            cinliers = np.empty((ngrid,), dtype=bool)

        if affine_interpolation:
            self.fine_interp_deltas = [None]*self.total_nimgs
            self.fine_interp_weights = [None]*self.total_nimgs
        else:
            # output of this function, the "affine-filtered" deltas
            self.filtered_deltas = np.zeros((self.total_nimgs,self.nneighbors,ngrid,2), dtype=np.double)

        # iterate the fine deltas and fit local regions with affines.
        # use the local affine fitted deltas as the new delta at each point, an "affine filter"
        poly = preprocessing.PolynomialFeatures(degree=affine_degree)
        poly.fit_transform(np.random.rand(3,2)) # just so features are populated for 2D
        clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)

        # outer loop over slices, fits are done independently over slices and skips
        #for i in range(self.total_nimgs):
        for i in range(self.img_range[0],self.img_range[1]):
            if affine_interpolation:
                self.fine_interp_deltas[i] = np.empty((self.nneighbors,ngrid,2), dtype=np.double)
                self.fine_interp_deltas[i].fill(np.nan)
                self.fine_interp_weights[i] = np.zeros((self.nneighbors,ngrid), dtype=np.double)

            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice_str = self.region_strs[i_wafer_ind][self.solved_orders[i_wafer_ind][i_ind]]
            i_wafer_id = self.wafer_ids[i_wafer_ind]
            # inner loop over number of neighboring slices to use
            for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):
                j = i+k #; ak = abs(k)
                if j < 0 or j >= self.total_nimgs: continue
                j_wafer_ind = np.nonzero(j / self.cum_wafers_nimgs[1:] < 1.)[0][0]
                j_ind = (j - self.cum_wafers_nimgs[j_wafer_ind]) % self.wafers_nimgs[j_wafer_ind]
                j_slice_str = self.region_strs[j_wafer_ind][self.solved_orders[j_wafer_ind][j_ind]]
                j_wafer_id = self.wafer_ids[j_wafer_ind]

                if not self.fine_valid_comparison[i,ik]: continue

                if self.verbose_iterations:
                    print('\nProcessing total order ind %d' % (i,))
                    print('\tind %d slice %s wafer %d, compare to offset %d, ind %d slice %s wafer %d' % \
                          (i_ind, i_slice_str, i_wafer_id, k, j_ind, j_slice_str, j_wafer_id))

                if use_interp_points:
                    cinliers = (self.fine_weights[i][ik,:] > 0)
                else:
                    # get the current source and dest points
                    cinliers.fill(1)
                    cinliers[self.all_fine_outliers[i][-1][ik]] = 0
                    #cexcluded = np.logical_not(np.isfinite(self.xcorrs[i][ik,:])) # converted to bool in init_fine
                    cexcluded = np.logical_not(self.xcorrs[i][ik,:])
                    cinliers[cexcluded] = 0

                for g in range(ngrid):
                    if affine_interpolation:
                        # for interpolation mode, skip points outside polygon.
                        # "partially process" inliers, but only to remove inliers with very few inlier
                        #   deltas within the affine filter shape (shape_pixels).
                        if cexcluded[g]: continue
                    else:
                        # do not filter outlier points in normal mode, but fill in in interpolation mode
                        if not cinliers[g]: continue
                    sel_pts = np.logical_and(\
                            grid_pts >= grid_pts[g,:]-shape_pixels/2,
                            grid_pts <= grid_pts[g,:]+shape_pixels/2).all(1)
                    sel_pts = np.logical_and(sel_pts, cinliers)

                    # if there are not enough points to fit the affine
                    if sel_pts.sum() < int(poly.n_output_features_*output_features_scale):
                        if affine_interpolation:
                            if cinliers[g]:
                                # xxx - hacky, when loaded for aggregation, allows for a "second-round"
                                #   of outlier removal. mostly added this here to avoid having to rerun
                                #   a long outlier detection step for the ultrafine alignment.
                                self.fine_interp_weights[i][ik,g] = -1.
                                #print('WARNING: at grid point {}, only {} inliers'.format(g,sel_pts.sum()))
                        else:
                            print('WARNING: at grid point {}, only {} inliers'.format(g,sel_pts.sum()))
                            # better to leave the delta at zero, or to copy the original?
                            #self.filtered_deltas[i,ik,g,:] = self.deltas[i,ik,g,:]
                            assert(False) # with MLS interp should not happen
                        continue
                    # no further processing in interpolation mode for inlier points
                    if affine_interpolation and cinliers[g]: continue

                    # fit the affine and use it to estimate the current point ("affine-filter")
                    cgrid_pts = grid_pts[sel_pts,:]
                    cgrid_pts_dst = cgrid_pts + self.deltas[i][ik,sel_pts,:]
                    cgrid_pts_src = poly.fit_transform(cgrid_pts)

                    clf.fit(cgrid_pts_src, cgrid_pts_dst)
                    fit_pt = clf.predict(poly.fit_transform(grid_pts[g,:][None,:]))
                    if affine_interpolation:
                        # replace with the interpolated points
                        self.fine_interp_deltas[i][ik,g,:] = fit_pt - grid_pts[g,:]
                    else:
                        self.filtered_deltas[i,ik,g,:] = fit_pt - grid_pts[g,:]
                #for g in range(ngrid):

                if doplots:
                    d = self.deltas[i][ik,:,:].copy()
                    sel = (self.fine_weights[i][ik,:] == 0); d[sel, :] = 0
                    make_delta_plot(grid_pts, deltas=d, figno=10)
                    d = self.filtered_deltas[i,ik,:,:].copy()
                    make_delta_plot(grid_pts, deltas=d, figno=11)
                    plt.show()
            #for k,ik in zip(self.neighbor_rng, range(self.nneighbors)):

            if affine_interpolation:
                sel_valid = np.isfinite(self.fine_interp_deltas[i]).all(2)
                # sanity check, both interpolated x/y should either be nan or not nan
                assert( np.logical_not(np.isfinite(self.fine_interp_deltas[i][np.logical_not(sel_valid),:])).all() )
                self.fine_interp_weights[i][sel_valid] = 1.
        #for i in range(self.img_range[0],self.img_range[1]):

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def fine_deltas_affine_filter(self):

    def solved_deltas_affine_filter(self, shape_pixels, nworkers=1, affine_degree=1):
        if self.wafer_aggregator_verbose:
            print('Filtering solved deltas with affine filter, ngrid points {}, img range {}-{}'.format(self.ngrid,
                self.img_range[0],self.img_range[1]))
            print('Filter pixel shape {} x {}'.format(shape_pixels[0], shape_pixels[1]))
            t = time.time()

        # select of the deltas that were actually solved and not zeroed out stored in this select.
        inliers = self.cum_comps_sel

        # inits
        nimgs = self.img_range[1] - self.img_range[0]
        workers = [None]*nworkers
        result_queue = mp.Queue(nimgs)
        #inds = np.random.permutation(nimgs) # NO, slicing is used below, also no need
        inds = np.arange(nimgs)
        inds = np.array_split(inds, nworkers)

        # iterating over the grid points for each image (the filtering aspect) is a bit slow,
        #   so parallelize by workers. could be externally parallelized by process using self.img_range,
        #   as was done for the outlier detection for example.
        # internally sklearn can still parallize the fits, use self.nthreads for this.
        for i in range(nworkers):
            workers[i] = mp.Process(target=filter_solved_deltas_job, daemon=True,
                    args=(i, inds[i], inliers[inds[i][0]:inds[i][-1]+1,:],
                        self.cum_deltas[inds[i][0]:inds[i][-1]+1,:,:], self.grid_locations_pixels, shape_pixels,
                        affine_degree, self.nthreads, result_queue, self.wafer_aggregator_verbose))
            workers[i].start()
        # NOTE: only call join after queue is emptied
        # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

        # output of this function, the "affine-filtered" solved (cum_)deltas
        filtered_deltas = np.zeros((self.total_nimgs,self.ngrid,2), dtype=np.double)

        # collect the worker results.
        nprint = 200
        dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for i in range(nimgs):
        i = 0
        while i < nimgs:
            if self.wafer_aggregator_verbose and i>0 and i%nprint==0:
                print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
                print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            filtered_deltas[res['ind'],:,:] = res['filtered_deltas']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        # store the filtered deltas back
        self.cum_deltas = filtered_deltas

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def solved_deltas_affine_filter(self):

    def fine_deltas_to_rough_affines(self, affine_degree=1):
        if self.wafer_aggregator_verbose:
            print('Affine fitting solved fine deltas, ngrid points {}, img range {}-{}'.format(self.ngrid,
                self.img_range[0],self.img_range[1]))
            t = time.time()

        # inits
        if self.wafers_imaged_order_rough_affines is None:
            self.wafers_imaged_order_rough_affines = [None]*self.nwafer_ids
            for i in range(self.nwafer_ids):
                self.wafers_imaged_order_rough_affines[i] = [None]*self.wafers_nregions[i]
        self.cum_affines = [None]*self.total_nimgs
        poly = preprocessing.PolynomialFeatures(degree=affine_degree)
        poly.fit_transform(np.random.rand(3,2)) # just so features are populated for 2D
        clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)

        for i in range(self.img_range[0],self.img_range[1]):
            i_wafer_ind = np.nonzero(i / self.cum_wafers_nimgs[1:] < 1.)[0][0]
            i_ind = (i - self.cum_wafers_nimgs[i_wafer_ind]) % self.wafers_nimgs[i_wafer_ind]
            i_slice = self.solved_orders[i_wafer_ind][i_ind]
            #i_wafer_id = self.wafer_ids[i_wafer_ind]

            cinliers = self.cum_comps_sel[i,:]
            #assert(cinliers.sum() > poly.n_output_features_+1) # not enough inlier grid points
            assert(cinliers.sum() > 2*poly.n_output_features_) # not enough inlier grid points

            # affines calculated on the points are the inverse of those for the images.
            cgrid_pts_dst = self.grid_locations_pixels[cinliers,:]
            cgrid_pts = cgrid_pts_dst + self.cum_deltas[i,cinliers,:]
            cgrid_pts_src = poly.fit_transform(cgrid_pts)

            # fit the affines
            clf.fit(cgrid_pts_src, cgrid_pts_dst)
            # scikit learn puts constant terms on the left, flip and augment
            caffine = clf.coef_
            caffine = np.concatenate( (np.concatenate( (caffine[:,1:], caffine[:,0][:,None]), axis=1 ),
                                       np.zeros((1,3), dtype=caffine.dtype)), axis=0 )
            caffine[2,2] = 1

            # save current affine for the wafer and over all wafers.
            A = self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice]
            if A is None:
                self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice] = caffine
                self.cum_affines[i] = caffine
            else:
                # apply the affine on top of an existing affine
                B = np.dot(caffine, A)
                self.wafers_imaged_order_rough_affines[i_wafer_ind][i_slice] = B
                self.cum_affines[i] = B

        #for i in range(self.img_range[0],self.img_range[1]):

        if self.wafer_aggregator_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def fine_deltas_to_rough_affines(self):

    # new fine alignment "reconciler" >>>
