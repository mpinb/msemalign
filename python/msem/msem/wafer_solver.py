"""wafer_solver.py

Helper class for wafer class that solves slice ordering and gross rotation
  and translation amounts (using further downsampled thumbnails of regions).
These are the components of the "rough alignment", order solving and computing
  rigid or affine transformations for each slice, imporoving dramatically
  on the "microscope" alignment, but without applying any elastic warping.

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

import os
import sys
import time
#import warnings

# xxx - really did not want to add the loading capability inside of any of the core msem package
#   but needed a way to only have to load a subset of keypoints, because for large wafers they
#   might not all load into memory at once. this works but is not modular, so a better idea is in order.
import dill

import numpy as np
import scipy.sparse as sp
import scipy.linalg as lin
import scipy.ndimage as nd
import scipy.spatial.distance as scidist
import cv2
import skimage.measure as measure
from skimage import exposure, img_as_ubyte
import math

import tifffile

# from sklearnex import patch_sklearn
# patch_sklearn()
# from sklearn.neighbors import NearestNeighbors
# optimized sklearn important in this module so that
#   SIFT comparisons can be done "efficiently" on cpu
#   for rough alignment when gpus not available.
from sklearnex.neighbors import NearestNeighbors
from sklearn import preprocessing

try:
    import faiss
except:
    print('IMPORT: faiss unavailable, wafer_solver may not work')

from io import BytesIO
import tempfile
import subprocess

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.path import Path

from .utils import get_num_threads, PolyCentroid, nsd
from .zimages import zimages
from .AffineRANSACRegressor import AffineRANSACRegressor, _ransac_repeat
from .procrustes import RigidRegression_types

import multiprocessing as mp
import queue


# xxx - is there some way to get these programatically?
SIFT_descriptor_dtype = np.float32
SIFT_descriptor_ndims = 256

def compute_keypoints_job(nfeatures, nthreads_per_job, ind, inds, imgs, result_queue, polygons, masks, mask_ds,
        filter_size, rescale, min_features, verbose):
    if verbose: print('\tworker%d started' % (ind, ))
    cv2.setNumThreads(nthreads_per_job)
    # Initiate SIFT detector
    if nfeatures is not None:
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    else:
        sift = cv2.SIFT_create()

    nimgs = len(imgs) #; dt = time.time()
    for i in range(nimgs):
        #if verbose and i>0 and i%100==0:
        #    print('\tworker %d: %d of %d in %.4f s' % (ind,i+1,nimgs,time.time() - dt, )); dt = time.time()
        if inds[i] < 0:
            result = {'ind':-1, 'iworker':ind} # missing region
        else:
            pimg = imgs[i]

            # optional preprocessing before keypoints computed
            if filter_size > 0: pimg = nd.median_filter(pimg, size=filter_size)
            if rescale: pimg = exposure.rescale_intensity(pimg)

            # non-optional preprocessing, SIFT only works on 8 bit grayscale images (at least in cv2).
            if pimg.dtype != np.uint8: pimg = img_as_ubyte(pimg)

            # find the keypoints and descriptors with SIFT
            # do not add a try here, whatever the issue is, it should be dealt with before moving on.
            #   for example a blank image, an out-of-memory error, etc.
            keypoints, descriptors = sift.detectAndCompute(pimg,None)

            # normalizing makes l2 distance the same as cosine distance, xxx - this seems to give worse result
            #d = self.wafer_descriptors[i]; self.wafer_descriptors[i] = d / np.linalg.norm(d, axis=1)[:,None]

            mask = None
            if masks[i] is not None:
                ipts = np.round(np.array([x.pt for x in keypoints]) / mask_ds).astype(np.int64)
                # round can put some points over the max
                ipts[ipts[:,0] == masks[i].shape[1],0] = masks[i].shape[1]-1
                ipts[ipts[:,1] == masks[i].shape[0],1] = masks[i].shape[0]-1
                mask = [masks[i][x[1],x[0]] for x in ipts]
            if polygons[i] is not None:
                polygon = Path(polygons[i])
                pmask = polygon.contains_points(np.array([x.pt for x in keypoints]))
                mask = np.logical_and(mask,pmask) if mask is not None else pmask
            if mask is not None:
                # do not mask if it results in very few keypoints
                tmp = [x for x,m in zip(keypoints,mask) if m]
                if len(tmp) < min_features:
                    # try the polygon-only mask
                    tmp = [x for x,m in zip(keypoints,pmask) if m]
                    mask = pmask
                if len(tmp) >= min_features:
                    keypoints = tmp
                    descriptors = descriptors[mask,:]

            # https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
            # this is needed for saving to dill file, but also can not insert the keypoints into the
            #   result queue at all without this. the queue implementation also uses pickling.
            pickleable_keypoints = [(point.pt, point.size, point.angle, point.response, point.octave,
                                                      point.class_id) for point in keypoints]
            result = {'ind':inds[i], 'pickleable_keypoints':pickleable_keypoints, 'descriptors':descriptors,
                      'iworker':ind}
        result_queue.put(result)
    if verbose: print('\tworker%d completed' % (ind, ))


def compute_matches_job(gpu_index, ind, ntimgs, xinds, yinds, full, xdescriptors, ydescriptors,
        xkeypoints, ykeypoints, mask_inds, result_queue, max_npts_feature_correspondence, lowe_ratio,
        min_feature_matches, dsthumbnail, ransac, poly, ransac_repeats, nworkers, verbose):
    gpu_res = faiss.StandardGpuResources()      # use a single GPU
    if verbose: print('\tworker{}: on gpu {} started'.format(ind, gpu_index))
    nximgs = len(xdescriptors); nyimgs = len(ydescriptors)
    for x in range(nximgs):
        d = xdescriptors[x].shape[1]
        if mask_inds is None or mask_inds[xinds].any() or mask_inds[yinds].any():
            # xxx - since we did this they added some methods for the full search in faiss
            #   that supposedly can save some memory. it seemed overly complicated relative
            #   to this, so punted for now, but probably worth revisiting.
            index_cpu = faiss.IndexFlatL2(d)        # build the index
            index_cpu.add(xdescriptors[x])           # add vectors to the index
            index = faiss.index_cpu_to_gpu(gpu_res, gpu_index, index_cpu)

        for y in range(nyimgs):
            # diagonal is always left at zero (comparison of image to itself).
            # if we are not doing the full comparison, then skip any tril compares.
            if xinds[x] == yinds[y] or (not full and xinds[x] < yinds[y]): continue

            if mask_inds is None or mask_inds[xinds[x]] or mask_inds[yinds[y]]:
                k = 2 # nearest-neighbors
                distances, indices = index.search(ydescriptors[y], k)     # actual search
                sel = (distances[:,0] > 0); distances[np.logical_not(sel),0] = 1
                msk = np.logical_and(sel, distances[:,1] / distances[:,0] > lowe_ratio)

                # heuristic feature to reject points that mostly map to the same point
                if max_npts_feature_correspondence > 0:
                    cnt_nmsk = (np.bincount(indices[:,0]) > max_npts_feature_correspondence)
                    remove_msk = np.in1d(indices[:,0], np.nonzero(cnt_nmsk)[0])
                    msk = np.logical_and(msk, np.logical_not(remove_msk))

                msum = msk.sum()
            else:
                msum = None

            # threshold percent_matches at the specified minimum matching features level.
            if msum is not None and msum >= min_feature_matches:
                if xkeypoints is None:
                    # return percent matches purely as percent matching SIFT descriptors
                    result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':msum / msk.size, 'iworker':ind}
                else:
                    pypts_src = [x.pt for x in xkeypoints[x]]
                    pts_src = np.array(pypts_src)[indices[:,0],:]
                    pts_dst = np.array([x.pt for x in ykeypoints[y]])

                    pts_src = pts_src[msk,:] * dsthumbnail
                    pts_dst = pts_dst[msk,:] * dsthumbnail
                    Xpts = poly.fit_transform(pts_src) if poly is not None else pts_src
                    #def _ransac_repeat(Xpts, ypts, ransac, ransac_repeats, verbose=False, nworkers=1):
                    _, cmask = _ransac_repeat(Xpts, pts_dst, ransac, ransac_repeats,
                            verbose=verbose, nworkers=nworkers)

                    if cmask is not None:
                        # return percent matches as percent of ransac inliers taken from matching SIFT descriptors
                        result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':cmask.sum() / msk.size,
                                'iworker':ind}
                    else:
                        result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':0., 'iworker':ind}
            else:
                result = {'indx':xinds[x], 'indy':yinds[y], 'percent_match':0., 'iworker':ind}

            result_queue.put(result)
        #for y in range(nyimgs):
    #for x in range(nximgs):
    if verbose: print('\tworker{}: on gpu {} completed'.format(ind, gpu_index))


class wafer_solver(zimages):
    """msem wafer helper class.

    Solving slice (region) order from a wafer of unordered but sequential slices.

    .. note::


    """

    ### fixed parameters not exposed

    #CONCORDE_EXE = os.path.expanduser('~/projects/concorde_tsp/concorde/build/TSP/concorde')
    CONCORDE_EXE = os.path.expanduser('~/gits/concorde_tsp/concorde/build/TSP/concorde')

    # so that this can be consistent across whole msem package using env variable.
    nthreads = get_num_threads()

    #knn_search_method = 'sklearn' # painfully slow brute force
    #knn_search_method = 'faiss-cpu' # only tried brute force, not much point in using this
    knn_search_method = 'faiss-gpu' # key to the game!

    # whether it's safe to assume that all the input images are the same size or not
    images_all_same_size = True


    def __init__(self, wafer, lowe_ratio=1.5, min_feature_matches=3, min_fit_pts_radial_std_um=0,
            thumbnail_subfolders='', thumbnail_suffix='.tiff', solved_order=None, dsthumbnail=1,
            max_npts_feature_correspondence=0, max_fit_translation_um=None, roi_polygon_scales=[0.],
            residual_threshold_um=1., rigid_type=RigidRegression_types.affine, ransac_repeats=1, ransac_max=1000,
            keypoints_dill_fns=None, keypoints_nworkers_per_process=1, keypoints_nprocesses=1, verbose=False):

        zimages.__init__(self)
        # xxx - just completely disable the cache'ing for now, maybe not super useful?
        #   doing it here as kindof a hacky way to still allow for the "cache clearing" in zimages init
        self.cache_dir = ''

        # wafer solver is only meant to solve order for one wafer at a time,
        #   OR must be exactly two regions for the cross-wafer rough alignments,
        assert(wafer.nwafer_ids==1 or wafer.nregions == 2)
        self.wafer = wafer
        self.wafer_solver_verbose = verbose

        self.lowe_ratio = lowe_ratio
        self.min_feature_matches = min_feature_matches

        # subfolder underneath the main processed image folder where the thumbnails are saved
        self.thumbnail_subfolders = thumbnail_subfolders
        self.thumbnail_subfolder = self.thumbnail_subfolders[0]
        self.thumbnail_mask_subfolder = self.thumbnail_subfolders[1]

        # suffix to use for loading the downsampled thumbnails of the montaged region images
        self.thumbnail_suffix = thumbnail_suffix

        # downsampling amount of the thumbnails relative to the original montaged images exported by run_region.py
        self.dsthumbnail = dsthumbnail

        # NOTE: do not apply dsthumbnail here, because the xforms are computed in
        #   the resolution that was exported by run_regions.py
        #dsscl = 1e3 / (self.wafer.scale_nm*self.wafer.dsstep*self.dsthumbnail) # NO!
        dsscl = 1e3 / (self.wafer.scale_nm*self.wafer.dsstep)

        # ransac parameter specifying threshold in pixels for inlier classification for affine fits.
        self.residual_threshold = residual_threshold_um * dsscl

        # heuristic for rejecting ransac fits, point spread
        self.min_fit_pts_radial_std = min_fit_pts_radial_std_um * dsscl

        # heuristic for rejecting ransac fits, max translation
        if max_fit_translation_um is None:
            self.max_fit_translation = np.array([np.inf, np.inf])
        else:
            self.max_fit_translation = np.array(max_fit_translation_um) * dsscl

        # default is only for rigidbody rotation rough alignment. specify this to calculate full affine xform.
        self.rigid_type = rigid_type

        # this is a method to keep larger context around the thumbnails, but remove the sift keypoints
        #   from these context areas so they do not influence the affine fits.
        # scale indicates the factor by which to expand or shrink the roi polygon.
        # points outside the scaled polygon are removed.
        # zero means disable this feature.
        # multiple values are for compute_wafer_alignments. if a bad_match occurs, use the next scale.
        self.roi_polygon_scales = sorted([float(x) for x in roi_polygon_scales], reverse=True)
        self.roi_polygon_nscales = len(self.roi_polygon_scales)
        assert(self.roi_polygon_nscales > 0)

        # specify the correct ordering of the slices (regions)
        self.solved_order = solved_order

        # heuristic feature to remove sift correspondence points that exceed this threshold of many-to-one mappings.
        # disable feature by setting to zero.
        self.max_npts_feature_correspondence = max_npts_feature_correspondence

        # number of times to repeat ransac fit, parallized by workers
        self.ransac_repeats = ransac_repeats

        # number of individual ransac max iterations
        self.ransac_max = ransac_max

        # this is to support incremental loading of keypoints incase they can not all fit in memory at once.
        self.keypoints_dill_fns = keypoints_dill_fns
        assert( keypoints_dill_fns is None or len(keypoints_dill_fns) == self.wafer.nwafer_ids )
        self.keypoints_nworkers_per_process = keypoints_nworkers_per_process
        self.keypoints_nprocesses = keypoints_nprocesses
        self.keypoints_nworkers_total = keypoints_nprocesses*keypoints_nworkers_per_process
        if self.keypoints_dill_fns is not None:
            # this specifies the on-demand or incremental loading mode
            self.keypoints_loaded_inds = [[np.zeros((0,), dtype=np.int64) for x in range(keypoints_nprocesses)] \
                    for x in range(self.wafer.nwafer_ids)]
        self.wafer_pickleable_keypoints = [None]*self.wafer.nregions
        self.wafer_descriptors = [None]*self.wafer.nregions
        self.wafer_keypoints = [None]*self.wafer.nregions

        self.wafer_images = None
        self.percent_matches = None


    # use concorde tsp solver to find shortest hamiltonian path through the region correlation distances
    # http://www.math.uwaterloo.ca/tsp/concorde/DOC/README.html
    # https://www.math.uwaterloo.ca/~bico/qsopt/beta/index.html
    # http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    # if there are multiple tours with the same distance, concorde seems to not be deterministic.
    #   it might return a different optimal tour on subsequent calls.
    @staticmethod
    def normxcorr_tsp_solver(Cin, endpoints_first=False, iendpoints=None):
        # NOTE: only the triu elements of Cin are used. set tril elements to zero incase triu was not passed.
        Cin = sp.triu(Cin,1) if sp.issparse(Cin) else np.triu(Cin,1)
        n = Cin.shape[0]; assert( n == Cin.shape[1] ) # need triu weighted adjacency matrix

        if iendpoints is not None: endpoints = iendpoints
        if endpoints_first and iendpoints is None:
            # take as endpoints the two slices with the smallest second correlations.
            C2m = Cin + Cin.T # symmetric corrleation distance using average between directional comparisons
            C2 = np.zeros((n,), dtype=np.double)
            for i in range(n):
                tmp = C2m[i,:].todense().A1 if sp.issparse(C2m) else C2m[i,:]
                C2[i] = np.sort(tmp.reshape(-1))[-2]
            endpoints = np.argsort(C2)[:2]
            C2m, C2, tmp = None, None, None

        # data-type and min/max values to use for actual correlations.
        # these are only dependent on concorde so did not see any need to parameterize.
        # need to keep large values to indicate no connection (for sparse graph).
        # need to keep small values to connect endpoints together.
        crd_dtype = np.int16
        crd_conn = 0 # forced connection (i.e., try to force route through this edge)
        crd_noconn = np.iinfo(crd_dtype).max # no connection (i.e., no edge exists in the graph)
        # xxx - parameterize min / max relative forced / no connection values?
        crd_min = 512 # non-zero, to distinguish from forced connection
        crd_max = crd_noconn - 2560 # non-max, to distinguish from no connection
        crd_rng = crd_max - crd_min

        # normalize, invert and convert correlations to integer type supported by concorde.
        # xxx - could not figure out how to use sparse distance matrix (edge list section) with concorde.
        #   so if the correlation input is sparse, convert to dense.
        #   sparse data or zero data (for dense input) are then treated as "no connection"
        Cd = Cin.toarray() if sp.issparse(Cin) else Cin.copy()
        sel = (Cd > 0)
        Cd[sel] -= Cd[sel].min(); Cd[sel] /= Cd[sel].max()
        Cd[sel] = 1 - Cd[sel] # convert from "correlation" to distance
        Cd[sel] = (Cd[sel]*crd_rng + crd_min)
        Cd = Cd.astype(crd_dtype)
        Cd[Cd == 0] = crd_noconn

        if endpoints_first or iendpoints is not None:
            # add a small-weight connection between the endpoints to create a circuit
            if endpoints[1] > endpoints[0]:
                Cd[endpoints[0], endpoints[1]] = crd_conn
            else:
                Cd[endpoints[1], endpoints[0]] = crd_conn
        iu = np.triu_indices(n, 1); W = np.asarray(Cd[iu]).reshape(-1)

        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = '/home/pwatkins/Downloads'
            tmpfn = os.path.join(tmpdirname, 'tmp.tsp')
            tmpfh = open(tmpfn, 'w')

            header = 'NAME: tmp\nTYPE: TSP\nCOMMENT: tmp\nDIMENSION: %d\n' % (n)
            # xxx - could not figure out how to use sparse distance matrix (edge list section) with concorde
            #header += 'EDGE_DATA_FORMAT: EDGE_LIST\n
            header += 'EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: UPPER_ROW'
            print(header, file=tmpfh)

            # xxx - could not figure out how to use sparse distance matrix (edge list section) with concorde
            #fh = BytesIO()
            #np.savetxt(fh, np.transpose(C.nonzero()),
            #    fmt='%d %d', delimiter='', newline='\n', header='EDGE_DATA_SECTION', footer='', comments='')
            #cstr = fh.getvalue(); fh.close(); print(cstr.decode('UTF-8')[:-1], file=tmpfh)
            #print('-1', file=tmpfh)

            llen = 10
            wlen = (W.size // llen) * llen; rlen = W.size % llen
            print('EDGE_WEIGHT_SECTION', file=tmpfh)
            if wlen > 0:
                fh = BytesIO()
                np.savetxt(fh, W[:wlen].reshape((-1,llen)),
                    fmt='%d ', delimiter='', newline='\n', header='', footer='', comments='')
                cstr = fh.getvalue(); fh.close(); print(cstr.decode('UTF-8')[:-1], file=tmpfh)
            if rlen > 0:
                fh = BytesIO()
                np.savetxt(fh, W[-rlen:][None,:],
                    fmt='%d ', delimiter='', newline='\n', header='', footer='', comments='')
                cstr = fh.getvalue(); fh.close(); print(cstr.decode('UTF-8')[:-1], file=tmpfh)

            tmpfh.close()
            tmpout = os.path.join(tmpdirname, 'tmp.sol')
            subprocess.call([wafer_solver.CONCORDE_EXE, '-o' + tmpout, tmpfn], cwd=tmpdirname,
                            stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT, close_fds=True)

            with open(tmpout, 'r') as tmpfh:
                tour = np.fromstring(tmpfh.read(), dtype=np.int64, sep=' ')[1:]

        # xxx - did not enumerate other failure conditions when testing previously, what happens if no path is found?
        assert(np.unique(tour).size == n) # concorde failed

        if not endpoints_first and iendpoints is None:
            # take as the endpoints the "weakest link"
            Cd2 = Cd + Cd.T
            tour_dist = Cd2[tour, np.roll(tour,-1)]
            ind = np.argmax(tour_dist)
            # also check distance from first to last point
            endpoints = tour[ind:ind+2] if tour_dist[ind] > Cd2[tour[0], tour[-1]] else [tour[0], tour[-1]]

        # shift the endpoints to the end positions
        i = np.nonzero(tour == endpoints[0])[0][0]; j = np.nonzero(tour == endpoints[1])[0][0]
        if not ((i==0 and j==n-1) or (j==0 and i==n-1)):
            tour = np.roll(tour, -i) if i > j else np.roll(tour, -j)
        # if endpoints were specified, enforce the beg/end ordering
        if iendpoints is not None and tour[0] != endpoints[0]:
            tour = tour[::-1]

        return tour, endpoints

    # piggyback on the tsp solver glue logic above to directly support regular TSP,
    #   i.e., the shortest route between points.
    @staticmethod
    def distance_tsp_solver(pts, iendpoints=None, corner_endpoints=False, topn_cols=0):
        print('Getting distance matrix'); t = time.time()
        pd = scidist.pdist(pts); maxd = pd.max(); pd = scidist.squareform(pd)
        print('\tdone in %.4f s' % (time.time() - t, ))
        minc = pts.min(0); maxc = pts.max(0)
        if iendpoints is None and corner_endpoints:
            # max the endpoints the points closest to the min/max of the points bounding box
            iendpoints = [np.argmin(((pts - minc)**2).sum(1)), np.argmin(((pts - maxc)**2).sum(1))]
        # convert to "correlations" expected by normxcorr_tsp_solver
        pd = 1 - (pd / maxd / 2)
        pd, _ = wafer_solver.preprocess_percent_matches(pd, topn_cols=topn_cols)
        print('Computing shortest path through slices'); t = time.time()
        tour, endpoints = wafer_solver.normxcorr_tsp_solver(pd, iendpoints=iendpoints)
        print('\tdone in %.4f s' % (time.time() - t, ))
        return tour, endpoints


    def load_wafer_images(self, unload=False, rng=None):
        if self.wafer_solver_verbose:
            print('Loading images for wafer number %d' % (self.wafer.wafer_ids[0],))
            if rng is not None:
                print('\tloading subset from {}-{}'.format(rng[0],rng[-1]))
            t = time.time()

        nregions = self.wafer.nregions
        self.wafer_images = [None]*nregions; cnt = 0
        self.wafer_images_size = np.zeros((nregions,2), dtype=np.int64)
        self.wafer_nimages = nregions
        self.wafer_roi_points = [None]*nregions
        self.wafer_roi_points_scaled = [[None]*nregions for x in range(self.roi_polygon_nscales)]
        self.wafer_tissue_masks = [None]*nregions
        images_size = None
        urng = range(nregions) if rng is None else rng
        for i in urng:
            c = self.wafer
            ind = i if c.is_cross_wafer else 0
            if c.sel_missing_regions[i]:
                fn = ''
            else:
                img_fn = c.wafer_strs[ind] + c.wafer_region_strs[i] + self.thumbnail_suffix
                fn = os.path.join(c.alignment_folders[ind], self.thumbnail_subfolder, img_fn)
            if os.path.isfile(fn):
                if unload and self.images_all_same_size and images_size is not None:
                    # prevents having to load all images if we can assume all sizes the the same
                    self.wafer_images_size[i,:] = images_size
                else:
                    self.wafer_images[i] = tifffile.imread(fn)
                    self.wafer_images_size[i,:] = np.array(self.wafer_images[i].shape)[::-1]
                    images_size = self.wafer_images_size[i,:].copy()
                cnt += 1
                # to save memory, might only need the keypoints and not the actual images
                if unload: self.wafer_images[i] = None

                # do not bother loading the masks at all if unload is specified
                if self.wafer.use_tissue_masks and not unload:
                    if self.wafer.tissue_mask_path is not None:
                        if self.wafer.tissue_mask_fn_str:
                            # get the z-index
                            tind = i if self.wafer.wafer_ids[0] < 2 else \
                                    (i + self.wafer.cum_wafer_manifest_cnts[self.wafer.wafer_ids[0]-2])
                            fn = os.path.join(self.wafer.tissue_mask_path, self.wafer.tissue_mask_fn_str.format(tind))
                        else:
                            # in the case that the thumbnail export name is preserved
                            fn = os.path.join(self.wafer.tissue_mask_path, img_fn)
                    else:
                        fn = os.path.join(c.alignment_folders[ind], self.thumbnail_subfolders[1], img_fn)
                    bw = tifffile.imread(fn).astype(bool)

                    if self.wafer.tissue_mask_min_size > 0:
                        # remove small components
                        labels, nlbls = nd.label(bw, structure=nd.generate_binary_structure(2,2))
                        if nlbls > 0:
                            sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                            rmv = np.nonzero(sizes < self.wafer.tissue_mask_min_size)[0] + 1
                            if rmv.size > 0:
                                bw[np.isin(labels, rmv)] = 0

                    if self.wafer.tissue_mask_min_hole_size > 0:
                        # remove small holes
                        labels, nlbls = nd.label(np.logical_not(bw),
                            structure=nd.generate_binary_structure(2,1))
                        if nlbls > 0:
                            sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                            add = np.nonzero(sizes < self.wafer.tissue_mask_min_hole_size)[0] + 1
                            if add.size > 0:
                                bw[np.isin(labels, add)] = 1

                    self.wafer_tissue_masks[i] = bw

                    doplots = False
                    if doplots:
                        # for debug only
                        img = self.wafer_images[i]
                        ovlp = bw
                        dsstep = self.wafer.tissue_mask_ds // self.dsthumbnail
                        pad = (dsstep - np.array(img.shape) % dsstep) % dsstep
                        oimg = measure.block_reduce(np.pad(img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                block_size=(dsstep, dsstep), func=self.blkrdc_func).astype(img.dtype)
                        print(oimg.shape, ovlp.shape)
                        oimg = cv2.cvtColor(oimg, cv2.COLOR_GRAY2RGB)
                        tmp = np.zeros(ovlp.shape + (3,), dtype=np.uint8)
                        tmp[ovlp,0] = 255
                        ovlp = tmp
                        oimg = cv2.addWeighted(ovlp, 0.5, oimg, 1, 0.0)
                        plt.figure(1); plt.gcf().clf()
                        plt.imshow(oimg)
                        plt.show()
                #if self.wafer.tissue_mask_path is not None:

                nfn, _ = os.path.splitext(img_fn); nfn += '_region_coords.csv'
                nfn = os.path.join(c.alignment_folders[ind], self.thumbnail_subfolder, nfn)
                if os.path.isfile(nfn):
                    pts = zimages.get_roi_coordinates(None, nfn, scl=1, cache_dn=self.cache_dir)
                    self.wafer_roi_points[i] = pts
                    ctr = PolyCentroid(pts[:,0], pts[:,1])
                    for j in range(self.roi_polygon_nscales):
                        if self.roi_polygon_scales[j] > 0:
                            self.wafer_roi_points_scaled[j][i] = (pts - ctr)*self.roi_polygon_scales[j] + ctr
                else:
                    # must have roi points saved for this option
                    assert( all([x==0 for x in self.roi_polygon_scales]) )
            else: # if os.path.isfile(fn):
                if not c.sel_missing_regions[i]:
                    print(fn)
                    assert(False) # thumbnail missing, decided to make this an error case, comment this if not

        if (rng is None and cnt < self.wafer.nvalid_regions) or (rng is not None and cnt < len(rng)):
            nvalid = self.wafer.nvalid_regions if rng is None else len(rng)
            print('WARNING: %d images of %d regions loaded (maybe ok for subset)' % (cnt, nvalid))
        assert(cnt > 1) # fatal error with cross-wafer alignment
        self.wafer_nloaded_images = cnt

        if self.wafer_solver_verbose:
             print('\tdone in %.4f s' % (time.time() - t, ))


    # had to introduce this as an internal function in order to support incremental loading of keypoints.
    # large wafers can run out of memory with all the keypoints loaded at once.
    def load_keypoints_process_dills(self, keypoints_dill_fns=None, inds=None, iroi_polygon_scale=None,
            load_all_in_dill=True):
        nwafer_ids = self.wafer.nwafer_ids
        is_cross_wafer = self.wafer.is_cross_wafer
        assert( self.keypoints_dill_fns is not None or keypoints_dill_fns is not None )
        use_keypoints_dill_fns = keypoints_dill_fns if keypoints_dill_fns is not None else self.keypoints_dill_fns
        keypoints_nprocesses = len(use_keypoints_dill_fns[0])

        nloaded = 0
        for i in range(nwafer_ids):
            for j in range(keypoints_nprocesses):
                # optimization to skip the dills that do not need to be loaded.
                _,inds_proc = self._get_keypoints_inds(j, iwafer=i)
                if inds is not None:
                    if (not is_cross_wafer and not np.in1d(inds, inds_proc).any()) or \
                            (is_cross_wafer and inds[i] not in inds_proc):
                        if self.keypoints_loaded_inds[i][j].size > 0:
                            print('Unloading keypoints dill wafer {}/{} proc {}/{}'.format(i,nwafer_ids,
                                j,keypoints_nprocesses))
                            # unload all the keypoints for this dill from memory
                            for k in self.keypoints_loaded_inds[i][j]:
                                self.wafer_descriptors[k] = None
                                self.wafer_pickleable_keypoints[k] = None
                                self.wafer_keypoints[k] = None
                            self.keypoints_loaded_inds[i][j] = np.zeros((0,), dtype=np.int64)
                        continue

                if self.keypoints_loaded_inds[i][j].size == 0:
                    print('Loading keypoints dill wafer {}/{} proc {}/{}'.format(i,nwafer_ids,j,keypoints_nprocesses))
                    t = time.time()
                    with open(use_keypoints_dill_fns[i][j], 'rb') as f: d = dill.load(f)
                    print('\tdone in %.4f s' % (time.time() - t, ))
                    assert( (inds_proc.size == d['wafer_processed_keypoints'].size) and \
                        ((inds_proc == d['wafer_processed_keypoints']).all()) )
                    # if inds is None or for a single wafer, load all the keypoints in this dill even if not requested.
                    # this allows the further optimization of not having to re-load a previously loaded dill
                    #   many times in a row.
                    all_inds_insert = (inds is None or (not is_cross_wafer and load_all_in_dill))
                    for k in d['wafer_processed_keypoints']:
                        single_ind_insert = (not all_inds_insert and \
                            ((not is_cross_wafer and k in inds) or (is_cross_wafer and k == inds[i])))
                        if all_inds_insert or single_ind_insert:
                            ik = nloaded if is_cross_wafer else k
                            self.wafer_descriptors[ik] = d['wafer_descriptors'][k]
                            self.wafer_pickleable_keypoints[ik] = d['wafer_pickleable_keypoints'][k]
                        if single_ind_insert:
                            self.keypoints_loaded_inds[i][j] = np.append(self.keypoints_loaded_inds[i][j], k)
                    if all_inds_insert:
                        self.keypoints_loaded_inds[i][j] = d['wafer_processed_keypoints']
                    nloaded += self.keypoints_loaded_inds[i][j].size
                    del d
                else:
                    print('Use previously loaded keypoints dill wafer {}/{} proc {}/{}'.format(i,nwafer_ids,
                        j,keypoints_nprocesses))

        # https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
        nunpickle = sum([x is not None for x in self.wafer_pickleable_keypoints])
        if nunpickle > 0:
            if self.wafer_solver_verbose:
                print('UnPacking pickable keypoints for %d images' % (nunpickle,)); t = time.time()
            for i in range(self.wafer.nregions):
                if self.wafer_pickleable_keypoints[i] is None: continue
                self.wafer_keypoints[i] = [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                                           response=point[3], octave=point[4], class_id=point[5]) \
                                           for point in self.wafer_pickleable_keypoints[i]]
                self.wafer_pickleable_keypoints[i] = None
            if self.wafer_solver_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))
            if iroi_polygon_scale is not None:
                self.filter_keypoints_descriptors_iroi_polygon_scale(iroi_polygon_scale)


    # this is functionized because the split computation is used both when creating the keypoints
    #   and also when deciding which keypoints dills to load when computing the matches.
    #   this prevents having to save a separate dill with just the indices in order to
    #   optimize the "on-demand" keypoints loading during the matches computation.
    def _get_keypoints_inds(self, iprocess, iwafer=0):
        #rng = np.arange(self.wafer_nimages)
        # this is mostly because of cross-wafer rough alignment. the keypoints would always be computed
        #   for all of the slices that are in the manifests (which is indexed_nregions).
        rng = np.arange(self.wafer.indexed_nregions[iwafer])
        #rng[self.wafer.sel_missing_regions] = -1 # xxx - this probably just causes problems?
        inds = np.array_split(rng, self.keypoints_nworkers_total)
        inds = inds[iprocess*self.keypoints_nworkers_per_process:(iprocess+1)*self.keypoints_nworkers_per_process]
        inds_proc = np.concatenate(inds)
        return inds, inds_proc


    def compute_wafer_keypoints(self, nfeatures, filter_size=0, rescale=False, nthreads_per_job=None, iprocess=0):
        if self.wafer_solver_verbose:
            print('Computing keypoints / descriptors for {} images'.format(self.wafer.nregions,))
            print('\tusing {} jobs with {} cv2 threads per job and {} processes'.format(\
                self.keypoints_nworkers_per_process, nthreads_per_job, self.keypoints_nprocesses))
            t = time.time()

        # this sets the number of threads used for the SIFT computation in opencv
        if nthreads_per_job is None: nthreads_per_job = self.nthreads

        inds,inds_proc = self._get_keypoints_inds(iprocess)
        nworkers = self.keypoints_nworkers_per_process
        self.wafer_processed_keypoints = inds_proc
        nimgs_proc = inds_proc.size

        if self.wafer_images is None:
            # only load images necessary for this process
            self.load_wafer_images(rng=range(inds_proc[0],inds_proc[-1]+1))

        # incase images had already been loaded, free the memory for any images that will not be processed
        self.wafer_images[:inds_proc[0]] = [None]*inds_proc[0]
        self.wafer_images[inds_proc[-1]+1:] = [None]*(self.wafer.nregions - (inds_proc[-1]+1))

        if self.wafer_solver_verbose:
            print('\tprocessing {} images in this process'.format(nimgs_proc,))
        workers = [None]*nworkers
        result_queue = mp.Queue(nimgs_proc)
        for i in range(nworkers):
            if inds[i].size == 0: continue # in case there are more total workers than images
            pts = self.wafer_roi_points_scaled[0][inds[i][0]:inds[i][-1]+1]
            msks = self.wafer_tissue_masks[inds[i][0]:inds[i][-1]+1]
            rel_ds = self.wafer.tissue_mask_ds // self.dsthumbnail
            workers[i] = mp.Process(target=compute_keypoints_job, daemon=True,
                    args=(nfeatures, nthreads_per_job, i, inds[i], self.wafer_images[inds[i][0]:inds[i][-1]+1],
                        result_queue, pts, msks, rel_ds, filter_size, rescale, self.min_feature_matches,
                        self.wafer_solver_verbose))
            workers[i].start()
        # NOTE: only call join after queue is emptied
        # https://stackoverflow.com/questions/45948463/python-multiprocessing-join-deadlock-depends-on-worker-function

        dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for i in range(nimgs_proc):
        i = 0
        while i < nimgs_proc:
            if self.wafer_solver_verbose and i>0 and i%100==0:
                print('100 through q in %.3f s, worker_cnts:' % (time.time()-dt,)); dt = time.time()
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

            if res['ind'] >= 0:
                self.wafer_pickleable_keypoints[res['ind']] = res['pickleable_keypoints']
                self.wafer_descriptors[res['ind']] = res['descriptors']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers if x is not None]
        [x.close() for x in workers if x is not None]

        # deal with missing regions by filling with two all zero descriptors
        for i in range(self.wafer_nimages):
            if self.wafer.sel_missing_regions[i]:
                self.wafer_descriptors[i] = np.zeros((2,SIFT_descriptor_ndims), dtype=SIFT_descriptor_dtype)

        if self.wafer_solver_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))


    def filter_keypoints_descriptors_iroi_polygon_scale(self, iroi_polygon_scale):
        if self.wafer_solver_verbose:
            print('\tFiltering keypoints/descriptors based on roi_polygon scale {}'.\
                format(self.roi_polygon_scales[iroi_polygon_scale],))
            dt = time.time()

        for i in range(self.wafer_nimages):
            if self.wafer_keypoints[i] is None: continue
            pts = self.wafer_roi_points_scaled[iroi_polygon_scale][i]
            if pts is not None:
                polygon = Path(pts)
                mask = polygon.contains_points(np.array([x.pt for x in self.wafer_keypoints[i]]))
                self.wafer_keypoints[i] = [x for x,m in zip(self.wafer_keypoints[i],mask) if m]
                self.wafer_descriptors[i] = self.wafer_descriptors[i][mask,:]

        if self.wafer_solver_verbose:
            print('\t\tdone in %.4f s' % (time.time() - dt, ))

    def compute_wafer_matches(self, iroi_polygon_scale=0, full=False, gpus=[0], njobs_per_gpu=1, nprocesses=1,
            iprocess=0, run_ransac=False, include_regions=None):
        if self.wafer_images is None:
            self.load_wafer_images(unload=True)
        nimgs = self.wafer_nimages
        # keypoints need to be already loaded, or using "on-demand" keypoints loading.
        #   on-demand is needed when all the keypoints and descriptors for a wafer will not fit in memory at once.
        assert(self.keypoints_dill_fns is not None or self.wafer_keypoints is not None)
        assert(self.knn_search_method == 'faiss-gpu') # only implemented the parallel version with faiss-gpu

        if include_regions is not None:
            mask_inds = np.zeros(nimgs, dtype=bool)
            mask_inds[np.array(include_regions) - 1] = 1
        else:
            mask_inds = None

        ngpus = len(gpus)
        if self.wafer_solver_verbose:
            print('Computing all pairwise wafer matches for wafer number %d' % (self.wafer.wafer_ids[0],))
            print('\tusing %d processes, %d gpus with %d jobs per gpu per process' % (nprocesses,ngpus,njobs_per_gpu))
            t = time.time()

        # if the keypoints are loaded already all at once
        if self.keypoints_dill_fns is None:
            self.filter_keypoints_descriptors_iroi_polygon_scale(iroi_polygon_scale)

        # parallelize the percent matches computation
        # nprocesses is the total number of independent processes, then these are further divided by
        #   workers in each process which is ngpus*njobs_per_gpu.
        # xxx - this is a little painful, the caller has to be aware of proper values for nprocesses.
        #   not using the argument for the actual number of processes seemed more confusing though.
        if full:
            nblk_imgs = math.isqrt(nprocesses)
            assert( nblk_imgs*nblk_imgs == nprocesses) # for full nprocesses needs to be a perfect square
        else:
            nblk_imgs = math.isqrt(2*nprocesses)
            assert( nblk_imgs*(nblk_imgs+1)//2 == nprocesses) # for triu, nprocesses needs to be x*(x-1)//2
        nworkers = ngpus*njobs_per_gpu # workers per process
        assert(iprocess < nprocesses) # bad iprocess specified

        # parallelize in blocks by splitting each axis into nblk_imgs pieces.
        blk_img_inds = np.array_split(np.arange(nimgs), nblk_imgs)
        blk_inds = np.indices((nblk_imgs, nblk_imgs), dtype=np.int64).reshape((2,-1))
        if not full:
            blk_inds = blk_inds[:,blk_inds[0,:] <= blk_inds[1,:]]
        assert( blk_inds.shape[1] == nprocesses ) # number of blocks computation is wrong
        iblk_inds = blk_inds[:,iprocess]
        xblk_inds = blk_img_inds[iblk_inds[0]]; yblk_inds = blk_img_inds[iblk_inds[1]]
        nxblk_inds = xblk_inds.size; nyblk_inds = yblk_inds.size
        load_inds = np.unique(np.concatenate((xblk_inds,yblk_inds)))
        if iblk_inds[0] == iblk_inds[1]:
            assert( nxblk_inds == nyblk_inds ) # diagonal blocks should always be square
            ntcompares = nxblk_inds*(nxblk_inds-1) if full else nxblk_inds*(nxblk_inds+1)//2
        else:
            ntcompares = nxblk_inds * nyblk_inds
        if self.wafer_solver_verbose:
            print('process total ncompares = {}'.format(ntcompares))

        # parallelize by gpu workers within this process block
        x,y = nsd(nworkers)
        if x > y and nxblk_inds > nyblk_inds:
            nxblks,nyblks = x,y
        else:
            nxblks,nyblks = y,x
        xinds = np.array_split(xblk_inds, nxblks)
        yinds = np.array_split(yblk_inds, nyblks)

        # compute the number of comparisons for each worker.
        # basically this prevents having to wait for all workers to finish
        #   in order to terminate the job in case one worker dies (gpu OOM for example).
        ncompares = np.zeros(nworkers, dtype=np.int64)
        i = 0
        for x in range(nxblks):
            for y in range(nyblks):
                xi, yi = np.meshgrid(xinds[x], yinds[y], indexing='ij')
                if full:
                    sel = (xi.reshape(-1) != yi.reshape(-1))
                else:
                    sel = (xi.reshape(-1) < yi.reshape(-1))
                ncompares[i] = sel.sum()
                i += 1
        assert(ncompares.sum() == ntcompares) # your math is wrong
        del xi, yi, sel

        if run_ransac:
            if self.rigid_type == RigidRegression_types.affine:
                poly = preprocessing.PolynomialFeatures(degree=1)
            else:
                poly = None
            ransac = AffineRANSACRegressor(stop_probability=1-1e-6, max_trials=self.ransac_max,
                    loss=lambda true,pred: ((true - pred)**2).sum(1),
                    residual_threshold=self.residual_threshold**2, rigid_type=self.rigid_type)
        else:
            ransac = poly = None

        # load keypoints and descriptors for this process block.
        # the parallelization is designed in xy blocks of the percent matches matrix in order to keep these local.
        # for many datasets all the keypoints and descriptors can not fit into memory at once.
        if self.keypoints_dill_fns is not None:
            self.load_keypoints_process_dills(inds=load_inds, iroi_polygon_scale=iroi_polygon_scale)

        # separate this loop so assert kills the job before the workers start.
        # otherwise the non-daemon processes will continue to completion as zombies.
        workers = [None]*nworkers
        result_queue = mp.Queue(ntcompares)
        i = 0
        for x in range(nxblks):
            for y in range(nyblks):
                xd = self.wafer_descriptors[xinds[x][0]:xinds[x][-1]+1]
                yd = self.wafer_descriptors[yinds[y][0]:yinds[y][-1]+1]
                if run_ransac:
                    xk = self.wafer_keypoints[xinds[x][0]:xinds[x][-1]+1]
                    yk = self.wafer_keypoints[yinds[y][0]:yinds[y][-1]+1]
                else:
                    xk = yk = None
                # normally set daemon true so any workers get killed and no zombie processes are left.
                # for the ransac percent matches tho, the workers also need to spawn ransac workers,
                #   so just do not worry about the zombies, slurm should kill them (xxx - ???)
                workers[i] = mp.Process(target=compute_matches_job, daemon=not run_ransac,
                        args=(gpus[i % ngpus], i, nimgs, xinds[x], yinds[y], full, xd, yd, xk, yk,
                            mask_inds, result_queue, self.max_npts_feature_correspondence, self.lowe_ratio,
                            self.min_feature_matches, self.dsthumbnail, ransac, poly, self.ransac_repeats,
                            self.nthreads, self.wafer_solver_verbose))
                workers[i].start()
                i += 1
        # NOTE: only call join after queue is emptied

        self.percent_matches = np.zeros((nimgs,nimgs), dtype=np.double)
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        nprint = 1000
        i = 0; dt = time.time()
        while i < ntcompares:
            if self.wafer_solver_verbose and i > 0 and i % nprint ==0:
                print('{} through q in {:.3f} s, worker_cnts:'.format(nprint, time.time()-dt,))
                print(worker_cnts); dt = time.time()

            try:
                res = result_queue.get(block=True, timeout=self.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != ncompares[x]:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,
                                worker_cnts[x], ncompares[x]))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue
            self.percent_matches[res['indx'], res['indy']] = res['percent_match']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers if x is not None]
        [x.close() for x in workers if x is not None]

        if self.wafer_solver_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))


    def _get_points_from_features(self, isrc, idst, gpus=[], iprocess=0):
        if self.knn_search_method == 'sklearn':
            nbrs = NearestNeighbors(n_neighbors=2, metric='l2', algorithm='kd_tree',
                                    n_jobs=self.nthreads).fit(self.wafer_descriptors[isrc])
            if self.wafer_solver_verbose:
                print('sklearn kneighbors for sift compare, {} to {}'.format(isrc,idst)); t = time.time()
            distances, indices = nbrs.kneighbors(self.wafer_descriptors[idst])
            if self.wafer_solver_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))
        elif self.knn_search_method[:5] == 'faiss':
            # xxx - since we did this they added some methods for the full search in faiss
            #   that supposedly can save some memory. it seemed overly complicated relative
            #   to this, so punted for now, but probably worth revisiting.
            d = self.wafer_descriptors[isrc].shape[1]
            index_cpu = faiss.IndexFlatL2(d)            # build the index
            index_cpu.add(self.wafer_descriptors[isrc]) # add vectors to the index
            if self.knn_search_method == 'faiss-gpu':
                res = faiss.StandardGpuResources()      # use a single GPU
                ngpus = len(gpus)
                gpu_index = self.wafer.default_gpu_index if ngpus==0 else gpus[iprocess % ngpus]
                index = faiss.index_cpu_to_gpu(res, gpu_index, index_cpu)
            else:
                assert(False) # bad or unimplemented faiss search method
            k = 2 # nearest-neighbors
            distances, indices = index.search(self.wafer_descriptors[idst], k)     # actual search
        else:
            assert(False) # bad or unimplemented search method

        if self.knn_search_method == 'sklearn' or self.knn_search_method[:5] == 'faiss':
            sel = (distances[:,0] > 0); distances[np.logical_not(sel),0] = 1
            msk = np.logical_and(sel, distances[:,1] / distances[:,0] > self.lowe_ratio)

        # heuristic feature to reject points that mostly map to the same point
        if self.max_npts_feature_correspondence > 0:
            if self.knn_search_method == 'sklearn' or self.knn_search_method[:5] == 'faiss':
                cnt_nmsk = (np.bincount(indices[:,0]) > self.max_npts_feature_correspondence)
                remove_msk = np.in1d(indices[:,0], np.nonzero(cnt_nmsk)[0])
                msk = np.logical_and(msk, np.logical_not(remove_msk))
            else:
                assert(False) # max_npts_feature_correspondence not implemented for other search methods

        # return the closest points to pts_dst in pts_src for points that passed the lowe test.
        # this is a mapping for each point in pts_dst to the closest point in pts_src.
        pypts_src = [x.pt for x in self.wafer_keypoints[isrc]]
        if self.knn_search_method == 'sklearn' or self.knn_search_method[:5] == 'faiss':
            pts_src = np.array(pypts_src)[indices[:,0],:]
        pts_dst = np.array([x.pt for x in self.wafer_keypoints[idst]])

        return msk.sum(), pts_src[msk,:], pts_dst[msk,:]


    # this is a wrapper around the shared _ransac_repeat that is specific to the solver, i.e.,
    #   retreiving the matching sift points, and using some heuristics to reject really bad point matches.
    def _iterative_ransac(self, x, y, poly, ransac, iscl, gpus=[], iprocess=0, nworkers=1):
        npts, tpts_src, tpts_dst = self._get_points_from_features(x,y, gpus=gpus, iprocess=iprocess)

        # trim points by the scaled roi polygon if specified
        if self.wafer_roi_points_scaled[iscl][x] is not None:
            polygon = Path(self.wafer_roi_points_scaled[iscl][x])
            mask = polygon.contains_points(tpts_src)
            tpts_src = tpts_src[mask,:]; tpts_dst = tpts_dst[mask,:]
        if self.wafer_roi_points_scaled[iscl][y] is not None:
            polygon = Path(self.wafer_roi_points_scaled[iscl][y])
            mask = polygon.contains_points(tpts_dst)
            tpts_src = tpts_src[mask,:]; tpts_dst = tpts_dst[mask,:]
        npts = tpts_src.shape[0] # update npts incase points were trimmed

        # transform the points to the image space.
        # NOTE: do NOT center or translate the points.
        #   this does not work for applying the affine to the image directly.
        pts_src = tpts_src * self.dsthumbnail
        pts_dst = tpts_dst * self.dsthumbnail

        # initialize
        mask = np.zeros((npts,), dtype=bool); npts_fit = 0
        any_ransac_success = False; fail_cnt = 0; affine = None
        # this is mostly for debugging problems solving the order.
        # indicates what is failing in the case of all failures (ranscac or which heuristic).
        #  0 success
        # -1 no solution from ransac
        # -2 not enough points from feature matches
        # -3 not enough fitted ransac points
        # -4 missing regions lumping
        #  1 radial std heuristic
        #  2 translation heuristic
        fail_type = -2

        # if the feature matching already is below min feature matches
        #   before fitting the xform, then return failure immediately.
        if npts < self.min_feature_matches:
            return npts, pts_src, pts_dst, tpts_src, tpts_dst, affine, mask, npts_fit, any_ransac_success, fail_type
        fail_type = -1

        Xpts = poly.fit_transform(pts_src) if self.rigid_type == RigidRegression_types.affine else pts_src

        for cnt in range(self.ransac_repeats):
            # NOTE: use nworkers for repeats so they all run in parallel.
            #   when the ransac was instantiated we divided ransac max by nworkers,
            #     so use nworkers here as number of "repeats".
            #   the true ransac repeats is the outer loop here.
            caffine, cmask = _ransac_repeat(Xpts, pts_dst, ransac, nworkers,
                    verbose=self.wafer_solver_verbose, nworkers=nworkers)

            if caffine is not None:
                if self.rigid_type == RigidRegression_types.affine:
                    # scikit learn puts constant terms on the left, flip and augment
                    caffine = np.concatenate( (np.concatenate( (caffine[:,1:], caffine[:,0][:,None]), axis=1 ),
                                               np.zeros((1,3), dtype=caffine.dtype)), axis=0 )
                    caffine[2,2] = 1

                # heuristics to flag bad matches

                # reject this fit if most of the fitted points are nearby each other.
                # NOTE: this heuristic can also fail for points in a circle or thin annulus.
                d1 = np.sqrt(((pts_src[cmask,:] - pts_src[cmask,:].mean(0))**2).sum(1))
                d2 = np.sqrt(((pts_dst[cmask,:] - pts_dst[cmask,:].mean(0))**2).sum(1))
                #stds = [np.std(d1), np.std(d2)]
                # use median absolute deviation to be more robust to outliers
                stds = [1.4826*np.median(np.abs(x - np.median(x))) for x in [d1,d2]]
                heuristic_radial_std_pass = all([x > self.min_fit_pts_radial_std for x in stds])

                # reject this fit if the translation is outside of some tolerance
                heuristic_translation_pass = (np.abs(caffine[:2,2]) <= self.max_fit_translation).all()

                # check that all heuristics passed and encode failture type in a bit string.
                # ransac fitting failure is -1, default set above.
                heuristic_pass = (heuristic_radial_std_pass and heuristic_translation_pass)

                if heuristic_pass:
                    if cmask.sum() > npts_fit:
                        any_ransac_success = True; mask = cmask; affine = caffine
                        npts_fit = mask.sum(); fail_type = 0; fail_cnt = 0
                else:
                    fail_cnt += 1
                    if fail_type != 0:
                        fail_type = ((not heuristic_radial_std_pass)<<0) + ((not heuristic_translation_pass)<<1)

        # code path for this combines forward and reverse outside this function.
        # only set failure type here.
        if npts_fit < self.min_feature_matches: fail_type = -3

        return npts, pts_src, pts_dst, tpts_src, tpts_dst, affine, mask, npts_fit, any_ransac_success, fail_type


    # <<< plot functions for sift correspondences
    def _matching_keypoints_masks(self, i, pts, msk):
        # get keypoints that did not match at all (Lowe test)
        ckeypts_nomatch = np.array([xx.pt for xx in self.wafer_keypoints[i]])
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts)
        d = nbrs.kneighbors(ckeypts_nomatch, return_distance=True)[0]
        ckeypts_nomatch = ckeypts_nomatch[d.reshape(-1) > 0, :]

        # get the matching keypoints and those fitting the transform
        ckeypts_match_nofit = pts[np.logical_not(msk),:]
        ckeypts_match = pts[msk,:]

        return ckeypts_nomatch, ckeypts_match_nofit, ckeypts_match

    def plot_correspondence(self,i,ind,ix,iy,msk,inpts,ppts_src,ppts_dst):
        xpts_nomatch, _, _ = self._matching_keypoints_masks(ix, ppts_src, msk)
        ypts_nomatch, _, _ = self._matching_keypoints_masks(iy, ppts_dst, msk)

        # setting this to false makes a better "figure" plot, to view the entire slices at once.
        # however, for debug, it is not zoomable, so that both images can be zoomed together.
        # so, for a pretty picture, use False, for debuging correspondence use True
        zoomable_correspondence = True
        if zoomable_correspondence:
            gs1 = gridspec.GridSpec(2,2)
            gs1.update(wspace=0, hspace=0.1) # set the spacing between axes.

        # opencv function to make this plot was a pain to call (without exact opencv inputs).
        if ind==0:
            plt.figure(1, figsize = (10,10)); plt.gcf().clf()
        else:
            plt.figure(1)
        nmsk = np.logical_not(msk)
        clrs = [[x/255 for x in y] for y in [[255,0,0], [255,0,0], [0,0,255]]] # no match, no fit, match/fit
        if zoomable_correspondence:
            ax1 = plt.subplot(gs1[2*ind])
            plt.imshow(self.wafer_images[ix], cmap='gray'); plt.axis('off')
            plt.scatter(ppts_src[msk,0], ppts_src[msk,1], s=12, edgecolors=clrs[2], facecolors='none')
            plt.scatter(ppts_src[nmsk,0], ppts_src[nmsk,1], s=12, edgecolors=clrs[1], facecolors='none')
            plt.scatter(xpts_nomatch[:,0], xpts_nomatch[:,1], s=12, edgecolors=clrs[0], facecolors='none')
            ax2 = plt.subplot(gs1[2*ind+1])
            plt.imshow(self.wafer_images[iy], cmap='gray'); plt.axis('off')
            plt.scatter(ppts_dst[msk,0], ppts_dst[msk,1], s=12, edgecolors=clrs[2], facecolors='none')
            plt.scatter(ppts_dst[nmsk,0], ppts_dst[nmsk,1], s=12, edgecolors=clrs[1], facecolors='none')
            plt.scatter(ypts_nomatch[:,0], ypts_nomatch[:,1], s=12, edgecolors=clrs[0], facecolors='none')
            for j in range(inpts):
                if msk[j]:
                    con = patches.ConnectionPatch(
                              xyA=tuple(ppts_src[j,:].tolist()),
                              xyB=tuple(ppts_dst[j,:].tolist()),
                              coordsA="data", coordsB="data", axesA=ax1, axesB=ax2,
                              color=clrs[2], linewidth=0.5)
                    ax1.add_artist(con)
            ax1.set_zorder(1); ax2.set_zorder(0)
        else:
            show_correspondence_lines = False
            sz = self.wafer_images_size[ix,:].copy(); sz[1] = 0
            img = np.concatenate((self.wafer_images[ix],self.wafer_images[iy]), axis=1)
            plt.subplot(2,1,ind+1)
            plt.imshow(img, cmap='gray'); plt.axis('off')
            plt.scatter(ppts_src[msk,0], ppts_src[msk,1], s=12, edgecolors=clrs[2], facecolors='none', alpha=0.1)
            plt.scatter(ppts_src[nmsk,0], ppts_src[nmsk,1], s=12, edgecolors=clrs[1], facecolors='none', alpha=0.1)
            plt.scatter(xpts_nomatch[:,0], xpts_nomatch[:,1], s=12, edgecolors=clrs[0], facecolors='none', alpha=0.1)
            xpts_dst = ppts_dst + sz
            plt.scatter(xpts_dst[msk,0], ppts_dst[msk,1], s=12, edgecolors=clrs[2], facecolors='none', alpha=0.1)
            plt.scatter(xpts_dst[nmsk,0], ppts_dst[nmsk,1], s=12, edgecolors=clrs[1], facecolors='none', alpha=0.1)
            tmp = ypts_nomatch + sz
            plt.scatter(tmp[:,0], tmp[:,1], s=12, edgecolors=clrs[0], facecolors='none', alpha=0.1)
            if show_correspondence_lines:
                for j in range(inpts):
                    if msk[j]:
                        plt.plot([ppts_src[j,0], xpts_dst[j,0]], [ppts_src[j,1], ppts_dst[j,1]],
                                 clrs[2], linewidth=1, alpha=0.2)
            plt.plot([sz[0], sz[0]], [0-sz[1]*0.05, sz[1]*1.05], 'k', linewidth=5)
        plt.title('(all base1) order %d, compare %s to %s, %d / %d matches' % \
                  (i+1, self.wafer.wafer_region_strs[ix],
                   self.wafer.wafer_region_strs[iy], msk.sum(), inpts))

    def plot_overlay(self,ind,ix,iy,aff,msk,inpts,ppts_src,ppts_dst):
        plt.figure(2)
        if ind==0: plt.gcf().clf()
        if aff is None: return
        plt.subplot(2,2,2*ind+1)
        plt.scatter(ppts_src[msk,0], ppts_src[msk,1], s=12,edgecolors='r',facecolors='none',alpha=0.2)
        plt.scatter(ppts_dst[msk,0], ppts_dst[msk,1], s=12,edgecolors='b',facecolors='none',alpha=0.2)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
        plt.subplot(2,2,2*ind+2)
        xpts_src = np.dot(ppts_src, aff[:-1,:-1].T) + aff[:-1,-1]
        plt.scatter(xpts_src[msk,0], xpts_src[msk,1], s=12,edgecolors='r',facecolors='none',alpha=0.2)
        plt.scatter(ppts_dst[msk,0], ppts_dst[msk,1], s=12,edgecolors='b',facecolors='none',alpha=0.2)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')

    def keypoints_overlay_image(self, x, pts, msk, dosave_path, fnstr, iorder):
        ckeypts_nomatch, ckeypts_match_nofit, ckeypts_match = self._matching_keypoints_masks(x, pts, msk)

        # get keypoints that did not match at all (Lowe test)
        ckeypts_nomatch = np.array([xx.pt for xx in self.wafer_keypoints[x]])
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pts)
        d = nbrs.kneighbors(ckeypts_nomatch, return_distance=True)[0]
        ckeypts_nomatch = ckeypts_nomatch[d.reshape(-1) > 0, :]

        # get the matching keypoints and those fitting the transform
        ckeypts_match_nofit = pts[np.logical_not(msk),:]
        ckeypts_match = pts[msk,:]

        # convert to color image for the overlays
        oimg = cv2.cvtColor(self.wafer_images[x], cv2.COLOR_GRAY2RGB)

        ckeypts = [ckeypts_nomatch, ckeypts_match_nofit, ckeypts_match]
        clrs = [[255,0,0], [255,0,0], [0,0,255]] # match vs no match only
        for i in range(len(ckeypts)):
            # create a boolean overlay containing dilated keypoints
            ikeypts = np.round(ckeypts[i]).astype(np.int64)
            bwkeypts = np.zeros(self.wafer_images_size[x,::-1], dtype=bool)
            bwkeypts[ikeypts[:,1], ikeypts[:,0]] = 1
            conn8 = nd.generate_binary_structure(2,2)
            bwkeypts = nd.binary_dilation(bwkeypts, structure=conn8, iterations=3)

            # overlay the dilated keypoints
            #if i==0: overlay = np.zeros(bwkeypts.shape + (3,), dtype=np.uint8)
            #overlay[bwkeypts, :] = 0 # overwrite any existing pixels so that colors do not mix, comment for mixing
            #overlay[bwkeypts, :] = clrs[i]
            # overwrite pixels with the dilated keypoints.
            # because of lack of true overlay support, this looks better (?).
            oimg[bwkeypts, :] = clrs[i]
        # opencv does not support true overlays, depsite most of what you read on the internet,
        #   this is NOT the same as an alpha channel.
        #oimg = cv2.addWeighted(overlay, 1, oimg, 1, 0.0)

        # # also overlay the roi polygon
        # line_thickness = 3 #; circle_rad = 5
        # iroi_points = self.wafer_roi_points_scaled[0][x].astype(np.int32)
        # cv2.polylines(oimg, [iroi_points.reshape((-1,1,2))], True, (255,0,0), line_thickness)

        fn = os.path.join(dosave_path, 'sift', self.wafer.wafer_strs[0] + \
                'iorder{:05d}_'.format(iorder) + fnstr + self.wafer.wafer_region_strs[x] + self.thumbnail_suffix)
        print(fn)
        tifffile.imwrite(fn, oimg)
    # plot functions for sift correspondences >>>


    @staticmethod
    def preprocess_percent_matches(percent_matches, missing_regions=None, topn_cols=0, normalize=False,
            normalize_minmax=False, random_exclude_perc=0.):
        n = percent_matches.shape[0]

        # xxx - see comments in normxcorr_tsp_solver, could not get concorde to work with sparse inputs
        #   just convert it to dense here if sparse specified.
        if sp.issparse(percent_matches): percent_matches = percent_matches.todense()

        if random_exclude_perc > 0:
            assert( missing_regions is None or len(missing_regions)==0 ) # not with random_exclude_perc
            nexclude = int(random_exclude_perc * n)
            if nexclude > 0:
                missing_regions = np.random.choice(np.arange(1,n+1), size=nexclude, replace=False)

        isfull = (np.tril(percent_matches) > 0).any()
        havemissing = (missing_regions is not None and missing_regions.size > 0)
        # modifications below assume the matrix is full, so make it full if it's not.
        if not isfull:
            percent_matches = percent_matches + percent_matches.T
        elif topn_cols > 0 or havemissing:
            percent_matches = percent_matches.copy()

        assert( topn_cols == 0 or not normalize ) # not meant to work together
        assert( not normalize_minmax or normalize ) # specify both
        if topn_cols > 0:
            # remove everything excpet the top-n, but in place... xxx - must be an easier way to do this select.
            i = np.argsort(percent_matches, axis=1)
            i = np.concatenate((np.repeat(np.arange(n)[:,None], topn_cols, axis=1)[:,:,None],
                    i[:,-topn_cols:][:,:,None]), axis=2).reshape(-1,2)
            sel = np.ones_like(percent_matches, dtype=bool); sel[i[:,0], i[:,1]] = 0
            percent_matches[sel] = 0
            # rescale linearly using rank order of top percent matches over columns
            i = np.arange(n)[np.argsort(percent_matches.max(1))]
            # xxx - param for range here?
            scl = np.linspace(0.2, 1., percent_matches.shape[0])[i]
            percent_matches = percent_matches / percent_matches.max(1)[:,None] * scl[:,None]
        elif normalize:
            # assume the matrix is "roughly symmetric" and normalize the rows.
            # this prevents any really good matching regions from being weighted more heavily than others.
            if normalize_minmax:
                percent_matches = (percent_matches - percent_matches.min(1)[:,None]) / percent_matches.max(1)[:,None]
            else:
                percent_matches = (percent_matches - percent_matches.mean(1)[:,None]) / percent_matches.std(1)[:,None]
                # re-normalize to [0,1] for "correlation"
                percent_matches = (percent_matches - percent_matches.min())
                percent_matches = (percent_matches / percent_matches.max())

        # another possibility for excluding regions before the solver but without modifying the manifest.
        # give all the exclude regions high percent matches to each other but "no connection" to everything else.
        if havemissing:
            excl = missing_regions - 1 # missing_regions was left as 1-based indexing
            percent_matches[excl,:] = 0; percent_matches[:,excl] = 0
            # xxx - this did not help somehow,
            #   the excludes seemed to cluster together even without forcing connections
            #for i in excl:
            #    cexcl = excl[excl != i]
            #    percent_matches[i,cexcl] = pmax; percent_matches[cexcl,i] = pmax

        # xxx - concorde does not support assymetric TSP, so would have to convert using the
        #   "doubling the nodes" method. did not seem worth it.
        #   so, instead, aggreagate them before the solver is called.
        # xxx - what is the best method of converting the assymetric to symmetric distances? min/max/mean?
        #   so that you do not try this again, you compared min/max/mean with mea 2pt3 and emperically
        #     max gives the best results. likely you tried this in the past also. do not currently have
        #     a good intuitive explanation as to why max is best.
        percent_matches = np.maximum(np.triu(percent_matches), np.tril(percent_matches).T)

        return percent_matches, missing_regions


    def compute_wafer_alignments(self, solved_order=None, solved_order_mask=None, beg_ind=0, end_ind=-1,
            iroi_polygon_scale=-1, percent_matches_topn=0, percent_matches_normalize=False,
            percent_matches_normalize_minmax=False, random_exclude_perc=0., gpus=[], iprocess=0, nworkers=1,
            doplots=False, dosave_path='', keypoints_overlays=False):
        if self.wafer_solver_verbose:
            print('Computing wafer alignments for wafer number %d' % (self.wafer.wafer_ids[0],))
        do_overlays = (keypoints_overlays and doplots)

        assert( solved_order is not None or self.percent_matches is not None ) # compute_matches first
        if solved_order is None:
            assert( self.wafer.nregions == self.percent_matches.shape[0] ) # mismatching matches matrix
            percent_matches, random_excludes = wafer_solver.preprocess_percent_matches(self.percent_matches,
                    self.wafer.missing_regions, topn_cols=percent_matches_topn, normalize=percent_matches_normalize,
                        normalize_minmax=percent_matches_normalize_minmax, random_exclude_perc=random_exclude_perc)

            print('Computing optimal route with TSP solver'); t = time.time()
            # use tsp solver to find the optimal hamiltonian path, the solved region ordering.
            self.solved_order, endpoints = wafer_solver.normxcorr_tsp_solver(percent_matches)
            print('\tdone in %.4f s' % (time.time() - t, ))

            tmp = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=sys.maxsize)
            print(self.solved_order)
            np.set_printoptions(threshold=tmp)
            return random_excludes

        # default to cpu if gpus are not available.
        # NOTE: can not do query cuda devices in a global init because cuda can not be forked,
        #   meaning any processes that try to use cuda will fail. another option is 'spawn'
        #   but all the conditional code required for this is a nightmare.
        self.wafer.query_cuda_devices()
        if self.wafer.cuda_device_count == 0:
            print('WARNING: no gpus available, defaulting to cpu method for feature matching')
            self.knn_search_method = 'sklearn'

        self.solved_order = np.array(solved_order)
        # NOTE: can not easily optimize this (i.e. only load subset) because the images are loaded in
        #   the manifest ordering, and then accessed below in the solved ordering. we would have to
        #   use the solved ordering along with inds_proc to load the appropriate images.
        if self.wafer_images is None:
            self.load_wafer_images(unload=not doplots)
        nimgs = self.wafer_nimages
        nloaded = self.wafer_nloaded_images
        assert(self.solved_order.size == nloaded) # bad manual region order
        if end_ind < 0: end_ind = self.solved_order.size-1

        if solved_order_mask is None:
            solved_order_mask = np.ones((self.solved_order.size,), dtype=bool)
        else:
            assert(solved_order_mask.size == self.solved_order.size)

        # array of polygon scale indices to be used below, need here for the keypoints loading also.
        iscls = range(self.roi_polygon_nscales) if iroi_polygon_scale < 0 else [iroi_polygon_scale]

        # filter at the first scaled polygon index before the loop.
        # this is the largest index, so remove any keypoints outside of there to save on
        #   descriptor nearest neighbor search time, since these points would be discarded
        #   afterwards anyways.
        if self.keypoints_dill_fns is None:
            self.filter_keypoints_descriptors_iroi_polygon_scale(iscls[0])
        else:
            # load the keypoints just for the subset of slices to be run, if they were not all loaded.
            # in this codepath keypoint filtering is done inside self.load_keypoints_process_dills
            inds_proc = (self.wafer.region_inds-1) if self.wafer.is_cross_wafer \
                    else self.solved_order[beg_ind:end_ind+1]
            self.load_keypoints_process_dills(inds=inds_proc, iroi_polygon_scale=iscls[0], load_all_in_dill=False)

        # save the points in the region order that do not match well based on a minimum number of matching features
        #   and a good ransac rigid alignment fit.
        self.solved_order_bad_matches = np.zeros((nimgs,2), dtype=np.int32)
        self.solved_order_bad_matches_inds = np.zeros((nimgs,), dtype=np.int32)
        self.bad_matches_fail_types = np.zeros((nimgs,2), dtype=np.int32)
        nbad_matches = 0

        if self.wafer_solver_verbose:
            print('\tComputing transforms for rng %d-%d, %d (of %d) images' % (beg_ind, end_ind,
                end_ind-beg_ind, nloaded,))
            if do_overlays:
                print('\tOnly exporting keypoints overlaid on images')
                assert(not self.wafer.is_cross_wafer) # what are you doing???
            t = time.time(); dt = t

        poly = preprocessing.PolynomialFeatures(degree=1) if self.rigid_type == RigidRegression_types.affine else None
        # a bit not as advertised here, wanted to keep _ransac_repeat generic, so decided to do the parallel
        #   versions using fractions of ransac_max, so that multiple repeats can be checked against the heuristics.
        ransac = AffineRANSACRegressor(stop_probability=1-1e-6, max_trials=int(np.ceil(self.ransac_max / nworkers)),
                                      loss=lambda true,pred: ((true - pred)**2).sum(1),
                                      residual_threshold=self.residual_threshold**2, rigid_type=self.rigid_type)

        # allocate same number offines as images and not -1 to leave room for the "cross-wafer" affines.
        self.forward_affines = [None]*nloaded
        self.reverse_affines = [None]*nloaded
        self.forward_pts_src = [None]*nloaded; self.forward_pts_dst = [None]*nloaded
        self.reverse_pts_src = [None]*nloaded; self.reverse_pts_dst = [None]*nloaded
        self.affine_percent_matches = np.zeros((nloaded,2), dtype=np.double)
        self.roi_polygon_scale_index = -np.ones((nloaded,), dtype=np.int8)
        if self.wafer_solver_verbose:
            print('\tusing roi polygon scales:'); print(self.roi_polygon_scales)
            print('\tusing roi polygon scale indices:'); print(iscls)
        #for i in range(beg_ind, end_ind-1): # nasty bug found during robin rough alignment
        for i in range(beg_ind, end_ind if end_ind < self.solved_order.size else end_ind-1):
            x,y = self.solved_order[i], self.solved_order[i+1]

            for iscl in iscls:
                self.roi_polygon_scale_index[i] = iscl
                if self.wafer_solver_verbose and i>0 and i%100==0:
                    print('\t\tiscl %d of %d, imgs %d of %d in %.4f s' % (iscl,len(iscls),i+1,nimgs-1,
                        time.time() - dt, )); dt = time.time()
                if self.wafer.sel_missing_regions[x] or self.wafer.sel_missing_regions[y] or not solved_order_mask[i]:
                    bad_match = not solved_order_mask[i]
                    # this is to "lump together" missing regions.
                    # don't flag a bad match for multiple missing in a row.
                    bad_match = bad_match or (self.wafer.sel_missing_regions[x] != self.wafer.sel_missing_regions[y])
                    forward_fail_type = reverse_fail_type = -4
                    break # do not iterate the roi polygon scale loop for missing regions
                else:
                    npts_forward, pts_src, pts_dst, tpts_src, tpts_dst, affine_forward, mask_forward, \
                        npts_fit_forward, forward_ransac_success, forward_fail_type = self._iterative_ransac(x,y,
                            poly, ransac, iscl, gpus=gpus, iprocess=iprocess, nworkers=nworkers)
                    forward_pts_src = pts_src[mask_forward,:]; forward_pts_dst = pts_dst[mask_forward,:]

                    if do_overlays:
                        self.keypoints_overlay_image(x, tpts_src, mask_forward, dosave_path, 'forward_src_', i)
                        self.keypoints_overlay_image(y, tpts_dst, mask_forward, dosave_path, 'forward_dst_', i)
                    elif doplots:
                        self.plot_correspondence(i,0,x,y,mask_forward,npts_forward,tpts_src,tpts_dst)
                        self.plot_overlay(0,x,y,affine_forward,mask_forward,npts_forward,pts_src,pts_dst)

                    npts_reverse, pts_src, pts_dst, tpts_src, tpts_dst, affine_reverse, mask_reverse, \
                        npts_fit_reverse, reverse_ransac_success, reverse_fail_type = self._iterative_ransac(y,x,
                            poly, ransac, iscl, gpus=gpus, iprocess=iprocess, nworkers=nworkers)
                    reverse_pts_src = pts_src[mask_reverse,:]; reverse_pts_dst = pts_dst[mask_reverse,:]

                    if do_overlays:
                        self.keypoints_overlay_image(x, tpts_dst, mask_reverse, dosave_path, 'reverse_dst_', i)
                        self.keypoints_overlay_image(y, tpts_src, mask_reverse, dosave_path, 'reverse_src_', i)
                    elif doplots:
                        self.plot_correspondence(i,1,y,x,mask_reverse,npts_reverse,tpts_src,tpts_dst)
                        self.plot_overlay(1,y,x,affine_reverse,mask_reverse,npts_reverse,pts_src,pts_dst)
                        if dosave_path:
                            # to save instead of show figures
                            for f in plt.get_fignums():
                                plt.figure(f)
                                plt.savefig(os.path.join(dosave_path,
                                    'wafer%d_%d_%d.png' % (self.wafer.wafer_ids[0],i,f)))
                        else:
                            plt.show()

                    # xxx - add another parameter here, whether to strive for both directions without bad matches?
                    #bad_match = (not forward_ransac_success or not reverse_ransac_success)
                    bad_match = (not forward_ransac_success and not reverse_ransac_success)
                    # also mark this as a bad match if both the number of ransac fits are below some threshold
                    bad_match = bad_match or (max([npts_fit_forward, npts_fit_reverse]) < self.min_feature_matches)
                    if not bad_match: break # if the match is good, do not iterate the roi polygon scale loop
                #if self.wafer.sel_missing_regions[x] or self.wafer.sel_missing_regions[y] or not solved_order_mask[i]:
            #for iscl in range(self.roi_polygon_nscales):
            if do_overlays:
                continue # hijacked this function to optionally export images with sift features
            if bad_match:
                # save the "bad matches" edges that need to be fixed manually
                self.solved_order_bad_matches[nbad_matches,:] = [x,y]
                self.solved_order_bad_matches_inds[nbad_matches] = i+1
                self.bad_matches_fail_types[nbad_matches,:] = [forward_fail_type, reverse_fail_type]
                nbad_matches += 1
            else:
                # replace a match that is below threshold if the other is above.
                # already detected above if both are below threshold (bad_match)
                if npts_fit_forward < self.min_feature_matches:
                    affine_forward = lin.inv(affine_reverse)
                    forward_pts_src = forward_pts_dst = None
                else:
                    self.affine_percent_matches[i,0] = npts_fit_forward / len(self.wafer_keypoints[y])
                if npts_fit_reverse < self.min_feature_matches:
                    affine_reverse = lin.inv(affine_forward)
                    reverse_pts_src = reverse_pts_dst = None
                else:
                    self.affine_percent_matches[i,1] = npts_fit_reverse / len(self.wafer_keypoints[x])

                self.forward_affines[i] = affine_forward; self.reverse_affines[i] = affine_reverse
                self.forward_pts_src[i] = forward_pts_src; self.forward_pts_dst[i] = forward_pts_dst
                self.reverse_pts_src[i] = reverse_pts_src; self.reverse_pts_dst[i] = reverse_pts_dst
        #for i in range(beg_ind, end_ind-1):

        self.solved_order_bad_matches = self.solved_order_bad_matches[:nbad_matches,:]
        self.solved_order_bad_matches_inds = self.solved_order_bad_matches_inds[:nbad_matches]
        self.bad_matches_fail_types = self.bad_matches_fail_types[:nbad_matches,:]

        if self.wafer_solver_verbose:
            print('\t\tdone in %.4f s' % (time.time() - t, ))
