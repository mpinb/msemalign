"""zimages.py

Base class for the msemalign package. Implements much of the mSEM tile
  and acquisition parameter file loading and also tile montaging.

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
import argparse
import os
#import sys
import uuid
import shutil
#import glob
import re
from enum import IntEnum

from PIL import Image
import imageio
import skimage.measure as measure
import scipy.linalg as lin
import scipy.ndimage as nd
import cv2
from matplotlib.path import Path
import matplotlib.pyplot as plt

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression #, LinearRegression
from scipy.special import expit

from .utils import get_num_threads, get_gpu_index, get_fft_types, get_fft_backend, get_cache_dir, get_block_reduce_func
from .utils import get_delta_interp_method, tile_nblks_to_ranges

import logging
logger = logging.getLogger(__name__)

# DO NOT delete this unless we completely remove PIL from msem package.
# Since this is the base class for the package, setting it here is essentially "global".
# Disable PIL's maximum image limit.
Image.MAX_IMAGE_PIXELS = None


class msem_input_data_types(IntEnum):
    new_msem_data = 0; zen_msem_data = 1; image_stack = 2; hdf5_stack = 3

    # SO how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch/43634746
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


zimages_blkrdc_func = get_block_reduce_func() # xxx - hacky, for static methods
class zimages(object):
    """Zeiss image object.

    Parent class for zeiss msem library.
    Representation of images in zeiss msem hexagonal grid / section layout.

    .. note::


    """

    # this is so number of cpu threads / jobs can be easily set for all of msem by setting env variable
    nthreads = get_num_threads()

    # this is the default gpu index to use for any gpu compute resource utilized
    default_gpu_index = get_gpu_index()

    # this is the default gpu index to use for any gpu compute resource utilized
    default_fft_methods = get_fft_types()

    # for some of the fft methods mutliple backends are possible (xxx - how to clean this up?)
    default_fft_backend = get_fft_backend()

    # this is a temp location, ideally memory mapped or on a fast local drive.
    cache_dir = get_cache_dir()

    # to drive what reduction is used for all downsamplings in msem package.
    blkrdc_func = staticmethod(get_block_reduce_func())

    # which method to use for interpolating / extrapolating deformation deltas
    delta_interp_method = get_delta_interp_method()

    # this is the timeout for multiprocessing queues before checking for dead workers.
    # did not see a strong need for this to be drive from command line.
    queue_timeout = 180 # in seconds

    def __init__(self, verbose=False):
        self.zimages_verbose = verbose

    # NOTE: can not do this in init because it breaks any multiprocessing with the gpu because cuda runtime
    #   can not be forked... search "cuda fork" and have fun reading issues.
    # basically any cuda init has to be done AFTER forking and not before.
    # querying the device count must trigger a cuda init in cupy (but not just importing cupy).
    def query_cuda_devices(self):
        try:
            import cupy as cp
            self.cuda_device_count = cp.cuda.runtime.getDeviceCount()
        except:
            self.cuda_device_count = 0

        self.use_gpu = (self.cuda_device_count > 0)
        self.fft_method = self.default_fft_methods[int(self.use_gpu)]
        self.fft_backend = self.default_fft_backend

    # This is a non-filtering method for correcting the zeiss msem specific "vignetting" that results in a
    #   "darkening" at borders of the image due to areas being imaged twice because of image overlap.
    # It can also correct a "brightening" near the borders that results from... (xxx - forgot what this EM artifact is).
    # Use a median method for estimating the average change at the borders and then fit this change to a sigmoid
    #   but using linear regression via a log variable substitution.
    # http://mathfaculty.fullerton.edu/mathews/n2003/LogisticEquationMod.html
    @staticmethod
    def correct_overlaps(img, ovlp_sel, max_delta, clf, fit_line=False, vaf_cutoff=0.8, doplots=False):
        # this is to shift values away from 1,0 to avoid division by zero and log(0)
        ceps = 1. / np.iinfo(img.dtype).max
        dimg = img.astype(np.double)
        ztol = np.finfo(np.double).eps

        def fit_linear_logistic(y,overlap):
            x = np.arange(y.shape[1], dtype=np.double)

            # normalize change between 0-1.
            # normalizing to full range did not work very well.
            # median normalize
            # xxx - the 2, 4 factors for estimating end points of sigmoid were empirical, add parameters?
            #   this whole method is very dependent on a proper estimation of the actual overlap area.
            #   this is done semi-precisely in mfov class by taking max image overlap based on the zeiss coordinates.
            isdecreasing = np.median(y[:,:overlap//4],0).mean() > np.median(y[:,-overlap//2:],0).mean()
            if isdecreasing:
                normmin = np.mean(np.median(y,0)[-overlap//2:])
                normrng = np.mean(np.median(y - normmin,0)[:overlap//4])
            else:
                normmin = np.mean(np.median(y,0)[:overlap//4])
                normrng = np.mean(np.median(y - normmin,0)[-overlap//2:])
            if normrng < ztol:
                return True, x, None, None, normmin, normrng, None, isdecreasing

            ynorm = (y - normmin)/normrng

            # remove everything out of sigmoid range
            # other options here did not work as well (for example allow out of range, but only add eps near 0,1).
            ynorm[ynorm <= 0] = ceps; ynorm[ynorm >= 1] = 1-ceps

            ymedian = np.median(ynorm,0)
            if fit_line:
                # fit line to logit, original method
                #yfit = log(1./ynorm - 1) # fits to all points, does not work as well
                yfit = np.log(1/ymedian - 1) # fit to median, fit line to logit, works ok
                xfit = np.tile(x[None,:], (yfit.shape[0],1)) if yfit.ndim > 1 else x
                X = np.hstack((np.ones((xfit.size,), dtype=np.double)[:,None], xfit.reshape(-1)[:,None]))
                Y = yfit.reshape(-1)
                prm = lin.lstsq(X, Y, cond=None, check_finite=False)[0]
                if np.abs(prm[1]) < ztol:
                    return True, x, ynorm, None, normmin, normrng, prm, isdecreasing
                yfitted = 1 / (1 + np.exp(prm[1]*x + prm[0]))
            else:
                # fit logistic regression, better method, only slightly more compute cost
                yfit = (ymedian > 0.5) # fit to median, bit booleans with logistic regression, works best
                if yfit.all() or not yfit.any():
                    return True, x, ynorm, None, normmin, normrng, None, isdecreasing
                X = x.reshape(-1,1); Y = yfit.reshape(-1); clf.fit(X,Y)
                prm = [clf.coef_, clf.intercept_]
                yfitted = expit(x * clf.coef_ + clf.intercept_).ravel()

                vaf = 1 - ((ymedian - yfitted)**2).mean()/ymedian.var()
                if vaf < vaf_cutoff:
                    return True, x, ynorm, None, normmin, normrng, prm, isdecreasing

            return False, x, ynorm, yfitted, normmin, normrng, prm, isdecreasing

        # get border overlap and make integer multiple of 4.
        sz = np.array(img.shape)
        ovlp = (np.ceil(np.round(sz[::-1] - max_delta)/4)*4).astype(np.int32)
        # select parameter specifies overlap in order top, bottom, left, right
        ovlp = np.tile(ovlp[:,None], (1,2)).reshape(-1)

        # get the slices on the image in same order: top, bottom, left, right
        slcs = [np.s_[:2*ovlp[0],:], np.s_[-2*ovlp[0]:,:], np.s_[:,:2*ovlp[1]], np.s_[:,-2*ovlp[1]:]]
        trans = [True, True, False, False]
        begslc = [True, False, True, False]

        if doplots:
            strs = ['top', 'bottom', 'left', 'right']
            plt.figure()

        n = 4
        x, ynorm, yfitted = [None]*n, [None]*n, [None]*n
        normmin, normrng, prm, decr = [None]*n, [None]*n, [None]*n, [None]*n
        cimg = dimg.copy()
        for i in range(4):
            if ovlp_sel[i]:
                error, x[i], ynorm[i], yfitted[i], normmin[i], normrng[i], prm[i], decr[i] = \
                    fit_linear_logistic(dimg[slcs[i]].T if trans[i] else dimg[slcs[i]], ovlp[i])

                if not error:
                    # correct depending on whether fitted sigmoid is increasing or decreasing
                    #   and on whether this is the beginning or end slice of this dimension.
                    factor = yfitted[i]*normrng[i] + normmin[i]
                    if (decr[i] and begslc[i]) or (not decr[i] and not begslc[i]):
                        # ignore weird situation of zero min here
                        if normmin[i] <= 0:
                            factor = np.ones_like(factor)
                        else:
                            factor /= normmin[i]
                    else:
                        # zero range already error in fitting
                        factor /= (normrng[i] + normmin[i])
                    cimg[slcs[i]] /= (factor[:,None] if trans[i] else factor[None,:])

                    if doplots:
                        plt.subplot(2, 2, i+1)
                        plt.plot(x[i], np.median(ynorm[i],0), 'r.-')
                        plt.plot(x[i], yfitted[i], 'b.-')
                        plt.title(strs[i])
        cimg = np.clip(np.round(cimg), 0, np.iinfo(img.dtype).max).astype(img.dtype)

        if doplots:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title("original", color='k'); plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cimg, cmap='gray')
            plt.title("corrected", color='k'); plt.axis('off')
            plt.show()

        return cimg

    # https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html
    @staticmethod
    def pyramid_blend(imgA, imgB, corner, shape, nlevels=6):
        assert( all([x==y for x,y in zip(imgA.shape, imgB.shape)]) )

        # generate Gaussian pyramids
        gpA = [None]*(nlevels+1); gpA[0] = imgA
        gpB = [None]*(nlevels+1); gpB[0] = imgB
        shapes = np.zeros((nlevels+1,2),dtype=np.int32); shapes[0,:] = imgA.shape
        for i in range(nlevels):
            gpA[i+1] = cv2.pyrDown(gpA[i])
            gpB[i+1] = cv2.pyrDown(gpB[i])
            shapes[i+1,:] = gpA[i+1].shape

        # generate Laplacian pyramids
        lpA = [None]*(nlevels+1); lpA[0] = gpA[-1]
        lpB = [None]*(nlevels+1); lpB[0] = gpB[-1]
        for i in range(nlevels):
            lpA[i+1] = cv2.subtract(gpA[-i-2], cv2.pyrUp(gpA[-i-1])[:shapes[-i-2,0],:shapes[-i-2,1]])
            lpB[i+1] = cv2.subtract(gpB[-i-2], cv2.pyrUp(gpB[-i-1])[:shapes[-i-2,0],:shapes[-i-2,1]])

        # Now crop part of B into A at each level
        c = corner; s = shape
        for i in range(nlevels):
            lpA[i][c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = lpB[i][c[0]:c[0]+s[0],c[1]:c[1]+s[1]]

        # now reconstruct
        imgO = lpA[0]
        for i in range(nlevels):
            imgO = cv2.add(cv2.pyrUp(imgO)[:shapes[-i-2,0],:shapes[-i-2,1]], lpA[i+1])

        return imgO

    # stich images based on supplied coordinates to create a montage
    # xxx - this method in particular is in bad need of refactoring... some more general approach / modularization
    #   for how the images are loaded and initially manipulated would be more ideal.
    #   also some way to merge this image load with that used by read_images, several things are duplicated.
    @staticmethod
    def montage(images, coords, scale=1., img_shape=None, img_dtype=None, color_coords=None, color_alpha=0.8,
                image_load={}, img_scale_adjusts=None, bg=0, verbose_mod=None, blending_mode="None",
                get_histos_only=False, get_overlap_sum_only=False, overlap_sum=None, blending_mode_feathering_dist=10,
                histo_ntiles=[1,1], histo_roi_polygon=None, blending_mode_feathering_min_overlap_dist=10,
                crop_size=[0,0], cache_dn='', unique_str=None, nblks=[1,1], iblk=[0,0], novlp_pix=[0,0],
                decay_params=None, img_decay_params=None, adjust=None, res_nm=1.):
        nimgs = len(images)
        # no need to have this as a parameter as currently it's only use case is with blockwise processing.
        crop_corner=None

        # save memory by processing with single precision floats
        float_dtype = np.float32

        # some images can be passed as None
        none_sel = np.array([x is None for x in images])
        # xxx - leaving this here for reference, this is probably not a good solution
        ## decided to allow None images or NaN coords to drive "missing" or rect coord images
        #none_sel = np.logical_or(none_sel, np.logical_not(np.isfinite(coords).all(1)))
        not_none_sel = np.logical_not(none_sel)
        fnz = np.nonzero(not_none_sel)[0][0]
        first_img = None

        # set the corresponding coords for the None images to NaN in order to ignore them
        coords = coords.copy(); coords[none_sel,:] = np.NaN
        assert(np.isfinite(coords[not_none_sel,:]).all()) # NaN coord for non-None images not allowed

        # if specified, the scale adjusts are per mfov, so make sure nimgs is multiple of number of scale adjusts.
        if img_scale_adjusts is not None:
            nTiles = img_scale_adjusts.shape[0]
        if img_decay_params is not None:
            if img_scale_adjusts is not None:
                assert( nTiles == img_decay_params.shape[0] )
            else:
                nTiles = img_decay_params.shape[0]
        assert( (img_scale_adjusts is None and img_decay_params is None) or nimgs % nTiles == 0 )

        # also allow images to be a list of filenames to be loaded on-the-fly
        image_load['Xpts'] = None; image_load['fXpts'] = None
        image_load['old_shape'] = None; image_load['regressor'] = None
        clf = LogisticRegression()
        def load_image(cur, ind):
            d = image_load
            if isinstance(cur, str) and len(image_load) > 0:
                fn = os.path.join(d['folder'], cur)
                if cache_dn:
                    bfn = os.path.basename(fn)
                    fnload = os.path.join(cache_dn, uuid.uuid4().hex + '_' + bfn)
                    shutil.copyfile(fn, fnload)
                else:
                    fnload = fn
                #_pil = Image.open(fnload); _img = np.asanyarray(_pil); _pil.close()
                _img = imageio.imread(fnload)
                if cache_dn: os.remove(fnload)

                tl = d['crop']; br = np.array(_img.shape) - d['crop']
                if d['crop'] is not None: _img = _img[tl[0]:br[0], tl[1]:br[1]]

                # max delta is computed in the downsampled space, since this is the size in which
                #   the images are processed (this load is at the loweest level). so, scale up the
                #   max delta by dsstep, since we perform this correction at the original resolution.
                if d['ovlp_sel'] is not None:
                    max_delta = d['max_delta'] if d['max_delta'].ndim == 1 else d['max_delta'][ind,:]
                    _img = zimages.correct_overlaps(_img, d['ovlp_sel'], max_delta*d['dsstep'], clf)

                if d['invert']:
                    _img = np.iinfo(_img.dtype).max - _img

                if d['dsstep'] > 1:
                    pad = (d['dsstep'] - np.array(_img.shape) % d['dsstep']) % d['dsstep']
                    _img = measure.block_reduce(np.pad(_img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                        block_size=(d['dsstep'], d['dsstep']), func=d['reduce']).astype(_img.dtype)

                if d['decay'] is not None and d['decay'][0] != 0:
                    # decay is modeled with 1/f, matches well to data
                    decay = 1./(d['decay'][0]*np.arange(_img.shape[0])*res_nm + d['decay'][1])
                    idecay = np.ones(_img.shape, dtype=float_dtype)*(decay[:,None].astype(float_dtype) + 1)
                    _imgmax = np.iinfo(_img.dtype).max
                    _img = np.clip(np.round(_img.astype(float_dtype) / idecay), 0, _imgmax).astype(_img.dtype)

                if d['scale_adjust'] is not None:
                    _imgmax = np.iinfo(_img.dtype).max
                    _img = np.clip(np.round(_img / d['scale_adjust']), 0, _imgmax).astype(_img.dtype)

                return _img
            else:
                return cur

        # get the datatype and shape of the images depending on what parameters were passed
        if img_dtype is None:
            # all images must be same datatype
            if isinstance(images[fnz], str):
                # load the first image to get datatype
                first_img = load_image(images[fnz], fnz)
                img_dtype = first_img.dtype
            else:
                img_dtype = images[fnz].dtype
        if np.issubdtype(img_dtype, np.integer):
            imgmax = np.iinfo(img_dtype).max
        else:
            imgmax = np.finfo(img_dtype).max
        if img_shape is None:
            if isinstance(images[fnz], str):
                # for this case all the images must be the same shape
                if first_img is None: first_img = load_image(images[fnz], fnz)
                img_shape = first_img.shape
                img_shapes = np.tile(np.array(img_shape)[None,:], (nimgs,1))
            else:
                img_shapes = np.vstack(np.array([x.shape if x is not None else (0,0) for x in images]))
        else:
            img_shapes = np.tile(np.array(img_shape)[None,:], (nimgs,1))

        # set the shape of the output image based on the coords
        corners = coords * scale
        corners -= np.nanmin(corners, axis=0)
        orig_sz_out = np.ceil(np.nanmax(corners + img_shapes[:,::-1], axis=0)).astype(np.int64)
        corners[none_sel,:] = 0; corners = np.round(corners).astype(np.int64)
        old_shape = None; adjust_poly = None

        # create a select for image centers that are outside of the roi polygon
        ctrs = corners + img_shapes/2
        if histo_roi_polygon is not None:
            polygon = Path(histo_roi_polygon)
            roi_polygon_mask = polygon.contains_points(ctrs)

        # support for optional blockwise processing. this piggybacks on previous hooks for cropping the output.
        # if blocks are specified this overrides any cropping parameters.
        if not all([x==1 for x in nblks]):
            #rngs, max_shape, min_shape, rng =
            _, _, _, rng = tile_nblks_to_ranges(orig_sz_out[::-1], nblks, novlp_pix, iblk)
            crop_size = [x[1] - x[0] for x in rng][::-1]; crop_corner = [x[0] for x in rng][::-1]

        # figure out any cropping of the output image, if specified.
        # NOTE: the only check here is if the crop is bigger than the output size, then do not use it.
        #   Beware: there is no other bounds checking on the crop size or crop corner.
        #   This feature was originally intended to crop to 32 bit image sizes, and then piggy-backed
        #     for the blockwise processing mode.
        sz_out = np.array(crop_size)
        # crop size of zero means disabled, so just use the original size.
        sel = (sz_out == 0); sz_out[sel] = orig_sz_out[sel]
        # only enable the cropping if the original size is bigger than the crop size.
        sel = (sz_out > orig_sz_out); sz_out[sel] = orig_sz_out[sel]
        cmin = ((orig_sz_out - sz_out)//2) if crop_corner is None else np.array(crop_corner)
        cmax = cmin + sz_out; shp_out = sz_out[::-1]
        logger.debug("montage crop_size, orig_sz_out, sz_out, crop_corner:")
        logger.debug("%d %d, %d %d, %d %d, %d %d", crop_size[0], crop_size[1],
                     orig_sz_out[0], orig_sz_out[1], sz_out[0], sz_out[1], cmin[0], cmin[1])

        if get_histos_only:
            histo_ntiles = np.array(histo_ntiles)
            assert(histo_ntiles.ndim == 1 and histo_ntiles.size==2)
            histo_istiled = (np.prod(histo_ntiles) > 1)
            if histo_istiled:
                histos = -np.ones((nimgs, histo_ntiles[0], histo_ntiles[1], imgmax+1), dtype=np.int64)
            else:
                histos = -np.ones((nimgs, imgmax+1), dtype=np.int64)
        elif get_overlap_sum_only:
            image = np.zeros(shp_out, dtype=np.uint8)
        else:
            # allocate output image, color_coords allows for RGB coloring of hexagonal grids
            if color_coords is None:
                if blending_mode == "feathering":
                    assert( overlap_sum is not None )
                    # overlap sum gets modified below, so make a local copy
                    overlap_sum = overlap_sum.copy()
                    # xxx - parameterize? 1 means only feather the "top-image", greater values use "underneath" images
                    feathering_min_ovlp_cnt = 1
                    feather_sum = np.zeros(shp_out, dtype=float_dtype)
                    image = np.empty(shp_out, dtype=float_dtype); image.fill(bg)
                else:
                    image = np.empty(shp_out, dtype=img_dtype); image.fill(bg)
            else:
                #image = np.zeros(np.concatenate((shp_out,[4])), dtype=np.single)
                #image[:,:,3] = color_alpha
                image = np.zeros(np.concatenate((shp_out,[3])), dtype=img_dtype)

        t = time.time()
        for i in range(nimgs):
            if verbose_mod is not None and i % verbose_mod == (verbose_mod-1):
                msg = ('\timage %d of %d in %.4f s' % (i+1, nimgs, time.time() - t, ))
                logger.info(msg)
                t = time.time()
            if images[i] is None: continue

            # see if this image intersects with the cropped output image
            imin = corners[i,:]; csz = img_shapes[i,::-1]; imax = imin + csz
            if (imin >= cmax).any() or (cmin >= imax).any():
                # this image does not overlap with the cropped output at all, continue
                continue

            # optionally do not compute histograms for tiles outside of the specified roi polygon.
            #   in this case save time by also not even loading the image.
            if get_histos_only and not (histo_roi_polygon is None or roi_polygon_mask[i]): continue

            # deal with any images overlapping below the crop
            sel = (imin < cmin)
            cimin = np.array([0,0]); cimin[sel] = (cmin - imin)[sel]
            imin = imin - cmin; imin[sel] = 0
            # deal with any images overlapping above the crop
            sel = (imax > cmax)
            cimax = csz.copy(); cimax[sel] = (cimax - (imax - cmax))[sel]
            imax = imax - cmin; imax[sel] = sz_out[sel]

            img = load_image(images[i], i)
            assert( (img_shapes[i,:] == np.array(img.shape)).all() ) # an image shape did not match

            if img_decay_params is not None:
                # decay is modeled with 1/f, matches well to data
                p0, p1 = img_decay_params[i%nTiles,:]
                if p0 != 0:
                    decay = 1./(p0*np.arange(img_shapes[i,0])*res_nm + p1)
                    idecay = np.ones(img_shapes[i,:], dtype=float_dtype)*(decay[:,None].astype(float_dtype) + 1)
                    img = np.clip(np.round(img.astype(float_dtype) / idecay), 0, imgmax).astype(img_dtype)

            # crop the current image and set the corner/size for the montage accordingly
            img = img[cimin[1]:cimax[1], cimin[0]:cimax[0]]
            c = imin[::-1]; s = (imax - imin)[::-1]

            # some precomputed things that depend on image shape, only update if the shape changes.
            # xxx - made this general even though up above we assert all shapes must be same...
            #   this function is a mess...
            if old_shape is None or (old_shape != s).any():
                old_shape = np.array(img.shape)

                if not get_histos_only and not get_overlap_sum_only and color_coords is None and \
                        blending_mode == "feathering":
                    # feather based on the distance from the image border
                    brd = np.ones(s, dtype=bool); brd[:,0]=0; brd[:,-1]=0
                    brd_distx = nd.distance_transform_cdt(brd, metric='chessboard')
                    brd = np.ones(s, dtype=bool); brd[0,:]=0; brd[-1,:]=0
                    brd_disty = nd.distance_transform_cdt(brd, metric='chessboard')
                    brd_featherx = brd_distx / blending_mode_feathering_min_overlap_dist[0]
                    brd_feathery = brd_disty / blending_mode_feathering_min_overlap_dist[1]
                    brd_featherx[brd_featherx > 1] = 1; brd_feathery[brd_feathery > 1] = 1
                    brd_feather = np.minimum(brd_featherx, brd_feathery)

                if adjust is not None and adjust.shape[1] > 1:
                    x,y = np.mgrid[:s[1],:s[0]]
                    adjust_pts = np.concatenate((x.flat[:][:,None]*res_nm, y.flat[:][:,None]*res_nm), axis=1)
                    if adjust_poly is None:
                        # formula for degree of "complete homogeneous symmetric polynomials" from number of terms
                        degree = (np.sqrt(8*adjust.shape[1] + 1) - 3)/2
                        assert(degree == int(degree)) # bad brightness adjust number of polynomial terms
                        adjust_poly = preprocessing.PolynomialFeatures(degree=int(degree))
                    adjust_pts = adjust_poly.fit_transform(adjust_pts)

            if img_scale_adjusts is not None:
                img = np.clip(np.round(img / img_scale_adjusts[i%nTiles,cimin[1]:cimax[1],cimin[0]:cimax[0]]),
                        0, imgmax).astype(img_dtype)

            if decay_params is not None and decay_params[0] > 0:
                # decay is modeled with 1/f, matches well to data
                decay = 1./(decay_params[0]*np.arange(img_shapes[i,0])*res_nm + decay_params[1])
                idecay = np.ones(img_shapes[i,:], dtype=float_dtype)*(decay[:,None].astype(float_dtype) + 1)
                img = np.clip(np.round(img.astype(float_dtype) / idecay), 0, imgmax).astype(img_dtype)

            if adjust is not None and (adjust.size == 1 or np.isfinite(adjust[i,:]).all()):
                if adjust.size == 1:
                    # allow brightness adjust to be a single scalar for the whole montage
                    adjust_z = adjust[0]
                elif adjust.shape[1] == 1:
                    adjust_z = adjust[i]
                else:
                    adjust_z = (adjust_pts * adjust[i,:]).sum(1).reshape((s[1], s[0]))
                img = np.clip(np.round(img + adjust_z.T), 0, imgmax).astype(img_dtype)

            if get_histos_only:
                # bincount is way faster than histogram
                #step = 1; bins = np.arange(0, imgmax+2, step, dtype=np.int32); #cbins = bins[:-1] + step/2
                if histo_istiled:
                    csz = img_shapes[i,::-1]; shp = csz // histo_ntiles
                    for x in range(histo_ntiles[0]):
                        for y in range(histo_ntiles[1]):
                            xy = np.array([x,y]); beg = xy*shp; end = (xy+1)*shp
                            sel = np.array([x==histo_ntiles[0]-1, y==histo_ntiles[1]-1])
                            end[sel] = csz[sel]; crp = img[beg[1]:end[1],beg[0]:end[0]]
                            #histos[i,x,y,:], bins = np.histogram(crp, bins)
                            histos[i,x,y,:] = np.bincount(np.ravel(crp), minlength=imgmax+1)
                else:
                    #histos[i,:], bins = np.histogram(img, bins)
                    histos[i,:] = np.bincount(np.ravel(img), minlength=imgmax+1)
            elif get_overlap_sum_only:
                image[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] += 1
            else:

                if color_coords is None:
                    if blending_mode == "pyramid":
                        # xxx - way too slow without cropping... cropping introduces lines again?
                        new_image = np.empty_like(image); new_image.fill(bg)
                        new_image[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = img
                        image = zimages.pyramid_blend(image, new_image, c, s)
                    elif blending_mode == "feathering":
                        image_crp = image[c[0]:c[0]+s[0],c[1]:c[1]+s[1]]
                        overlap_sum_crp = overlap_sum[c[0]:c[0]+s[0],c[1]:c[1]+s[1]]
                        feather_sum_crp = feather_sum[c[0]:c[0]+s[0],c[1]:c[1]+s[1]]
                        assert( (overlap_sum_crp == 0).sum() == 0 )

                        # count down the overlap sum so that only the last image is used in the overlap regions.
                        # xxx - parameter for first or last? currently this order is set in mfov and region,
                        #   so that the tiles that were imaged first are on top.
                        overlap_sum[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = overlap_sum_crp - 1

                        # get the overlapping and non-overlapping selects
                        novlp = (overlap_sum_crp < feathering_min_ovlp_cnt); ovlp = np.logical_not(novlp)
                        # skip any completely overlapped region
                        if novlp.sum() == 0: continue

                        # feather based on the distance from the non-overlapping object
                        novlp_dist = nd.distance_transform_cdt(ovlp, metric='chessboard')
                        #novlp_dist = nd.distance_transform_edt(ovlp)
                        ovlp_feather = 1 - novlp_dist / blending_mode_feathering_dist
                        ovlp_feather[ovlp_feather < 0] = 0

                        #feather = ovlp_feather # this works ok, but doesn't go to zero at edges
                        feather = np.minimum(ovlp_feather, brd_feather).astype(float_dtype)

                        image[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = image_crp + img.astype(float_dtype)*feather
                        feather_sum[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = feather_sum_crp + feather
                    else:
                        image[c[0]:c[0]+s[0],c[1]:c[1]+s[1]] = img
                else:
                    # make 3-color hexagonal tiling coloring
                    even_row = (color_coords[i,0] == int(color_coords[i,0]))
                    rgb = int(color_coords[i,0]) % 3 if even_row else int(color_coords[i,0]-1.5) % 3
                    # decided to make the y dim as an offset
                    rgb = (rgb + int(color_coords[i,1])) % 3
                    #image[c[0]:c[0]+s[0],c[1]:c[1]+s[1], rgb] = img / imgmax
                    image[c[0]:c[0]+s[0],c[1]:c[1]+s[1], rgb] = img
            #if get_histos_only:
        #for i in range(nimgs):

        if blending_mode == "feathering" and color_coords is None:
            # split into multiple lines with del's to try and reduce memory footprint
            feather_sum[feather_sum == 0] = 1; image = image/feather_sum; del feather_sum
            #image = np.clip(np.round(image), 0, imgmax).astype(img_dtype)
            image = np.round(image); image = np.clip(image, 0, imgmax); image = image.astype(img_dtype)

        crop_info = {'noncropped_size':orig_sz_out, 'crop_min':cmin, 'crop_max':cmax}
        return (histos if get_histos_only else image), corners, crop_info

    # load_subset, map_subset and images used for loading subset of images, typically for neighboring mfov
    @staticmethod
    def read_images(folder, filenames, crop=None, dsstep=1, reduce=zimages_blkrdc_func, load_subset=None,
                    map_subset=None, images=None, invert=False, init_only=False, cache_dn=''):
        assert((load_subset is None and map_subset is None) or (load_subset.size == map_subset.size))

        #logger.debug('read_images: filenames: %s', filenames)

        if load_subset is None:
            nimgs = len(filenames)
            irng = range(nimgs)
        else:
            nimgs = load_subset.size
            irng = map_subset

        # return the images in their zeiss mfov tile ordering, or a specified ordering
        if images is None: images = [None]*nimgs
        for i in irng:
            cfn = os.path.splitext(os.path.basename(filenames[i].replace("\\","/")))[0]

            if load_subset is None:
                #ind = imgno-1
                ind = i
                assert(ind == i) # filenames expected to be ordered by image number
            else:
                # NOTE: this is dangerous here, as it depends specifically on the formatting of the Zeiss image
                #   filenames. I guess for a general way for this to work, the images need to be parsed along with
                #   a dictionary that maps a tile location to a naming convention... meh.
                # xxx - this happened in at least one other location in the code, commented with xxxTHMBS
                fn_parts = cfn.split('_')
                imgno = int(fn_parts[3 if fn_parts[0] == 'thumbnail' else 2])
                ind = np.nonzero(imgno-1 == load_subset)[0]
                if ind.size == 0: continue
                assert(ind.size == 1)
                ind = map_subset[ind[0]]
            if i==irng[0]: first_ind = ind

            if init_only and (i != irng[0]):
                # put a blank image the same size/dtype as the first image as a placeholder
                _img = np.empty(images[first_ind].shape, dtype=images[first_ind].dtype)
            else:
                fn = os.path.join(folder,filenames[i])
                if cache_dn:
                    bfn = os.path.basename(fn)
                    fnload = os.path.join(cache_dn, uuid.uuid4().hex + '_' + bfn)
                    shutil.copyfile(fn, fnload)
                else:
                    fnload = fn
                #_pil = Image.open(fnload; _img = np.asanyarray(_pil); _pil.close()
                _img = imageio.imread(fnload)
                if cache_dn: os.remove(fnload)

                tl = crop; br = np.array(_img.shape) - crop
                if crop is not None: _img = _img[tl[0]:br[0], tl[1]:br[1]]

                if invert:
                    _img = np.iinfo(_img.dtype).max - _img

                if dsstep > 1:
                    pad = (dsstep - np.array(_img.shape) % dsstep) % dsstep
                    _img = measure.block_reduce(np.pad(_img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                            block_size=(dsstep, dsstep), func=reduce).astype(_img.dtype)

            images[ind] = _img

        return images

    @staticmethod
    def read_images_neighbors(folder, all_filenames_imgs, all_coords_imgs, total_nimgs, mfov_nimgs, mfov, neighbors,
                              neighbors_edge, to_neighbor_tiles, from_neighbor_tiles, init_only=False,
                              crop=None, dsstep=1, reduce=zimages_blkrdc_func, invert=False, cache_dn=''):
        # first process the mfov being loaded
        coords_imgs = np.zeros((total_nimgs,2), dtype=all_coords_imgs.dtype)
        coords_imgs[:mfov_nimgs,:]  = all_coords_imgs[mfov,:,:]
        filenames_imgs = [None]*total_nimgs
        filenames_imgs[:mfov_nimgs] = all_filenames_imgs[mfov]
        images = [None]*total_nimgs
        images[:mfov_nimgs] = zimages.read_images(folder, all_filenames_imgs[mfov], crop, dsstep, reduce,
                invert=invert, init_only=init_only)
        # which mfov id that each image was loaded from
        mfov_ids = np.zeros((total_nimgs,), dtype=np.int64)
        mfov_ids[:mfov_nimgs] = mfov
        # the hexagonal index of each image
        mfov_tile_ids = -np.ones((total_nimgs,), dtype=np.int64)
        mfov_tile_ids[:mfov_nimgs] = np.arange(mfov_nimgs, dtype=np.int64)

        for n,i in zip(neighbors, range(len(neighbors))):
            # iterate each outer ring to load
            for r in range(len(to_neighbor_tiles)):
                ctiles = to_neighbor_tiles[r][neighbors_edge[i],:]
                cntiles = from_neighbor_tiles[r][neighbors_edge[i],:]
                for j,k in zip(ctiles,cntiles):
                    filenames_imgs[j] = all_filenames_imgs[n][k]
                coords_imgs[ctiles,:] = all_coords_imgs[n][cntiles,:]
                # keep track of mfov and mfov tile number for each image tile.
                mfov_ids[ctiles] = n; mfov_tile_ids[ctiles] = cntiles

                zimages.read_images(folder, filenames_imgs, crop=crop, dsstep=dsstep, reduce=reduce,
                    init_only=init_only, load_subset=cntiles, map_subset=ctiles, images=images, invert=invert,
                    cache_dn=cache_dn)

        return images, filenames_imgs, coords_imgs, mfov_ids, mfov_tile_ids

    # xxx - maybe delete? only the old code path that allowed reading of an individual mfov coordinates
    #   was using this anymore. commented this for now, also delete that spot if we abondon this entirely.
    # @staticmethod
    # def read_image_coords(fn, nimgs, ndims=2):
    #     print("read_image_coords '{}'".format(fn))
    #     fns = [None]*nimgs
    #     coords = np.zeros((nimgs,ndims), dtype=np.double)
    #     with open(fn, "r") as f:
    #         for line in f:
    #             sline = line.strip()
    #             if not sline or sline[0]=='#': continue
    #             line = sline.split()
    #             imgno = int(line[0].split('_')[2])-1
    #             coords[imgno,:] = [float(x) for x in line[1:(ndims+1)]]
    #             fns[imgno] = line[0]
    #     return coords, fns

    @staticmethod
    def read_all_image_coords(fn, nimgs_per_mfov, nalloc=10000, ndims=2, cache_dn='', nmfovs=None,
            expect_mfov_subdir=False, param_dict=None):
        #print("read_all_image_coords '{}'".format(fn))
        if cache_dn:
            bfn = os.path.basename(fn)
            fnload = os.path.join(cache_dn, uuid.uuid4().hex + '_' + bfn)
            shutil.copyfile(fn, fnload)
        else:
            fnload = fn
        #print("filename in zimages.read_all_image_coords '{}'".format(fnload))
        fns = [[None]*nimgs_per_mfov for x in range(nalloc)]
        if ndims > 0:
            # nmfovs, tiles per mfov, ndims
            coords = np.empty((nalloc,nimgs_per_mfov,ndims), dtype=np.double); coords.fill(np.nan)
        cmfovs = 0
        out_param_dict = {}
        with open(fn, "r") as f:
            for line in f:
                sline = line.strip()
                if not sline: continue
                line = sline.split()
                if sline[0]=='#':
                    # check for params
                    strkey = line[0][1:]
                    if param_dict is not None and strkey in param_dict.keys():
                        out_param_dict[strkey] = np.array([param_dict[strkey](x) for x in line[1:]])
                    continue
                fpath = line[0].replace("\\","/") # replace windows path delimiters
                fn_parts = os.path.splitext(os.path.basename(fpath))[0].split('_')
                # xxx - xxxTHMBS, see main comment with same tag
                ind = 2 if fn_parts[0] == 'thumbnail' else 1
                mfovno = int(fn_parts[ind])-1
                imgno = int(fn_parts[ind+1])-1
                if ndims <= 0:
                    ndims = len(line[1:]); assert(ndims > 0) # no data in "coords" file?
                    coords = np.empty((nalloc,nimgs_per_mfov,ndims), dtype=np.double); coords.fill(np.nan)
                coords[mfovno, imgno,:] = [float(x) for x in line[1:(ndims+1)]]
                fpaths = fpath.split('/')
                # xxx - super hacky, add the mfov subdir if it is not in the coords file.
                #   workaround for mfov subdir missing in some image coords files in new acquisition format.
                if expect_mfov_subdir and len(fpaths) == 1:
                    #print('WARNING: missing mfov directory, prepending it based on filename')
                    fpaths = ['{:03d}'.format(mfovno+1)] + fpaths
                fns[mfovno][imgno] = os.path.join(*fpaths)
                if mfovno+1 > cmfovs:
                    cmfovs = mfovno+1
        if cache_dn: os.remove(fnload)
        if nmfovs is None: nmfovs = cmfovs
        if param_dict is not None:
            return coords[:nmfovs,:,:], fns[:nmfovs], out_param_dict
        else:
            return coords[:nmfovs,:,:], fns[:nmfovs]

    @staticmethod
    def read_metadata(fn):
        with open(fn, "r") as f:
            meta = {}
            for line in f:
                line = line.split(':')
                meta[line[0]] = ':'.join(line[1:])
        return meta

    @staticmethod
    def write_image_coords(fn, fns, ixyz, rmv_thb=False, add_thb=False, param_dict={}):
        #print("write_image_coords '{}'".format(fn))
        xyz = ixyz.copy(); xyz[np.logical_not(np.isfinite(xyz))] = 0
        if rmv_thb:
            fns = [None if x is None else os.path.join(os.path.dirname(x), re.sub('^thumbnail_','',
                    os.path.basename(x))) for x in fns]
        if add_thb:
            fns = [None if x is None else os.path.join(os.path.dirname(x),
                    'thumbnail_' + os.path.basename(x)) for x in fns]
        # some trouble overwriting these files on gpfs, xxx - never fully diagnosed
        if os.path.isfile(fn): os.remove(fn)
        with open(fn, 'w') as f:
            for i in range(len(fns)):
                if fns[i] is None: continue
                ostr = "%s" % (fns[i],)
                for j in range(xyz.shape[1]):
                    ostr += ("\t%a" % (xyz[i,j],))
                f.write(ostr + '\n')
            for k,v in param_dict.items():
                f.write('#' + k + ' ' + ' '.join(['{}'.format(x) for x in v]) + '\n')

    @staticmethod
    def get_s2ics_matrix(region_dir):
        """ read ic2Stage matrix from protocol.txt in region directory,
        convert to stage2ics matrix
        input: string
        output: numpy array"""

        protocol_file = os.path.join(region_dir, 'protocol.txt')
        with open(protocol_file, 'r') as pfile:
            for line in pfile:
                #fields = line.split(';')
                # some protocol.txt contain these pointless pipe characters
                fields = line.replace('|','').split(';')
                for field in fields:
                    subfields = field.split(':')
                    for cnt,subfield in zip(range(len(subfields)),subfields):
                        if 'ICS2Stage' in subfield:
                            # Zeiss modified the format of protocol.txt at some point...
                            try:
                                f = [float(x) for x in subfields[cnt+2].split() if x]
                            except:
                                f = [float(x) for x in subfields[cnt+1].split() if x]
                            return lin.inv(np.array(f).reshape((2,2)))

    @staticmethod
    def get_roi_coordinates(region_dir, coordinate_file=None, scl=1000, sep=';', cache_dn=''):
        """ get region ROI coordinates from region_stage_coords.csv
            output: ROI coordinates in stage coordinate system"""
        if coordinate_file is None:
            coordinate_file = os.path.join(region_dir, 'region_stage_coords.csv')
        fn = coordinate_file
        if cache_dn:
            bfn = os.path.basename(fn)
            fnload = os.path.join(cache_dn, uuid.uuid4().hex + '_' + bfn)
            shutil.copyfile(fn, fnload)
        else:
            fnload = fn
        region_coords = []
        with open(fnload, 'r') as cfile:
            for line in cfile:
                if line.strip():
                    region_coords.append([float(x.strip()) for x in line.split(sep)])
        if cache_dn: os.remove(fnload)
        # xxx - region_coords seem to be in microns not nm, is this stored somewhere?
        pts = np.array(region_coords)*scl
        # thanks Zeiss, remove duplicates
        # https://stackoverflow.com/questions/37839928/remove-following-duplicates-in-a-numpy-array
        pts = pts[np.insert((np.diff(pts,axis=0)!=0).any(1),0,True),:]
        # remove final point if repeat of first point (msem package assumes polygon points are closed).
        if (pts[0,:] == pts[-1,:]).all(): pts = pts[:-1,:]
        return pts

    @staticmethod
    def load_slice_balance_file(fn, region_slcstr=None):
        adj = np.zeros((0,), dtype=np.double) if region_slcstr is None else 0.
        if os.path.isfile(fn):
            with open(fn, 'r') as f:
                for line in f:
                    line = line.split()
                    if len(line) > 1:
                        tmp = line[0].split('_')
                        if len(tmp) > 1:
                            if region_slcstr is None:
                                adj = np.concatenate((adj, [float(line[1])]))
                            elif tmp[-1] == region_slcstr:
                                adj = float(line[1]); break
        return adj

    @classmethod
    def zimages_args(cls, args):
        return cls(args.image_coordinates, args.mFoV_path, verbose=args.zimages_verbose)

    @staticmethod
    def addArgs(p):
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--image-coordinates', nargs=1, type=str, default='', help='Input coordinates text file')
        p.add_argument('--mFoV-path', nargs=1, type=str, default='', help='Path to the high-res MFoV images')
        p.add_argument('--zimages-verbose', action='store_true', help='Verbose output')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read image coordinates and Zeiss high-res images and montage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    zimages.addArgs(parser)
    args = parser.parse_args()

    # xxx - what does command line interface do for this object?
    dispMFoV = zimages.zimages_args(args)
