"""mfov.py

Class representation and alignment / stitching procedure for single Zeiss 
  multi-SEM fields of view (MFoVs).

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

# xxx - major improvement in readability possible here by change the types in the adjacency matrix to be consistent
#   with the directions in the hex grid. this is what requires the even/odd row adjustments in the crops.
#   possible this could prevent the adjacency looping entirely by having fixed indices for the hex directions?

import numpy as np
import os
import time
import re
#import glob
#import sys

from contextlib import ExitStack

import logging
logger = logging.getLogger(__name__)

#import scipy.linalg as lin
#import scipy.spatial.distance as scidist
import scipy
import scipy.sparse as sp

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing
# xxx - failures occured during solving for region brightness... some way to do a conditional import here?
#from sklearnex import linear_model
#from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

from .zimages import zimages
from ._template_match import template_match_preproc, normxcorr2_fft_adj, normxcorr2_adj_to_comps
from .utils import FFT_types, create_scipy_fft_context_manager

try:
    # xxx - this breaks something with logging, did not feel like figuring it out
    ## do not want to see the long cupy error every time
    ## https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    from rcc_xcorr.xcorr import BatchXCorr
except:
    print('WARNING: rcc-xcorr unavailable, needed for one method for template matching with cross-correlations')

class mfov(zimages):
    """Zeiss mSEM field of view (MFoV) object.

    For loading / stitching images in single mfov with support for boundary images from neighbors.

    .. note::


    """

    ### fixed parameters not exposed

    # set > 1 to crop out appropriate portion of the template image
    template_crop_factor = 2
    # set > 1 to true to also crop the target image (if template_crop is true)
    image_crop_factor = 1

    # for the least square weightings, two values for cutoffs and weights:
    #   first for image comparisons within the same mfov.
    #   second for image comparisons for tiles in different mfovs.
    #C_cutoff = [-1., -1.] # for off
    C_cutoff = [0., 0.] # basically off, only discards really bad correlations
    #C_cutoff = [0.1, 0.1] # reasonable hard cutoff

    # <<< these are now just defaults, exposed in def_common_params
    # previous cutoff until issues found with robin dataset
    #C_cutoff = [0.1, 0.1] # reasonable hard threshold for spurious correlations if preproc params are appropriate
    # set cutoff for discarding deltas that are out of expected range, specify in nm relative to "tiled" deltas.
    # there are two sets of delta cutoffs, for horizontal adjacencies and for vertical/diagonal adjacencies.
    D_cutoff = [[np.array([np.inf, np.inf]), np.array([np.inf, np.inf])] for x in range(2)] # for off
    #D_cutoff = [[np.array([0., 0.]), np.array([0., 0.])] for x in range(2)]  # all the way on, uses zeiss alignments
    ## previous cutuff until issues found with robin dataset
    ##D_cutoff = [[np.array([1024,64]), np.array([1024,1024])], # was using this for some time
    ##            [np.array([12000, 10000]), np.array([12000, 10000])]] # can be large overlap at mfov boundaries
    # D_cutoff = [[np.array([1024,128]), np.array([1024,1024])],
    #             [np.array([15000, 13000]), np.array([15000, 13000])]] # can be large overlap at mfov boundaries
    # what weighting to put on the default (Zeiss) coordinates for discarded correlations and deltas
    #   and also on non-discarded comparisons between mfovs (vs withing mfovs).
    # description by index:
    #   0 - weight for default delta (that replaces discarded delta) for within mfov comparisons
    #   1 - weight for default delta (that replaces discarded delta) for between mfov comparisons
    #   2 - weight for non-discarded delta for between mfov comparisons (relative to within mfov comparisons)
    #W_default = [1., 1., 1.] # everything has equal weighting relative to valid within mfov comparisons
    W_default = [0.1, 0.01, 1.] # used until issues with robin dataset
    # ignore correlations for images with low complexity based on variance.
    # be careful with this parameter, any image tiles below the cutoff rely on the zeiss coordinates entirely.
    V_cutoff = 0. # for off
    #V_cutoff = 1e-5 # essentially off, but still computes image variances
    ## previous cutoff until issues found with robin dataset
    ##V_cutoff = 196.
    # these are now just defaults, exposed in def_common_params >>>

    # this is another feature in addition to the "hard" lower bound for the correlations.
    # try and estimate the "lower bound" of the correlation peak.
    C_cutoff_soft_nGMM = 0 # to disable the soft cutoff
    # parameters used by the soft cutoff when C_cutoff_soft_nGMM > 0
    #C_cutoff_soft_nGMM = 5 # good choice for enabling the soft cutoff
    #C_cutoff_soft_rng = [0.25, 0.5] # if enabled, look for soft max peak in this range, template_crop_factor 3
    C_cutoff_soft_rng = [0.2, 0.4] # if enabled, look for soft max peak in this range, template_crop_factor 2
    C_cutoff_soft_nstds = 2.6 # number of stds below the fitted peak to take as soft cutoff

    # image processing during load
    # optional crop off border of image tiles, this crop is in the image space, so xy are flipped.
    # crops symmetrically from both "sides" of each dimension, xxx - any need to specify each edge individually?
    border_crop = [0,0] # for off
    ## xxx - not clear if this was really that useful and not necessarily a good idea for "pre-downsampled" data.
    ## from Tomasc at Zeiss, there is a problem with the first scan (top line of image), so discard.
    ##   the acquisition is not sync'ed with the scan speed until about half way through the first line.
    ## xxx - some need to make the crops 4 length array? currently discards the top and the bottom lines
    ##border_crop = [1,0]

    # whether to measure correlations in both directions for neighboring tiles
    symmetric_adj = True

    # whether to invert the images or not as they are loaded
    invert_images = True

    # these are for creating plots for presentation only. should normally be set to defaults.
    scale_tiled_coords_factor = 1. # default
    # to leave some space between images in image_tiled (non-aligned version in hex-arrangement)
    #scale_tiled_coords_factor = 1.05
    montage_background = 0 # default
    # set a white background on the images montaged for this mfov.
    #montage_background = 255

    # subfolder where slices are stored in the new msem format
    fullres_dir = 'fullres'

    # try to set resonable xcorr preprocessing defaults based on the pixel size.
    # if these do not work, they can also be overridden manually after init.
    # image processing before xcorr, parameters are dependent on downsampling amount.
    # whitening is essential for non-spurious xcorrs, clahe typically helps.
    def set_xcorr_preproc_params(self, off=False):
        pixel_scale_ds = self.scale_nm * self.dsstep
        if off:
            self._proc_filter_size = 0
            self._proc_whiten_sigma = 0. # for off
            self._proc_clahe_clipLimit = 0. # for off
            self._proc_clahe_tileGridSize = (2,2)
        elif pixel_scale_ds <= 5:
            self._proc_filter_size = 5
            self._proc_whiten_sigma = 5.
            self._proc_clahe_clipLimit = 30.
            self._proc_clahe_tileGridSize = (32,32)
        elif pixel_scale_ds <= 17:
            self._proc_filter_size = 0
            self._proc_whiten_sigma = 4.
            self._proc_clahe_clipLimit = 30.
            self._proc_clahe_tileGridSize = (32,32)
        elif pixel_scale_ds <= 33:
            self._proc_filter_size = 0
            self._proc_whiten_sigma = 2.
            self._proc_clahe_clipLimit = 20.
            self._proc_clahe_tileGridSize = (16,16)
        else:
            self._proc_filter_size = 0
            self._proc_whiten_sigma = 1.
            self._proc_clahe_clipLimit = 20.
            self._proc_clahe_tileGridSize = (8,8)

    def __init__(self, experiment_folders, region_strs, region_ind, mfov_id, dsstep=1, overlap_radius=2,
                 overlap_correction_borders=None, init_region_coords=True, region_coords=None, region_filenames=None,
                 mfov_tri=None, mfov_meta=None, false_color_montage=False, use_thumbnails_ds=8,
                 thumbnail_folders=[], D_cutoff=None, V_cutoff=None, W_default=None, legacy_zen_format=False,
                 nimages_per_mfov=None, scale_nm=None, C_cutoff_soft_nGMM=0, verbose=False):

        zimages.__init__(self)
        # xxx - just completely disable the cache'ing for now, maybe not super useful?
        #   doing it here as kindof a hacky way to still allow for the "cache clearing" in zimages init
        self.cache_dir = ''

        self.mfov_verbose = verbose
        self.legacy_zen_format = legacy_zen_format
        assert(legacy_zen_format or nimages_per_mfov is not None)
        self.nimages_per_mfov = nimages_per_mfov

        # several options here for specifying which region we are in.
        assert(region_strs is not None) # legacy glob'ing mode is no longer supported
        if isinstance(experiment_folders, (list, tuple)):
            assert(region_ind > 0) # region_ind must be positive if region_strs is list
            # interpret region_ind as a base-1 index as typically slices are parallel to image order.
            self.region_ind = region_ind-1
            # this is a new mode where all of the region strings are specified in order from a top-level file
            #   and the ind just represents the index to use in this list.
            clens = np.cumsum(np.array([len(x) for x in region_strs]))
            exp_ind = np.nonzero(self.region_ind / clens < 1.)[0][0]
            exp_region_ind = self.region_ind - (clens[exp_ind-1] if exp_ind > 0 else 0)
            self.experimental_ind = exp_ind

            self.experiment_folder = experiment_folders[exp_ind]
            self.thumbnail_folder = thumbnail_folders[exp_ind] if len(thumbnail_folders) > 0 else thumbnail_folders
            if not self.legacy_zen_format:
                self.experiment_folder = os.path.join(self.experiment_folder, self.fullres_dir)
                self.thumbnail_folder = os.path.join(self.thumbnail_folder, self.fullres_dir)
            # xxx - this glob'ing is a throwback to before the manifest, keeping for reference, probably delete
            # tmp = glob.glob(os.path.join(self.experiment_folder, '*_' + region_strs[exp_ind][exp_region_ind]))
            # if len(tmp) != 1:
            #     logger.debug('found %d folders matching %s in folder %s',
            #         len(tmp), region_strs[exp_ind][exp_region_ind], self.experiment_folder)
            self.region_folder = os.path.join(self.experiment_folder, region_strs[exp_ind][exp_region_ind])
        else:
            assert(region_ind >= 0) # region_ind must be non-negative for fully specified region_str
            # this is a new mode where the experiment folder, region_str and thumbnail folder (if specified)
            #   have already been determined, and are passed in as strings.
            # that is, the path to the region is fully specified.
            self.experiment_folder = experiment_folders
            self.region_folder = os.path.join(self.experiment_folder,region_strs)
            self.thumbnail_folder = thumbnail_folders
            # in the fully specified mode, take region_ind as it was specified (region already determined).
            self.region_ind = region_ind
        assert(os.path.isdir(self.region_folder)) # causes even more confusing problems later, should exist

        # convert mfov_id from zeiss numbering starting at 1 to numbering starting at 0
        # mfovs (unlike regions) are always specified as integers which correspond to the imaging order.
        self.mfov_id = mfov_id-1
        self.mfov_str = ('{:06d}' if self.legacy_zen_format else '{:03d}').format(self.mfov_id+1)
        self.mfov_folder = os.path.join(self.region_folder, self.mfov_str)

        # get the Zeiss slice and region number part of the region_folder.
        # typically regions are named ???_S%dR%d where ? is integer imaging order (region_numstr)
        #   and S and R are Zeiss slice and region number (region_slcstr).
        self.region_str = os.path.split(self.region_folder)[1]
        tmp = self.region_str.split('_')
        if len(tmp) > 1:
            self.region_numstr = tmp[0]; self.region_slcstr =  tmp[1]
        else:
            self.region_numstr = self.region_slcstr = tmp[0]
        logger.debug("mfov _ind %d, region _ind %d, _str '%s', _numstr '%s', _slcstr '%s'",
            self.mfov_id, self.region_ind, self.region_str, self.region_numstr, self.region_slcstr)

        # parameter serves double duty, > 0 indicates on and if on it represents the ds amount
        self.use_thumbnails_ds = use_thumbnails_ds
        thumbnails_ds = use_thumbnails_ds if use_thumbnails_ds > 0 else 1
        # load images from the load_folder, which points at the thumbnails folder
        #   if this folder was given and if thumnbnails were requested.
        if not self.thumbnail_folder: self.thumbnail_folder = ''
        if self.use_thumbnails_ds and self.thumbnail_folder:
            self.images_load_folder = os.path.join(self.thumbnail_folder,self.region_str)
        else:
            self.images_load_folder = self.region_folder

        logger.debug('self.experiment_folder: %s', self.experiment_folder)
        logger.debug('self.region_folder: %s', self.region_folder)
        logger.debug('self.thumbnail_folder: %s', self.thumbnail_folder)
        logger.debug('self.images_load_folder: %s', self.images_load_folder)

        # generate stitched images using the false coloring for visualizing stitching
        self.false_color_montage = false_color_montage

        # enable fitting of sigmoid in border overlap areas to try to correct "overlap darkening" without filtering
        # which borders to correct for in array are in order: top, bottom, left, right
        if overlap_correction_borders is None:
            self.overlap_correction_borders = np.zeros((4,), dtype=bool)
            self.overlap_correction = False
        else:
            self.overlap_correction_borders = np.array(overlap_correction_borders, dtype=bool)
            self.overlap_correction = self.overlap_correction_borders.any()

        if self.nimages_per_mfov is None:
            # NOTE: just enumerating the images is dangerous... could be thumbnails or other images
            #   and we're not totally sure what the format is without the coords file.
            # set self.imfov_diameter based on how many image files are in the coords file
            fn = os.path.join(self.mfov_folder, "image_coordinates.txt"); mfov_count=0
            if os.path.isfile(fn):
                with open(fn, 'r') as f:
                    for line in f:
                        if re.match("[0-9]{3}.*(tif|bmp)", line):
                            mfov_count += 1
        else:
            # NOTE: in the new acquisition format many mfovs do not have image_coordinates.txt
            #   decided it was no longer worth doing this automatically, but just specify as parameter.
            # One could also parse full_image_coordinates for the slice/region but this can be a very large
            #   file and did not really want to add another heavyweight text parse here (this can significantly
            #   increase the init time).
            mfov_count = self.nimages_per_mfov
        logger.debug('number of tiles in mFOV: %s', mfov_count)
        self.imfov_diameter = mfov.mfov_diameter_from_ntiles(mfov_count)
        if self.imfov_diameter <= 1:
            # do not complain about empty mfovs so that empty regions can be handled.
            if mfov_count > 0:
                logger.critical("unexpected number of mFOV tiles: %s", mfov_count)
                assert(False) # non-zero number of image tiles that is not a hex tiling number
            return # make empty mfovs (regions) a graceful error
        # code can theoretically handle other diameters but this is not extensively tested.
        assert (self.imfov_diameter == 9 or self.imfov_diameter == 11) # unexpected for current zeiss setup

        # amount of downsampling to apply to image tiles just after loading
        self.dsstep = dsstep

        # the diameter to use for stitching to include neighboring MFoVs
        self.overlap_diameter = 2*overlap_radius
        self.omfov_diameter = self.imfov_diameter + self.overlap_diameter
        assert( self.overlap_diameter >= 0 ) # need non-negative overlap_radius
        # code does not work for overlap beyond hex-adjacent neighbors
        assert( self.omfov_diameter <= 2*self.imfov_diameter )


        # number of tiles (images) in the actual hexagonal MFoV layout, i.e., in a single mfov.
        # this is essentially fixed, but configurable in the case of number of beams in msem changes.
        self.niTiles = mfov.ntiles_from_mfov_diamter(self.imfov_diameter)
        self.niTilesRect = self.imfov_diameter*self.imfov_diameter

        # number of tiles (images) in the actual hexagonal layout that contains overlapping tiles with neighbors.
        self.nTiles = mfov.ntiles_from_mfov_diamter(self.omfov_diameter)
        # number of tiles in the square region inscribed by the hexagon tiles
        self.nTilesRect = self.omfov_diameter*self.omfov_diameter

        # loading all region coords for a whole wafer, for example, is very costly (lots of small text file io),
        #   so added an option to disable in some situations (like when wafer is instantiated for the wafer_solver).
        if not init_region_coords:
            self.nmfovs = 1 # as a "non-error" placeholder
            self.region_coords, self.region_filenames, self.mfov_tri = None, None, None
        else:
            # avoid reloading coordinates or recalcuting region triangulation,
            #   if already done for another mfov in this region.
            if region_coords is None:
                # read in the zeiss coordinates and filenames for the entire region.
                fn = os.path.join(self.region_folder, "full_image_coordinates.txt")
                if os.path.isfile(fn):
                    self.region_coords, self.region_filenames = zimages.read_all_image_coords(fn, self.niTiles,
                            cache_dn=self.cache_dir, expect_mfov_subdir=True)
                else:
                    # if no coords file is available in the region, then use the mfov coordinates file
                    assert(False) # do not use mfov image_coordinates... revalidate if we need it again
                    # fn = os.path.join(self.mfov_folder, "image_coordinates.txt")
                    # assert( os.path.isfile(fn) ) # can not find any image coordinates file
                    # n = self.mfov_id+1; ndims=2
                    # self.region_filenames = [[None]*self.niTiles for x in range(n)]
                    # self.region_coords = np.empty((n,self.niTiles,ndims), dtype=np.double)
                    # self.region_coords.fill(np.nan)
                    # coords, fns = zimages.read_image_coords(fn, self.niTiles, ndims=ndims)
                    # for i in range(self.niTiles):
                    #     self.region_filenames[self.mfov_id][i] = os.path.join(self.mfov_str, fns[i])
                    # self.region_coords[self.mfov_id,:,:] = coords
                self.mfov_tri = None
                # if this value is greater than zero, it specifies to use the thumbnails.
                # the value is the factor by which the thumbnails were downsampled.
                if use_thumbnails_ds > 0:
                    self.region_filenames = [[os.path.join(os.path.dirname(x), 'thumbnail_' + os.path.basename(x)) \
                                                           for x in y] for y in self.region_filenames]
                    self.region_coords /= thumbnails_ds
                    logger.debug('using pre-saved thumbnails with ds factor of %d', use_thumbnails_ds)
            else:
                # xxx - do not reload coordinates or recalculate triangulation if already done for another mfov
                self.region_coords, self.region_filenames, self.mfov_tri = region_coords, region_filenames, mfov_tri
            self.nmfovs = len(self.region_filenames)
            self.get_mfov_neighbors()

        # load metadata text file for this mfov.
        if mfov_meta is None:
            # only ended up using meta for the resolution, so if scale is specified
            #   then do not bother to load metadata.
            if scale_nm is None:
                # xxx - if necessary, need to modify this to read the new acquisition metadata.txt
                #   for now for any new acquisition, just specify the resolution as a param.
                self.mfov_meta = zimages.read_metadata(os.path.join(self.mfov_folder,'metadata.txt'))
            else:
                self.mfov_meta = None
        else:
            self.mfov_meta = mfov_meta
        if self.mfov_meta is not None:
            self.native_scale_nm = float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                                             self.mfov_meta['Pixelsize'])[0])
        else:
            assert( scale_nm is not None )
            self.native_scale_nm = scale_nm
        self.scale_nm = self.native_scale_nm * thumbnails_ds

        # compute hexagonal layout for zeiss tiling, along with hexagonal coordinates and ring sizes
        o = self.omfov_diameter; i = self.imfov_diameter
        coords_list, ntiles, coords, hex_to_rect, rect_to_hex = mfov.tileCoords_mfov(o, order='C')
        from_neighbor, to_neighbor = mfov.get_overlapping_tiles(i//2, o//2, ntiles, np.cumsum(ntiles))
        # create a select for inner and outer diameter tiles
        is_inner = np.zeros((self.nTiles,), dtype=bool); is_inner[:self.niTiles] = 1
        is_inner_rect = np.zeros((self.nTilesRect,), dtype=bool); is_inner_rect[hex_to_rect] = is_inner
        is_outer = np.zeros((self.nTiles,), dtype=bool); is_outer[self.niTiles:] = 1
        is_outer_rect = np.zeros((self.nTilesRect,), dtype=bool); is_outer_rect[hex_to_rect] = is_outer
        self.ring = {'coords_list':coords_list, 'ntiles':ntiles, 'coords':coords, 'hex_to_rect':hex_to_rect,
                     'rect_to_hex':rect_to_hex, 'from_neighbor':from_neighbor, 'to_neighbor':to_neighbor,
                     'is_inner':is_inner, 'is_inner_rect':is_inner_rect,
                     'is_outer':is_outer, 'is_outer_rect':is_outer_rect}

        # create mappings for the within mfov images only
        coords_list, ntiles, coords, hex_to_rect, rect_to_hex = mfov.tileCoords_mfov(i, order='C')
        self.iring = {'coords_list':coords_list, 'ntiles':ntiles, 'coords':coords, 'hex_to_rect':hex_to_rect,
                      'rect_to_hex':rect_to_hex}

        # calculate D_cutoff in pixels
        if D_cutoff is None: D_cutoff = self.D_cutoff
        # xxx - hacky, specify negative value to not scale
        if any([any([(x < 0).any() for x in y]) for y in D_cutoff]):
            self.D_cutoff = [x*np.sign(x) for x in D_cutoff]
        else:
            self.D_cutoff = [[x / self.scale_nm / self.dsstep for x in D_cutoff[y]] for y in range(2)]

        # save the image variance cutoff
        if V_cutoff is not None: self.V_cutoff = V_cutoff

        # save the default weight if specified
        if W_default is not None: self.W_default = W_default

        # to enable the soft C cutoff method, specify > 0
        self.C_cutoff_soft_nGMM = C_cutoff_soft_nGMM

        # try to set reasonable defaults for the xcorr preprocessing
        self.set_xcorr_preproc_params()

        # some member inits
        self.images = None
        self.image_adjust_rect = None
        self.mfov_filenames = None
        self.export_xcorr_comps_path = None

    # see documentation regarding how mfovs are laid out and the total number of tiles in each mfov.
    @staticmethod
    def ntiles_from_mfov_diamter(d):
        assert( d % 2 == 1 and d > 1 ) # xxx - nothing works with even ring diameter, or only a single tile per mfov
        # formula for the symmetric hexagon
        return int((3*d*d + 1)/4)

    @staticmethod
    def mfov_diameter_from_ntiles(n):
        d2 = (4*n - 1) / 3; d = np.sqrt(d2) if d2 > 0 else 0; dint = int(d)
        # return 0 if number of tiles is not a number that corresponds to an odd diameter hexagonal ring
        return dint if ( d2 == int(d2) and d == dint ) else 0

    # compute angle in zeiss image coordinate space
    @staticmethod
    def get_coords_angles(pt_from, pt_to):
        v = pt_to - pt_from
        a = np.arctan2(-v[:,1], v[:,0]) if v.ndim==2 else np.array(np.arctan2(-v[1], v[0]))
        sel = (a < 0); a[sel] = 2*np.pi+a[sel]
        return a

    @staticmethod
    def tileCoords_at_radius_mfov(r):
        if r==0:
            return np.zeros((1,2), dtype=np.double)

        ri = r; r = float(r)
        # coords go in order in a hexagonal ring at each radius.
        # comments describe where each line segment of the hexagonal edges are generated (6 total line segments).
        # each line segment is inclusive of the first point but not the end point (starts with next segment).
        #print(np.arange(-r + 0.5*r - 1, -r, -1), np.arange(r,0,-1))
        coords = np.hstack((
            # quadrant-I diagonal
            np.vstack(( np.arange(r, r - 0.5*r, -0.5), np.arange(r), )),
            # top
            np.vstack(( np.arange(r - 0.5*r, -r + 0.5*r, -1), np.ones((ri,),dtype=np.double)*r, )),
            # quadrant-II diagonal
            np.vstack(( np.arange(-r + 0.5*r, -r, -0.5), np.arange(r,0,-1), )),
            # quadrant-III diagonal
            np.vstack(( np.arange(-r, -r + 0.5*r, 0.5), np.arange(0,-r,-1), )),
            # bottom
            np.vstack(( np.arange(-r + 0.5*r, r - 0.5*r, 1), np.ones((ri,),dtype=np.double)*-r, )),
            # quadrant-IV diagonal
            np.vstack(( np.arange(r - 0.5*r, r, 0.5), np.arange(-r,0,1), )),
        )).T

        # xxx - probably could fix this above, seems more intuitive in the normal counterclockwise angle direction.
        #   y direction is flipped in the image space.
        coords[:,1] = -coords[:,1]

        return coords

    @staticmethod
    def tileCoords_mfov(d, order='F'):
        r = d//2; ringCoord = [None]*(r+1); ringCount = np.zeros((r+1,), dtype=np.int64)
        for i in range(r+1):
            ringCoord[i] = mfov.tileCoords_at_radius_mfov(i)
            ringCount[i] = ringCoord[i].shape[0]

        # as a single concatenated array
        ringCoordCat = np.vstack(ringCoord)

        # convert to a single array of unraveled matrix coordinates for the enclosing square, top row "left shifted"
        # this creates a mapping array from Zeiss "polar hex coordinates" to rectangular F-order or C-order coordinates.
        hexToRect = np.ravel_multi_index(np.vstack([np.floor(x).astype(np.int64) + d//2 for x in ringCoord]).T,
                                         (d,d), order=order)

        # also create the inverse mapping from rectangular indices to hex indices
        rectToHex = -np.ones((d*d,), dtype=np.int64); nHex = ringCoordCat.shape[0]; nRect = d*d
        rectToHex[hexToRect] = np.arange(0,nHex,dtype=np.int64)
        rectToHex[rectToHex < 0] = np.arange(nHex,nRect,dtype=np.int64)

        return ringCoord, ringCount, ringCoordCat, hexToRect, rectToHex

    # gets the indices of tiles that overlap between mfovs for specified inner and outer hexagonal radius.
    @staticmethod
    def get_overlapping_tiles(irad, orad, nTilesRing, nTilesRingCum):
        # get the tile indices for the tiles that are to be loaded outside of this mfov
        #overlap_diameter = self.omfov_diameter - self.imfov_diameter
        drad = orad - irad
        to_neighbor_tiles = [None]*drad; from_neighbor_tiles = [None]*drad; rTiles = [None]*drad
        for i in range(drad):
            # get the hex indices of the "edge" tiles that are outside the inner mfov.
            rTiles[i] = nTilesRingCum[irad-i-1] + np.arange(nTilesRing[irad-i])
            # divide the tiles into the six hexagonal edges.
            tmp = rTiles[i].reshape(-1,nTilesRing[irad-i]//6)
            # repeat the hex corners to create a list of tiles that comprise each hex edge
            tmp = np.roll(np.hstack((tmp[:,0][:,None], tmp)).flat[:],-1).reshape(6,-1)
            # the boundary tiles are actually slices along hex directions, so include tiles from multiple rings
            #   if we're beyond the first boundary ring.
            for j in range(i):
                crad = irad+j+1-i
                tmp = np.concatenate((np.roll(rTiles[i-j-1][::-1].flat[j::crad][::-1],1)[:,None], tmp,
                                      np.roll(rTiles[i-j-1].flat[(j+1)::crad],-1)[:,None]), axis=1)
            # create the corresponding tiles from the neighbor mfov point of view
            from_neighbor_tiles[i] = tmp[(np.arange(6)-3)%6,:][:,::-1]

            # now create the tile numbers from the current mfov perspective, which are the rings after the mfov diameter
            to_neighbor_tiles[i] = (nTilesRingCum[irad+i] + np.arange(nTilesRing[irad+i+1])).\
                reshape(-1,nTilesRing[irad+i+1]//6)

        return from_neighbor_tiles, to_neighbor_tiles

    # returns adjacency matrix for a hex connectivity.
    # grid is created by either shifting every other row of a 2D rectangular cartesian grid.
    # the top row and every other row is shifted by half a coordinate.
    # the shift direction of the top row is left by default, unless shift_right is specified.
    @staticmethod
    def adjacency_matrix_hex(sz, shift_right=False, with_type=False, diag=False):
        # for testing
        #full_adj = adjacency_matrix_hex((4,5)) # test
        #np.set_printoptions(threshold=np.nan, linewidth=320); print(full_adj); #print(hex_to_rect_inds)

        c,r=sz # adj supported here only in F-order
        # modified from:
        #   % https://stackoverflow.com/questions/3277541/construct-adjacency-matrix-in-matlab
        # Make the first diagonal vector (for horizontal connections)
        diagVec1 = np.tile(np.concatenate((np.ones((c-1,),dtype=np.int64), [0])),r)
        diagVec1 = diagVec1[0:-1]                      # Remove the last value
        # Make the second diagonal vector (for anti-diagonal connections that skip every other row)
        tmp = np.ones((c-1,),dtype=np.int64); tmp[1::2] = 0
        diagVec2 = np.concatenate((np.tile(np.concatenate(([0], tmp)), r-1), [0]))
        diagVec3 = np.ones((c*(r-1),),dtype=np.int64)  # Make the third diagonal vector (for vertical connections)
        # Make the fourth diagonal vector (for diagonal connections that skip every other row)
        diagVec4 = diagVec2[0:-2]
        # Add the diagonals to a zero matrix and optionally include the type
        if with_type:
            adj = np.diag(diagVec1,1) + 2*np.diag(diagVec3,c) + 3*np.diag(diagVec2,c-1) + 4*np.diag(diagVec4,c+1)
        else:
            adj = np.diag(diagVec1,1) + np.diag(diagVec3,c) + np.diag(diagVec2,c-1) + np.diag(diagVec4,c+1)
        adj += adj.T # this is not just to make the adj symmetric, needed here to even create it properly
        if shift_right: adj = adj[::-1,::-1]
        # diagonal not intended to be used for adjacency, this was used for testing cropping / correlations.
        if diag: adj = adj + 5*np.eye(adj.shape[0])
        return adj

    # helper method for solve_stitching
    @staticmethod
    def _solver_remove_bias(crds, sel, regr_bias, poly, clf):
        if not regr_bias:
            # this is the old code that just removes the mean value.
            # from the calling code, the select is either known beforehand (valid indices for example),
            #   or is a select over the current component in the adjacency matrix.
            if crds.ndim==1:
                crds[sel] = crds[sel] - crds[sel].min()
                crds[sel] = crds[sel] - crds[sel].mean()
            else:
                crds[sel,:] = crds[sel,:] - crds[sel,:].min(0)
                crds[sel,:] = crds[sel,:] - crds[sel,:].mean(0)
        else:
            ncrds = crds[sel][:,None] if crds.ndim==1 else crds[sel,:]
            npts, ndims = ncrds.shape
            # NOTE: there is really no knowledge here of where the "original points" are that were used for
            #   computing the deltas. Probably the adjacency matrix could be used to create a generic nd
            #   coordinate system to use as the X points, but could not think of a use case for this that makes sense.
            #   If there was a problem with an accumulating bias in the 2D alignment, this could be useful,
            #     but so far this has not been observed.
            # So, currently this only makes sense for *index ordered* 1D data (i.e., the 3D alignment deltas).
            # Another option could be to convert back to the deltas (using the adj matrix), and then remove
            #   any bias in the deltas (like remove the mean of the differential), but then how to convert back
            #   to the regular deltas again?
            # NOTE: deltas are still 2d, so removed this assert. This is a buyer beware assumption, the points
            #   MUST be index ordered in their relation. If not this is going to produce weird results.
            #assert(ndims==1) # need knowledge of the indexing for this. read comments above this assert.
            x = np.arange(npts)[:,None]; X = poly.fit_transform(x)
            clf.fit(X, ncrds)
            crds[sel] = crds[sel] - clf.predict(X)

    # implement "kevin's formula" least squares solver.
    # in the documentation this is now referred to as the "emalign solver".
    @staticmethod
    def solve_stitching(adj_matrix, Dx,Dy=None, W=None, C=None,C_cutoff=None, Dx_z=None,Dy_z=None, Dx_t=None,Dy_t=None,
                        D_cutoff=None, W_default=None, adj_matrix_bmfov=None, center_sel=None, label_adj_min=-1,
                        l1_alpha=0., l2_alpha=0., adj_nonbinary={}, return_inds_subs=False, return_comps_sel=False,
                        return_deltas=False, default_tol=None, regr_bias=False, verbose=False):
        nimgs = adj_matrix.shape[0]
        nadj = adj_matrix.count_nonzero() if sp.issparse(adj_matrix) else np.count_nonzero(adj_matrix)

        # just for reference:
        # (weights x adj_diff) x global_coords = (weights x delta_coords)
        #global_coords = np.zeros((nimgs,ndims), dtype=np.double)
        #delta_coords = np.zeros((nadj,ndims), dtype=np.double)
        #weights = np.zeros((nadj,nadj), dtype=np.double)
        #adj_diff = np.zeros((nadj,nimgs), dtype=np.int) - +1 and -1 per row for image positions being compared
        #ndims = currently only 1 or 2 supported (theoretically works for higher dims)
        # xxx - support 3 or higher ndims? currently not necessary for msem package

        # get the subscripts and indices corresponding to the non-zeros in the adjacency, correlation and delta matrices
        subs = adj_matrix.nonzero(); inds = np.ravel_multi_index(subs, (nimgs,nimgs)); subs = np.transpose(subs)

        # use the indices to set the entries in the "image diff" matrix, size nadj by nimgs.
        # this assumes that the correlation matrix calculated the offsets using row image relative to column image.
        adj_diff = np.zeros((nadj,nimgs), dtype=np.double)
        i = np.arange(nadj, dtype=np.int64)
        inds_pos = np.ravel_multi_index((i,subs[:,0]), (nadj,nimgs))
        inds_neg = np.ravel_multi_index((i,subs[:,1]), (nadj,nimgs))
        # put negative first so that any regularization terms are positive.
        adj_diff.flat[inds_neg] = -1; adj_diff.flat[inds_pos] = 1

        if sp.issparse(Dx):
            Dxinds = Dx[subs[:,0], subs[:,1]]
            Dxinds = Dxinds.todense().A1 if sp.issparse(Dxinds) else Dxinds.A1
            if Dy is None:
                Dyinds = None
            else:
                Dyinds = Dy[subs[:,0], subs[:,1]]
                Dyinds = Dyinds.todense().A1 if sp.issparse(Dyinds) else Dyinds.A1
        else:
            Dxinds = Dx.flat[inds]
            Dyinds = None if Dy is None else Dy.flat[inds]

        if Dy is None:
            delta_coords = Dxinds.astype(np.double)[:,None]
        else:
            # use the indices to create the raveled delta matrix, size nadj by 2
            delta_coords = np.hstack((Dxinds.astype(np.double)[:,None], Dyinds.astype(np.double)[:,None]))
        if return_deltas: delta_coords_orig = delta_coords
        ndims = delta_coords.shape[1]

        Winds_ind = None
        if W is not None:
            assert( C is None ) # specify either C or W
            if sp.issparse(W):
                Winds = W[subs[:,0], subs[:,1]]
                Winds = Winds.todense().A1 if sp.issparse(Winds) else Winds.A1
            else:
                Winds = W.flat[inds]
        else:
            # weighting vector for the least-squares solution
            Winds = np.ones((inds.size,), dtype=np.double)

            # xxx - refactor this code so that the weights are sent in externally.
            #   this doesn't really belong here. it's very specific to the 2d alignment use case.
            if C is not None:
                if sp.issparse(C):
                    Cinds = C[subs[:,0], subs[:,1]]
                    Cinds = Cinds.todense().A1 if sp.issparse(Cinds) else Cinds.A1
                else:
                    Cinds = C.flat[inds]
                # Dx_z and Dy_z are the default deltas for outliers.
                # outliers are detected based on hard delta and correlation value cutoffs.
                if sp.issparse(Dx_z):
                    if Dx_z is None:
                        Dx_zinds = None
                    else:
                        Dx_zinds = Dx_z[subs[:,0], subs[:,1]]
                        Dx_zinds = Dx_zinds.todense().A1 if sp.issparse(Dx_zinds) else Dx_zinds.A1
                    if Dy_z is None:
                        Dy_zinds = None
                    else:
                        Dy_zinds = Dy_z[subs[:,0], subs[:,1]]
                        Dy_zinds = Dy_zinds.todense().A1 if sp.issparse(Dy_zinds) else Dy_zinds.A1
                else:
                    Dx_zinds = None if Dx_z is None else Dx_z.flat[inds]
                    Dy_zinds = None if Dy_z is None else Dy_z.flat[inds]

                if adj_matrix_bmfov is not None:
                    Winds_ind = np.zeros((inds.size,), dtype=np.int64)
                    if return_deltas:
                        delta_coords_orig = delta_coords.copy()

                    # get selects for vertical and horizontal connections
                    hsubs = np.nonzero(adj_matrix==2); hinds = np.ravel_multi_index(hsubs, (nimgs,nimgs))
                    hsel = np.in1d(inds, hinds); vsel = np.logical_not(hsel)

                    # xxx - depends on what other adjacency matrix is passed in exactly what these mean
                    #   need to clean this up once we've settled on something.
                    # the the indices for comparisons of images in different mfovs
                    inds_bmfov = np.ravel_multi_index(np.nonzero(adj_matrix_bmfov), (nimgs,nimgs))
                    # select on the inds for image comparisons between different mfovs
                    sel_bmfov = np.in1d(inds, inds_bmfov, assume_unique=True)
                    # select on the inds for image comparisons within the same mfovs
                    sel_smfov = np.logical_not(sel_bmfov)

                    if verbose:
                        print('\t{} total adjacencies, {} inner, {} outer'.format(nadj,
                            sel_smfov.sum(),sel_bmfov.sum()))
                        print('\tdefault weights: {} inner, {} outer'.format(W_default[0], W_default[1]))
                        print('\tweight {} outer vs inner'.format(W_default[2]))

                    # option for lesser weight for between mfov deltas
                    if W_default[2] != 1.:
                        Winds[sel_bmfov] = W_default[2]
                        Winds_ind[sel_bmfov] = 3

                    # select out the default deltas
                    #delta_coords_z = np.hstack((Dx_z.flat[inds].astype(np.double)[:,None],
                    #                            Dy_z.flat[inds].astype(np.double)[:,None]))
                    # better to use the tiled deltas for the cutoff comparison and the zeiss deltas for the defaults.
                    delta_coords_t = np.hstack((Dx_t.flat[inds].astype(np.double)[:,None],
                                                Dy_t.flat[inds].astype(np.double)[:,None]))

                    # use the correlation cutoff to set deltas for bad correlations to defaults
                    sel = (Cinds <= C_cutoff[0]); sel[sel_bmfov] = 0
                    if verbose:
                        print('\tC_cutoff %.4f, discarded %d bad "inner" mfov correlations' % \
                              (C_cutoff[0], sel.sum(),))
                    Winds[sel] = W_default[0]
                    Winds_ind[sel] = 1
                    delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]
                    sel = (Cinds <= C_cutoff[1]); sel[sel_smfov] = 0
                    if verbose:
                        print('\tC_cutoff %.4f, discarded %d bad "outer" mfov correlations' % \
                              (C_cutoff[1], sel.sum(),))
                    Winds[sel] = W_default[1]
                    Winds_ind[sel] = 2
                    delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                    # xxx - although these are similar, the default_tol is for more specific defaults obtained
                    #   during the twopass method... it's ugly that these comparisons are buried in here this way.
                    if default_tol is not None:
                        # use the tolerance parameter on the comparison deltas to set bad deltas to defaults
                        sel = (np.abs(delta_coords - delta_coords_t) > default_tol[0]).any(axis=1)
                        sel[sel_bmfov] = 0
                        if verbose:
                            print('\tdefault tolerance %g %g, discarded %d bad "inner" deltas' % \
                                  (default_tol[0][0], default_tol[0][1], sel.sum(),))
                        Winds[sel] = W_default[0]
                        Winds_ind[sel] = 1
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                        sel = (np.abs(delta_coords - delta_coords_t) > default_tol[1]).any(axis=1)
                        sel[sel_smfov] = 0
                        if verbose:
                            print('\tdefault tolerance %g %g, discarded %d bad "outer" deltas' % \
                                  (default_tol[1][0], default_tol[1][1], sel.sum(),))
                        Winds[sel] = W_default[1]
                        Winds_ind[sel] = 2
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                    else: # if default_tol is not None:
                        # the distance between the aligned tile centers should always be smaller than
                        #   the distance between the tiled tile centers.
                        dsel = ((delta_coords*delta_coords).sum(1) > (delta_coords_t*delta_coords_t).sum(1))

                        # use the delta cutoff to set bad deltas to defaults
                        sel = (np.abs(delta_coords - delta_coords_t) > D_cutoff[0][0]).any(axis=1)
                        sel = np.logical_or(sel, dsel)
                        sel[sel_bmfov] = 0; sel[vsel] = 0
                        if verbose:
                            print('\tD_cutoff %g %g, discarded %d bad horizontal "inner" deltas' % \
                                  (D_cutoff[0][0][0], D_cutoff[0][0][1],sel.sum(),))
                        Winds[sel] = W_default[0]
                        Winds_ind[sel] = 1
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                        sel = (np.abs(delta_coords - delta_coords_t) > D_cutoff[1][0]).any(axis=1)
                        sel = np.logical_or(sel, dsel)
                        sel[sel_smfov] = 0; sel[vsel] = 0
                        if verbose:
                            print('\tD_cutoff %g %g, discarded %d bad horizontal "outer" deltas' % \
                                  (D_cutoff[1][0][0], D_cutoff[1][0][1],sel.sum(),))
                        Winds[sel] = W_default[1]
                        Winds_ind[sel] = 2
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                        sel = (np.abs(delta_coords - delta_coords_t) > D_cutoff[0][1]).any(axis=1)
                        sel = np.logical_or(sel, dsel)
                        sel[sel_bmfov] = 0; sel[hsel] = 0
                        if verbose:
                            print('\tD_cutoff %g %g, discarded %d bad vertical "inner" deltas' % \
                                  (D_cutoff[0][1][0], D_cutoff[0][1][1],sel.sum(),))
                        Winds[sel] = W_default[0]
                        Winds_ind[sel] = 1
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]

                        sel = (np.abs(delta_coords - delta_coords_t) > D_cutoff[1][1]).any(axis=1)
                        sel = np.logical_or(sel, dsel)
                        sel[sel_smfov] = 0; sel[hsel] = 0
                        if verbose:
                            print('\tD_cutoff %g %g, discarded %d bad vertical "outer" deltas' % \
                                  (D_cutoff[1][1][0], D_cutoff[1][1][1],sel.sum(),))
                        Winds[sel] = W_default[1]
                        Winds_ind[sel] = 2
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]
                    #else: # if default_tol is not None:

                else:   # if adj_matrix_bmfov is not None:
                    # more generic weighting based on single correlation cutoff.
                    sel = (Cinds <= C_cutoff)
                    if verbose:
                        print('\tC_cutoff %.4f, discarded %d bad correlations' % (C_cutoff, sel.sum(),))
                    if Dy_zinds is None:
                        delta_coords[sel] = Dx_zinds[sel]
                    else:
                        delta_coords[sel,0] = Dx_zinds[sel]; delta_coords[sel,1] = Dy_zinds[sel]
                    Winds[sel] = W_default
                #else (if adj_matrix_bmfov is not None:)
            #if C is not None:
        #else (if W is not None:)

        # just broadcast weight multiplication instead of doing the huge outer product
        Winds = Winds[:,None]

        # added this feature to allow for adj rows that have more than two terms.
        inds_nonbinary = adj_nonbinary.keys()
        nnonbinary = len(inds_nonbinary)
        if nnonbinary > 0:
            adj_diff = np.vstack((adj_diff, np.zeros((nnonbinary, nimgs), dtype=np.double)))
            delta_coords = np.vstack((delta_coords, np.zeros((nnonbinary, ndims), dtype=np.double)))
            Winds = np.vstack((Winds, np.zeros((nnonbinary, 1), dtype=np.double)))
            for i,cinds in zip(range(nnonbinary), inds_nonbinary):
                adj_diff[nadj+i,cinds] = adj_nonbinary[cinds]['terms']
                Winds[nadj+i] = adj_nonbinary[cinds]['W']
                delta_coords[nadj+i,:] = adj_nonbinary[cinds]['delta']

        # <<< least-squares to solve for global coordinates, same as matlab mldivide, X\y

        # apply the weights. an outer product in "kevin's formula", this is way more efficient.
        X = adj_diff*Winds; y = delta_coords*Winds

        # scipy / numpy least squares, keep commented here for reference.
        #global_coords = lin.lstsq(X, y, cond=None, check_finite=False)[0] # scipy linalg
        ##global_coords = lin.lstsq(X, y)[0] # numpy linalg

        # use solver from sklearn, depending on requested L1/L2 regularization
        if l1_alpha > 0 and l2_alpha > 0:
            alpha = l1_alpha + l2_alpha
            l1_ratio = l1_alpha / alpha
            clf = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                    fit_intercept=False, copy_X=True, precompute=False)
        elif l1_alpha > 0:
            clf = linear_model.Lasso(alpha=l1_alpha, fit_intercept=False, copy_X=True, precompute='auto')
        elif l2_alpha > 0:
            clf = linear_model.Ridge(alpha=l2_alpha, fit_intercept=False, copy_X=True, solver='lsqr')
        else:
            clf = linear_model.LinearRegression(fit_intercept=False, copy_X=True, n_jobs=zimages.nthreads)
        clf.fit(X, y); global_coords = clf.coef_.T

        # least-squares to solve for global coordinates, same as matlab mldivide, X\y >>>

        if regr_bias:
            bpoly = preprocessing.PolynomialFeatures(degree=1)
            bclf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=zimages.nthreads)
        else:
            bpoly = bclf = None

        # this removes any large bias that may have resulted from the solver.
        # this can happen in cases where there is no regularization (since positions only solved relatively).
        if center_sel is not None:
            assert( label_adj_min < 0 ) # use only one of center_sel or label_adj_min
            mfov._solver_remove_bias(global_coords, center_sel, regr_bias, bpoly, bclf)
            if global_coords.ndim==1:
                global_coords[np.logical_not(center_sel)] = 0
            else:
                global_coords[np.logical_not(center_sel),:] = 0

        if label_adj_min >= 0:
            assert( center_sel is None ) # use only one of center_sel or label_adj_min
            # return a select indicating which components were used.
            if return_comps_sel: comps_sel = np.ones((nimgs,), dtype=bool)
            # this is a more specialized method for removing biases resulting from underspecified system.
            # works instead separately on connected graph components in the adjacency matrix,
            #   and removes biases for each individually.
            nlabels, labels = sp.csgraph.connected_components(adj_matrix, directed=False)
            sizes = np.bincount(labels)
            for i in range(nlabels):
                sel = (labels == i) # "images" in this connected component
                if sizes[i] < label_adj_min:
                    # set deltas for components less than threshold size to zero.
                    global_coords[sel] = 0
                    comps_sel[sel] = 0
                else:
                    mfov._solver_remove_bias(global_coords, sel, regr_bias, bpoly, bclf)

        ret = (global_coords, Winds_ind)
        if return_inds_subs: ret = ret + (inds, subs)
        if return_comps_sel: ret = ret + (comps_sel,)
        if return_deltas: ret = ret + (delta_coords, delta_coords_orig)
        return ret

    @staticmethod
    def create_2D_crops(images_rect_proc, image_size, hex_to_rect_inds, coords, factor=3, istemplate=False):
        crop_size = np.array(image_size, dtype=np.int64)//factor; m=crop_size
        c = factor-1 # this is to select the "end" portion of the image divided into a factor by factor grid
        ctr = (image_size - crop_size)//2 # centering the crop on the image
        nTilesRect = len(images_rect_proc)
        images_cropped = [[[None]*nTilesRect for x in range(2)] for x in range(4)]
        images_offset = [[[None]*nTilesRect for x in range(2)] for x in range(4)]

        # this controls whether the cropping is for the template or the target image.
        # it only serves to flip the "upper triangular" flag since the comparison is then reversed.
        t = istemplate; nt = (not istemplate)

        for i,j in zip(range(hex_to_rect_inds.size), hex_to_rect_inds):
            if images_rect_proc[j] is None: continue

            # NOTE: x/y are flipped when viewing the images (for the cropping).
            #   directions in the comments are given as when viewing the image (not based on coordinates).
            # The middle dimension is for upper and lower triangular in that order. upper triangular is row < col.

            # vertical comparisons in adjacency matrix creation - horizontal for F-order
            k = 1
            # right template to left image
            o = np.array([ctr[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right image
            o = np.array([ctr[0], c*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # horizontal comparisons in adjacency matrix creation - vertical for F-order
            k = 0
            # for vertical comparisons, crops depend on whether this is an even or odd row
            even_row = (coords[i,0] == int(coords[i,0]))
            if even_row:
                # even-row left template to right-up diagonal
                o = np.array([0*m[0], c*m[1]], dtype=np.int64)
                images_offset[k][nt][j] = o
                images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
                # even-row left template to right-down diagonal
                o = np.array([c*m[0], c*m[1]], dtype=np.int64)
                images_offset[k][t][j] = o
                images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            else:
                # odd-row right template to left-up diagonal
                o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][nt][j] = o
                images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
                # odd-row right template to left-down diagonal
                o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][t][j] = o
                images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # anti-diagonal comparisons in adjacency matrix creation
            k = 2
            # right template to left-down diagonal
            o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right-up diagonal
            o = np.array([0*m[0], c*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # diagonal comparisons in adjacency matrix creation
            k = 3
            # right template to left-up diagonal
            o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right-down diagonal
            o = np.array([c*m[0], c*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

        return crop_size, images_cropped, images_offset

    @staticmethod
    def create_1D_crops(images_rect_proc, image_size, hex_to_rect_inds, coords, factor=2, istemplate=False):
        c = factor-1 # this is to select the "end" portion of the image divided into a factor by factor grid
        nTilesRect = len(images_rect_proc)
        images_cropped = [[[None]*nTilesRect for x in range(2)] for x in range(4)]
        images_offset = [[[None]*nTilesRect for x in range(2)] for x in range(4)]

        # crop sizes only depend on whether we are comparing vertically or horizontally
        crop_sizes =[None]*4
        # vertical comparisons in adjacency matrix creation - horizontal for F-order
        k = 1; crop_sizes[k] = np.array(image_size, dtype=np.int64); crop_sizes[k][1] //= factor
        # horizontal comparisons in adjacency matrix creation - vertical for F-order
        k = 0; crop_sizes[k] = np.array(image_size, dtype=np.int64); crop_sizes[k][0] //= factor
        # anti-diagonal comparisons in adjacency matrix creation
        k = 2; crop_sizes[k] = crop_sizes[0]
        # diagonal comparisons in adjacency matrix creation
        k = 3; crop_sizes[k] = crop_sizes[0]

        # this controls whether the cropping is for the template or the target image.
        # it only serves to flip the "upper triangular" flag since the comparison is then reversed.
        t = istemplate; nt = (not istemplate)

        for i,j in zip(range(hex_to_rect_inds.size), hex_to_rect_inds):
            if images_rect_proc[j] is None: continue

            # NOTE: x/y are flipped when viewing the images (for the cropping).
            #   directions in the comments are given as when viewing the image (not based on coordinates).
            # The middle dimension is for upper and lower triangular in that order. upper triangular is row < col.

            # vertical comparisons in adjacency matrix creation - horizontal for F-order
            k = 1; m=crop_sizes[k]
            # right template to left image
            o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right image
            o = np.array([0*m[0], c*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # horizontal comparisons in adjacency matrix creation - vertical for F-order
            k = 0; m=crop_sizes[k]
            # for vertical comparisons, crops depend on whether this is an even or odd row
            even_row = (coords[i,0] == int(coords[i,0]))
            if even_row:
                # even-row left template to right-up diagonal
                o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][nt][j] = o
                images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
                # even-row left template to right-down diagonal
                o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][t][j] = o
                images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            else:
                # odd-row right template to left-up diagonal
                o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][nt][j] = o
                images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
                # odd-row right template to left-down diagonal
                o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
                images_offset[k][t][j] = o
                images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # anti-diagonal comparisons in adjacency matrix creation
            k = 2; m=crop_sizes[k]
            # right template to left-down diagonal
            o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right-up diagonal
            o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

            # diagonal comparisons in adjacency matrix creation
            k = 3; m=crop_sizes[k]
            # right template to left-up diagonal
            o = np.array([0*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][nt][j] = o
            images_cropped[k][nt][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]
            # left template to right-down diagonal
            o = np.array([c*m[0], 0*m[1]], dtype=np.int64)
            images_offset[k][t][j] = o
            images_cropped[k][t][j] = images_rect_proc[j][o[0]:o[0]+m[0], o[1]:o[1]+m[1]]

        return crop_sizes, images_cropped, images_offset

    # reload image coords from stitched region coords (i.e. that were previously solved here and saved).
    # this function subtracts off the min of the loaded coordinates (make them relative to region).
    # then the coords are offset by the original zeiss global coordinates.
    def load_stitched_region_image_coords(self, fn, scale=1, rmv_thb=False, add_thb=False):
        if not hasattr(self,'zeiss_region_coords'):
            self.zeiss_region_coords = self.region_coords.copy()
        zeiss_corners = self.zeiss_region_coords.reshape(-1,2).min(0)[None,None,:]
        self.region_coords, self.region_filenames = zimages.read_all_image_coords(fn, self.niTiles,
                        cache_dn=self.cache_dir, nmfovs=self.nmfovs, expect_mfov_subdir=True)
        self.region_coords = self.region_coords - np.nanmin(self.region_coords.reshape(-1,2), axis=0)[None,None,:]
        self.region_coords = scale * self.region_coords + zeiss_corners
        #self.get_mfov_neighbors() # xxx - why were we redoing this? Zeiss coords should suffice

        if rmv_thb:
            self.region_filenames = [[None if x is None else os.path.join(os.path.dirname(x),
                re.sub('^thumbnail_','', os.path.basename(x))) for x in y] for y in self.region_filenames]
        if add_thb:
            self.region_filenames = [[None if x is None else os.path.join(os.path.dirname(x),
                    'thumbnail_' + os.path.basename(x)) for x in y] for y in self.region_filenames]

    # get neighbors of current mfov_id based on delaunay triangulation of center image coordinates.
    def get_mfov_neighbors(self):
        if self.mfov_tri is None and self.nmfovs > 3:
            # create a mapping for all the images including ones that could be in neighboring mfovs
            self.mfov_tri = scipy.spatial.Delaunay(self.region_coords[:,0,:])
        tri, coords, mfov_id = self.mfov_tri, self.region_coords, self.mfov_id

        if self.nmfovs <= 3:
            # with less than 4 mfovs, neighbors is just the other two mfovs
            mfov_neighbors = np.arange(self.nmfovs); mfov_neighbors = mfov_neighbors[mfov_neighbors != mfov_id]
        else:
            tri = self.mfov_tri

            #Tuple of two ndarrays of int: (indices, indptr). The indices of neighboring vertices of vertex k
            #  are indptr[indices[k]:indices[k+1]]
            indices, indptr = tri.vertex_neighbor_vertices
            mfov_neighbors = indptr[indices[mfov_id]:indices[mfov_id+1]]
            # to remove "non-neighbors" along convex hull.
            # next closest neighbor in a hex grid should be sqrt(3)*dist, so use 1.5 as cutoff
            D = ((coords[mfov_neighbors,0,:] - coords[mfov_id,0,:])**2).sum(axis=1)
            mfov_neighbors = mfov_neighbors[D < D.min()*1.5]

        # xxx - if rotation of the mfovs is possible need to revisit this
        # put the neighbors in order based on direction in hex grid
        a = mfov.get_coords_angles(coords[mfov_id,0,:], coords[mfov_neighbors,0,:])
        inds = np.argsort(a); mfov_neighbors = mfov_neighbors[inds]
        # get the direction based on a regular hex grid, this is an ordered numbering of the hex edges
        mfov_hex_angles = np.arange(6, dtype=np.double)/3*np.pi + np.pi/6
        mfov_neighbors_edge = np.array([np.argmin(np.abs(x - mfov_hex_angles)) for x in a[inds]])

        self.mfov_neighbors, self.mfov_neighbors_edge = mfov_neighbors, mfov_neighbors_edge

        return tri

    def load_images(self, init_only=False, montage=False):
        if self.mfov_verbose:
            logger.debug("Reading and montaging with tiled and zeiss coords")
            t = time.time()

        if len(self.ring['from_neighbor']) > 0:
            # load tiles within mfov and including overlapping tiles from neighboring mfovs
            self.images, self.filenames_imgs, self.coords_imgs, self.mfov_ids, self.mfov_tile_ids = \
                zimages.read_images_neighbors(self.images_load_folder,
                    self.region_filenames, self.region_coords, self.nTiles, self.niTiles, self.mfov_id,
                    self.mfov_neighbors, self.mfov_neighbors_edge, self.ring['to_neighbor'],
                    self.ring['from_neighbor'], crop=self.border_crop, dsstep=self.dsstep, invert=self.invert_images,
                    reduce=self.blkrdc_func, init_only=init_only, cache_dn=self.cache_dir)
        else:
            # assembling fully tiled image with fixed indices (stitched image without any alignment)
            self.coords_imgs  = self.region_coords[self.mfov_id,:,:]
            self.filenames_imgs = self.region_filenames[self.mfov_id]
            self.images = zimages.read_images(self.images_load_folder, self.filenames_imgs, crop=self.border_crop,
                    dsstep=self.dsstep, invert=self.invert_images, reduce=self.blkrdc_func, init_only=init_only,
                    cache_dn=self.cache_dir)
            self.mfov_ids = np.empty((self.nTiles,), dtype=np.int64); self.mfov_ids.fill(self.mfov_id)
            self.mfov_tile_ids = np.arange(self.nTiles, dtype=np.int64)

        if self.scale_tiled_coords_factor > 1:
            coords = self.ring['coords']
            coords = coords - coords.mean(0)[None,:]
            coords *= self.scale_tiled_coords_factor
        else:
            coords = self.ring['coords']
        if self.false_color_montage:
            # decided to make the y dim as an offset
            hcoords = self.ring['coords'].copy(); hcoords[:,1] = 0
        else:
            hcoords = None
        if montage:
            self.image_tiled, self.corners_tiled, _ = zimages.montage(self.images, coords,
                color_coords=hcoords, scale=np.array(self.images[0].shape)[::-1], bg=self.montage_background)
            self.image_coords, self.corners_coords, _ = zimages.montage(self.images, self.coords_imgs,
                scale=1./self.dsstep, color_coords=hcoords, bg=self.montage_background)

        # None images are for neighbors from adjacent mfovs where there is no adjacent mfov
        sel = np.array([x is not None for x in self.images])
        self.nimgs = sel.sum()

        self.images_shape = np.array(self.images[0].shape); self.images_dtype = self.images[0].dtype

        if self.mfov_verbose:
            msg = ('\t%d total images out of %d tiles' % (self.nimgs,self.nTiles))
            logger.debug(msg)
            msg = ('\tdone in %.4f s' % (time.time() - t, ))
            logger.debug(msg)

    def get_full_adj(self):
        # the middle row is always shifted left (zeiss), so the top row shifts right for odd values of radius.
        center_row_is_odd = (self.omfov_diameter//2)%2
        full_adj = mfov.adjacency_matrix_hex((self.omfov_diameter,self.omfov_diameter), with_type=True,
                shift_right=center_row_is_odd)
        # remove adjacencies that are outside the hexagonal shape
        sel = (self.ring['rect_to_hex'] >= self.nTiles)
        full_adj[sel,:] = 0; full_adj[:,sel] = 0
        return full_adj

    def align_and_stitch(self, init_only=False, get_residuals=False, get_delta_coords=False, C=None, Dx=None, Dy=None,
            Dx_d=None, Dy_d=None, Dx_t=None, Dy_t=None, default_tol=None, init_only_load_images=True,
            montage_load_images=False, save_weights=False, nworkers=1, doplots=False, dosave_path=''):
        if self.mfov_verbose:
            msg = ('Stitching region %s, MFoV %d of %d' % (self.region_str, self.mfov_id+1, self.nmfovs))
            logger.debug(msg)
        # xxx - this init_only hack deepened, refactor this if we ever refactor zimages
        if self.images is None:
            self.load_images(init_only=(init_only and not init_only_load_images), montage=montage_load_images)

        full_adj = self.get_full_adj()
        nadj = np.count_nonzero(full_adj)
        if self.mfov_verbose:
            msg = ('Hexagonal ring diameter %d has %d tiles with %d adjacencies' % \
                   (self.omfov_diameter, self.nTiles, nadj,))
            logger.debug(msg)

        # convert coords, filenames, etc to "rectangular" indexed tile space.
        #   this is because the adjacency matrix is computed with rectangular numbered tiling.
        # xxx - more modularization here?

        # remap the unrolled hex indexed images into rectangular indexed images
        images_rect = [None]*self.nTilesRect; filenames_imgs_rect = [None]*self.nTilesRect
        for i,j in zip(range(self.nTiles), self.ring['hex_to_rect']):
            images_rect[j] = self.images[i]; filenames_imgs_rect[j] = self.filenames_imgs[i]
        coords_fixed_rect = np.zeros((self.nTilesRect,2), dtype=np.double)
        coords_fixed_rect[self.ring['hex_to_rect'],0] = self.ring['coords'][:,0]
        coords_fixed_rect[self.ring['hex_to_rect'],1] = self.ring['coords'][:,1]
        self.mfov_ids_rect = np.empty((self.nTilesRect,), dtype=np.int64); self.mfov_ids_rect.fill(-1)
        self.mfov_ids_rect[self.ring['hex_to_rect']] = self.mfov_ids
        self.mfov_tile_ids_rect = np.empty((self.nTilesRect,), dtype=np.int64); self.mfov_tile_ids_rect.fill(-1)
        self.mfov_tile_ids_rect[self.ring['hex_to_rect']] = self.mfov_tile_ids
        self.region_tile_ids_rect = self.mfov_ids_rect*self.niTiles + self.mfov_tile_ids_rect
        sel = (self.mfov_ids_rect < 0); self.region_tile_ids_rect[sel] = -1

        # remove any missing adjacencies
        sel = np.array([x is None for x in images_rect])
        full_adj[sel,:] = 0; full_adj[:,sel] = 0; nadj = np.count_nonzero(full_adj)
        if self.mfov_verbose:
            msg = ('%d adjacencies after removing missing neighbors (at region edges)' % (nadj,))
            logger.debug(msg)
        self.adj_matrix = full_adj; subs = np.transpose(self.adj_matrix.nonzero())

        # create another adjacency matrix that only includes "inter-mfov" adjacencies.
        # xxx - how to do this without a loop?
        full_adj_imfov = np.ones(full_adj.shape, dtype=bool)
        for i in range(self.nTilesRect):
            full_adj_imfov[i,:] = (self.mfov_ids_rect[i] != self.mfov_ids_rect)
        full_adj_imfov = np.logical_and(full_adj, full_adj_imfov); nadj_imfov = np.count_nonzero(full_adj_imfov)
        self.adj_matrix_btw_mfovs = full_adj_imfov
        if self.mfov_verbose:
            msg = ('%d adjacencies between tiles in different mfovs' % (nadj_imfov,))
            logger.debug(msg)

        # create rectangular indexed distance matrices for zeiss coordinates
        coords_imgs_rect = np.zeros((self.nTilesRect,2), dtype=np.double)
        coords_imgs_rect[self.ring['hex_to_rect'],0] = self.coords_imgs[:,0]
        coords_imgs_rect[self.ring['hex_to_rect'],1] = self.coords_imgs[:,1]
        Dx_z = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.double)
        Dy_z = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.double)
        Dx_z[subs[:,0], subs[:,1]] = coords_imgs_rect[subs[:,0],0] - coords_imgs_rect[subs[:,1],0]
        Dy_z[subs[:,0], subs[:,1]] = coords_imgs_rect[subs[:,0],1] - coords_imgs_rect[subs[:,1],1]

        # calculate the max delta between tiles in the horizontal and vertical directions.
        # do not include "inter-mfov" adjacencies.
        # this is used to to print information for guiding setting the feathering distances.
        # additionally it is used in the overlap correction during the image load.
        nsel = np.logical_not(full_adj_imfov, full_adj > 0)
        Dx_z /= self.dsstep; Dy_z /= self.dsstep
        self.max_delta_zeiss = np.array([np.abs(Dx_z[np.logical_and(full_adj==2, nsel)]).max(),
                                   np.abs(Dy_z[np.logical_and(np.logical_and(full_adj>0,full_adj!=2), nsel)]).max()])
        if self.mfov_verbose:
            #logger.debug('max delta zeiss %g %g', self.max_delta_zeiss[0], self.max_delta_zeiss[1])
            #logger.debug('tile size %d %d', self.images_shape[1], self.images_shape[0])
            tmp = self.images_shape[::-1] - self.max_delta_zeiss
            #logger.debug('max overlap pix %g %g', tmp[0], tmp[1])
            tmp /= (1e3/(self.scale_nm*self.dsstep))
            logger.debug('max overlap um %g %g', tmp[0], tmp[1])

        # use zeiss deltas if defaults not specified (used for "twopass" method in regions)
        if Dx_d is None:
            assert(Dy_d is None)
            Dx_d = Dx_z; Dy_d = Dy_z
        else:
            assert(Dy_d is not None)
            # double-default to Zeiss deltas for any default values not provided
            sel = np.logical_not(np.isfinite(Dx_d))
            Dx_d[sel] = Dx_z[sel]; Dy_d[sel] = Dy_z[sel]

        # save the rectangular-indexed filenames and coords for montage
        self.mfov_filenames = filenames_imgs_rect
        self.mfov_filenames_sel = [x is not None for x in filenames_imgs_rect]
        self.mfov_images = images_rect
        self.mfov_hex_coords = coords_fixed_rect

        # xxx - this is basically a hack to just get max_delta_zeiss without running the stitching
        #   this turned into a rabbit hole, see also comments in region.
        #   it's probably possible to cleanly fix this, but requires more major surgery.
        #   the main issues are (1) the image size comes from the bottom of the call stack,
        #     after at least one image is loaded and (2) the coordinates are shuffled around depending
        #     on the mfov neighbors which is also done at the bottom of the call stack in zimages.
        if init_only: return None

        if Dx_t is None:
            # create rectangular indexed distance matrices for tiled coordinates
            coords_imgs_rect = np.zeros((self.nTilesRect,2), dtype=np.double)
            coords = self.ring['coords']*np.array(self.images[0].shape)[::-1]
            coords_imgs_rect[self.ring['hex_to_rect'],0] = coords[:,0]
            coords_imgs_rect[self.ring['hex_to_rect'],1] = coords[:,1]
            Dx_t = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.double)
            Dy_t = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.double)
            Dx_t[subs[:,0], subs[:,1]] = coords_imgs_rect[subs[:,0],0] - coords_imgs_rect[subs[:,1],0]
            Dy_t[subs[:,0], subs[:,1]] = coords_imgs_rect[subs[:,0],1] - coords_imgs_rect[subs[:,1],1]

        ## create another adjacency matrix that only includes adjacencies between the inner mfov and the outer mfovs.
        #full_adj_bmfov = np.ones(full_adj.shape, dtype=bool);
        #full_adj_bmfov[ np.ix_(self.ring['is_outer_rect'], self.ring['is_outer_rect']) ] = 0
        #full_adj_bmfov[ np.ix_(self.ring['is_inner_rect'], self.ring['is_inner_rect']) ] = 0
        #full_adj_bmfov = np.logical_and(full_adj, full_adj_bmfov); nadj_bmfov = np.count_nonzero(full_adj_bmfov)
        #if self.mfov_verbose:
        #    print('%d adjacencies between tiles in inner mfov and outer mfovs' % (nadj_bmfov,))

        if self.mfov_verbose:
            msg = ('%d adjacencies between tiles in different mfovs' % (nadj_imfov,))
            logger.debug(msg)
            logger.debug('Preprocessing images before xcorr')
            t = time.time()

        images_rect_proc = [None]*self.nTilesRect
        self.low_complexity_rect_inds = np.zeros((self.nTilesRect,), dtype=np.int64); low_complexity_count = 0
        if self.V_cutoff > 0:
            self.images_rect_var = np.empty((self.nTilesRect,), dtype=np.double); self.images_rect_var.fill(np.nan)
        for i in self.ring['hex_to_rect']:
            if images_rect[i] is None: continue

            if self.V_cutoff > 0:
                # do not utilize correlations at all for low-complexity images, based on variance
                self.images_rect_var[i] = images_rect[i].var(dtype=np.double)
                if self.images_rect_var[i] < self.V_cutoff:
                    self.low_complexity_rect_inds[low_complexity_count] = i; low_complexity_count += 1
                    continue
            if C is None:
                images_rect_proc[i] = template_match_preproc(images_rect[i], whiten_sigma=self._proc_whiten_sigma,
                        clahe_clipLimit=self._proc_clahe_clipLimit, clahe_tileGridSize=self._proc_clahe_tileGridSize)
        # correlations are set to "below-off" below for low variance images so the zeiss defaults are used.
        self.low_complexity_rect_inds = self.low_complexity_rect_inds[:low_complexity_count]
        full_adj_c = full_adj.copy()
        full_adj_c[self.low_complexity_rect_inds,:] = 0; full_adj_c[:,self.low_complexity_rect_inds] = 0
        if self.mfov_verbose:
            msg = ('\tRemoved correlations for %d low complexity images based on variance cutoff %.2f' % \
                  (low_complexity_count, self.V_cutoff))
            logger.debug(msg)
            msg = ('\tdone in %.4f s' % (time.time() - t, ))
            logger.debug(msg)

        # if results of a previous xcorr run are not provided then run the xcorrs
        if C is None:
            self.C, Dx, Dy = self._run_xcorrs(images_rect_proc, full_adj_c, nworkers, doplots, dosave_path)
        else:
            self.C = C

        # put in values "below off" for the correlation for low variance images,
        #   so that low-variance images use the zeiss default deltas.
        sel = np.logical_and(full_adj_c==0, full_adj>0); self.C[sel] = -2.

        # these features are important for avoid bad alignments from spurious correlations.
        C_cutoff = self.C_cutoff
        ind = 0; means = [0.]; stds = [0.]; weights = [0.]; gmm_order=-1; use_C = self.C[self.C > 0]
        if self.C_cutoff_soft_nGMM > 0 and use_C.size > self.C_cutoff_soft_nGMM:
            # fit the distribution of correlations with a GMM
            X = use_C.flat[:].copy()[:,None]; min_bic = np.inf; imin = 0
            for i in range(self.C_cutoff_soft_nGMM):
                gmm = GaussianMixture(i+1, covariance_type='spherical')
                gmm.fit(X); bic = gmm.bic(X)
                if bic < min_bic:
                    min_bic = bic; imin = i; gmm_order = i+1; weights = gmm.weights_.flat[:]
                    means = gmm.means_.flat[:]; stds = np.sqrt(gmm.covariances_.flat[:])

            if imin > 0:
                # take the largest peak in the defined softmax range
                sel = np.logical_and(means >= self.C_cutoff_soft_rng[0], means <= self.C_cutoff_soft_rng[1])
                if sel.sum() > 0:
                    weights[np.logical_not(sel)] = -1; ind = np.argmax(weights)
                else:
                    # default - take the peak that is closest to the center of the defined softmax range
                    d = np.abs(means - np.mean(self.C_cutoff_soft_rng)); ind = np.argmin(d)
            Csoft = means[ind] - self.C_cutoff_soft_nstds*stds[ind] # n stds below the fitted max correlation peak

            # do not let softmax go below hard C_cutoff, or above defined soft max
            C_cutoff = np.hstack((np.array(self.C_cutoff)[:,None], np.array([Csoft,Csoft])[:,None])).max(1)
            C_cutoff[C_cutoff > self.C_cutoff_soft_rng[1]] = self.C_cutoff_soft_rng[1]
        if self.mfov_verbose:
            msg = ('\tUsing C_cutoff %.3f %.3f, GMM %d u=%.5f, s=%.5f, w=%.5f' % \
                   (C_cutoff[0], C_cutoff[1], gmm_order, means[ind], stds[ind], weights[ind]))
            logger.debug(msg)

        # create select for any tiles with adjacencies (for centering)
        sel_tiles = (full_adj.sum(0) > 0)
        try:
            if get_residuals or get_delta_coords:
                self.xy_fixed, Winds, _, subs, delta_coords, delta_coords_orig = mfov.solve_stitching(full_adj,
                        Dx, Dy=Dy, C=self.C, C_cutoff=C_cutoff, Dx_z=Dx_d, Dy_z=Dy_d, Dx_t=Dx_t, Dy_t=Dy_t,
                        D_cutoff=self.D_cutoff, W_default=self.W_default, adj_matrix_bmfov=full_adj_imfov,
                        center_sel=sel_tiles, return_inds_subs=True, return_deltas=True, default_tol=default_tol,
                        verbose=self.mfov_verbose)
            else:
                self.xy_fixed, Winds = mfov.solve_stitching(full_adj, Dx, Dy=Dy, C=self.C, C_cutoff=C_cutoff,
                        Dx_z=Dx_d, Dy_z=Dy_d, Dx_t=Dx_t, Dy_t=Dy_t, D_cutoff=self.D_cutoff, W_default=self.W_default,
                        adj_matrix_bmfov=full_adj_imfov, center_sel=sel_tiles, default_tol=default_tol,
                        verbose=self.mfov_verbose)
        except np.linalg.LinAlgError:
            logger.debug('WARNING: solver did not converge for this mfov, using zeiss deltas entirely')
            # if the solver does not converge, just use the zeiss coordinates
            if get_residuals or get_delta_coords:
                self.xy_fixed, Winds, _, subs, delta_coords, delta_coords_orig = mfov.solve_stitching(full_adj,
                        Dx_z, Dy=Dy_z, C=self.C, C_cutoff=C_cutoff, Dx_z=Dx_z, Dy_z=Dy_z, Dx_t=Dx_t, Dy_t=Dy_t,
                        D_cutoff=self.D_cutoff, W_default=self.W_default, adj_matrix_bmfov=full_adj_imfov,
                        center_sel=sel_tiles, return_inds_subs=True, return_deltas=True, verbose=self.mfov_verbose)
            else:
                self.xy_fixed, Winds = mfov.solve_stitching(full_adj, Dx_z, Dy=Dy_z, C=self.C, C_cutoff=C_cutoff,
                        Dx_z=Dx_z, Dy_z=Dy_z, Dx_t=Dx_t, Dy_t=Dy_t, D_cutoff=self.D_cutoff, W_default=self.W_default,
                        adj_matrix_bmfov=full_adj_imfov, verbose=self.mfov_verbose)
        if save_weights: self.mfov_weights = Winds.reshape(-1)

        if get_residuals:
            # solve_stitching always interprets adj matrix as row - col when creating diff matrix
            residuals = (self.xy_fixed[subs[:,0],:] - self.xy_fixed[subs[:,1],:]) - delta_coords
            residuals_xy = (self.xy_fixed[subs[:,0],:] + self.xy_fixed[subs[:,1],:])/2
            residuals_triu = (subs[:,0] < subs[:,1])
            residuals_orig = (self.xy_fixed[subs[:,0],:] - self.xy_fixed[subs[:,1],:]) - delta_coords_orig

        # save the rectangular-indexed filenames and coords for montage
        self.mfov_coords = self.xy_fixed * self.dsstep

        if get_residuals: return residuals, residuals_xy, residuals_triu, residuals_orig
        if get_delta_coords: return delta_coords_orig

    def _run_xcorrs(self, images_rect_proc, full_adj_c, nworkers, doplots, dosave_path):

        # NOTE: can not do query cuda devices in a global init because cuda can not be forked,
        #   meaning any processes that try to use cuda will fail. another option is 'spawn'
        #   but all the conditional code required for this is a nightmare.
        self.query_cuda_devices()
        # num_workers is only used by the rcc-xcorr gpu version, does not matter for other methods.
        msg = 'using {} method for computing xcorrs, backend {}, use_gpu {}, num_gpus {}, num_workers {}'.\
                format(self.fft_method, self.fft_backend, self.use_gpu, self.cuda_device_count, self.nthreads)
        logger.debug(msg)

        ctx_managers = create_scipy_fft_context_manager(self.use_gpu, self.fft_method, self.fft_backend, self.nthreads)
        with ExitStack() as stack:
            for mgr in ctx_managers:
                stack.enter_context(mgr)

            C, Dx, Dy = self._run_xcorrs_inner(images_rect_proc, full_adj_c, nworkers, doplots, dosave_path)

        return C, Dx, Dy

    def _run_xcorrs_inner(self, images_rect_proc, full_adj_c, nworkers, doplots, dosave_path):
        # get image size from the first loaded image, must be non-None.
        # unlike most other spots in msem package, this size is NOT reversed from the image shape.
        sz = np.array(self.images[0].shape)

        if self.mfov_verbose:
            logger.debug('Running normxcorr2')
            t = time.time()
        return_comps = (self.export_xcorr_comps_path is not None)
        if self.template_crop_factor > 1:
            # create the crops for running with normxcorr2
            Tsize, Timages, Toffset = mfov.create_2D_crops(images_rect_proc, sz, self.ring['hex_to_rect'],
                self.ring['coords'], factor=self.template_crop_factor, istemplate=True)
            if self.image_crop_factor > 1:
                Asize, Aimages, Aoffset = mfov.create_1D_crops(images_rect_proc, sz, self.ring['hex_to_rect'],
                    self.ring['coords'], factor=self.image_crop_factor, istemplate=False)
            else:
                Asize = self.images[0].shape
                Aimages = [images_rect_proc, images_rect_proc]

            C = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.double)
            Dx = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.int64)
            Dy = np.zeros((self.nTilesRect,self.nTilesRect), dtype=np.int64)
            comps_dicts = [None]*4

            if self.fft_method == FFT_types.rcc_xcorr:
                # xxx - maybe ultimately this replaces everything else in _template_match
                #   for now optionally use the new library rcc-xcorr for running the cross correlations
                assert( self.image_crop_factor == 1 ) # did not implement

                # unroll the images / templates and set comparisons based on the adjacency matrix.
                images = Aimages[0]
                templates = [None]*self.nTilesRect*2*4
                correlations = np.zeros((0,2), dtype=np.int64)
                for i in range(4):
                    comps = normxcorr2_adj_to_comps(len(images), full_adj_c==i+1,
                        use_tri_templates=True, use_tri_images=False)
                    comps[:,1] = comps[:,1] + self.nTilesRect*2*i
                    correlations = np.concatenate((correlations, comps), axis=0)
                    for j in range(2):
                        assert(len(Timages[i][j]) == self.nTilesRect)
                        ind = self.nTilesRect*2*i + self.nTilesRect*j
                        templates[ind:ind+self.nTilesRect] = Timages[i][j]

                # the custom_eps is the tol value used in _template_match. xxx - set this globally?
                batch_correlations = BatchXCorr.BatchXCorr(images, templates, correlations,
                        normalize_input=False, group_correlations=True, crop_output=(0, 0),
                        use_gpu=self.use_gpu, num_gpus=self.cuda_device_count, num_workers=nworkers,
                        disable_pbar=True, override_eps=True, custom_eps=1e-6)
                coords, peaks = batch_correlations.execute_batch()

                # re-roll the correlation peak and coord results and convert coords to image center deltas.
                T_size = np.array(Tsize)
                for c in range(correlations.shape[0]):
                    y = correlations[c,0]
                    x = correlations[c,1] % self.nTilesRect
                    Tistriu = correlations[c,1] // self.nTilesRect % 2
                    assert( (x < y) == Tistriu ) # sanity check
                    i = correlations[c,1] // self.nTilesRect // 2
                    C[x,y] = peaks[c]

                    deltaC = coords[c,:]
                    template_offset=Toffset[i]
                    img_offset=None
                    # code pulled out of normxcorr2_fft_adj to get deltas between the image centers
                    deltaA = deltaC - T_size + 1 # the correlation peak location in the image A
                    Dy[x,y], Dx[x,y] = deltaA - (template_offset[Tistriu][x] if template_offset is not None else 0) \
                        + (img_offset[Tistriu][x] if img_offset is not None else 0)

            else: # if self.fft_method == FFT_types.rcc_xcorr
                for i in range(4):
                    if self.image_crop_factor > 1:
                        cC,cDy,cDx,comps_dict = normxcorr2_fft_adj(Timages[i], Tsize, Aimages[i], Asize[i],
                            full_adj_c==i+1, template_offset=Toffset[i], img_offset=Aoffset[i], use_tri_templates=True,
                            use_tri_images=True, use_gpu=self.use_gpu, doplots=doplots, dosave_path=dosave_path)
                    else:
                        if i==0:
                            cC,cDy,cDx, Fa,Fb,T,F,fftT,fftF,local_sum_A,denom_A,comps_dicts[i] = normxcorr2_fft_adj(\
                                Timages[i], Tsize, Aimages, Asize, full_adj_c==i+1, template_offset=Toffset[i],
                                img_offset=None, use_tri_templates=True, use_tri_images=False, return_precomputes=True,
                                use_gpu=self.use_gpu, doplots=doplots, dosave_path=dosave_path,
                                return_comps=return_comps)
                        else:
                            cC,cDy,cDx,comps_dicts[i] = normxcorr2_fft_adj(Timages[i], Tsize, Aimages, Asize,
                                full_adj_c==i+1, template_offset=Toffset[i], img_offset=None,
                                Fa=Fa,Fb=Fb,T=T,F=F,fftT=fftT,fftF=fftF,local_sum_A=local_sum_A,denom_A=denom_A,
                                use_tri_templates=True, use_tri_images=False, use_gpu=self.use_gpu,
                                doplots=doplots, dosave_path=dosave_path, return_comps=return_comps)
                    C += cC; Dx += cDx; Dy += cDy
        else: #if self.template_crop_factor > 1
            assert( self.fft_method != FFT_types.rcc_xcorr ) # did not implement
            all_img = [images_rect_proc, images_rect_proc]
            C,Dy,Dx,comps_dict = normxcorr2_fft_adj(all_img, self.images[0].shape, all_img, self.images[0].shape,
                full_adj_c, use_gpu=self.use_gpu, doplots=doplots, dosave_path=dosave_path)
        if self.mfov_verbose:
            msg = ('\tdone in %.4f s' % (time.time() - t, ))
            logger.debug(msg)

        if self.export_xcorr_comps_path is not None:
            # xxx - hacky way to export xcorr comparisons / results for validating other methods
            import tifffile
            import dill
            import scipy.io as io
            logger.debug('Dumping xcorr results for external comparison / validation')
            t = time.time()
            pn = self.export_xcorr_comps_path
            assert(len(Aimages[0]) == self.nTilesRect)
            for i in range(self.nTilesRect):
                if Aimages[0][i] is not None:
                    tifffile.imwrite(os.path.join(pn,'image{:04d}.tif'.format(i)), Aimages[0][i])
            for i in range(4):
                for j in range(2):
                    assert(len(Timages[i][j]) == self.nTilesRect)
                    for k in range(self.nTilesRect):
                        if Timages[i][j][k] is not None:
                            ind = self.nTilesRect*2*i + self.nTilesRect*j + k
                            tifffile.imwrite(os.path.join(pn,'templ{:04d}.tif'.format(ind)), Timages[i][j][k])

                if i==0:
                    comps = comps_dicts[i]['comps']
                    Camax = comps_dicts[i]['Camax']
                    Cmax = comps_dicts[i]['Cmax']
                    Cimgs = comps_dicts[i]['C']
                else:
                    tmp = comps_dicts[i]['comps']; tmp[:,1] = tmp[:,1] + self.nTilesRect*2*i
                    comps = np.concatenate((comps, tmp), axis=0)
                    Camax = np.concatenate((Camax, comps_dicts[i]['Camax']), axis=0)
                    Cmax = np.concatenate((Cmax, comps_dicts[i]['Cmax']))
                    Cimgs = Cimgs + comps_dicts[i]['C']
            for i in range(len(Cimgs)):
                tifffile.imwrite(os.path.join(pn,'xcorr{:04d}.tif'.format(i)), Cimgs[i])
            d = {'comps':comps, 'Cmax':Cmax, 'Camax':Camax}
            with open(os.path.join(pn,'comps.dill'), 'wb') as f: dill.dump(d, f)
            io.savemat(os.path.join(pn,'comps.mat'), d)
            msg = ('\tdone in %.4f s' % (time.time() - t, ))
            logger.debug(msg)
        #if self.export_xcorr_comps_path is not None:

        return C, Dx, Dy
