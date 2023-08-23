
# this is a parameter defition import shared across all the alignment top level scripts.
# the parameters are defined per experiment (different for different experiment alignments).
# NOTE: defaults for any parameters that are set by argparse (command line) are defined in the
#   individual scripts and not here.
# xxx - a config parser is a cleaner choice for this, but particularly for the path definitions
#   due to the way the experiments were run, several things have to be determined at run time
#   at least in the first iteration of the workflow.
#   on the experimental side, they are promising for better validation during acquisition.
# xxx - seperate further into sections of "probably need to adjust" knobs, vs relatively static knobs
#   (file name format strings for example are relatively static).
print('def_common_params: 2019 ZF_retina - 3 wafers')

import os
import sys
import numpy as np

# <<< hacky method for common function definitions for def_common_params
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from def_common_params_common import cget_paths, csave_region_strs_to_meta, cload_region_strs_from_meta
from def_common_params_common import init_region_info, cglob_regions_exclude, cimport_exclude_regions
def get_paths(wafer_id):
    return cget_paths(wafer_id, root_raw, root_align, root_thumb, raw_folders_all, align_folders_all, region_strs_all,
            experiment_subfolder, index_protocol_folders, proc_subdirs, meta_subdirs)
def save_region_strs_to_meta(wafer_id):
    return csave_region_strs_to_meta(wafer_id, get_paths, raw_folders_all, alignment_folders, legacy_zen_format,
        reimage_beg_inds, fullres_dir, manifest_suffixes)
def load_region_strs_from_meta(wafer_id, fn=None):
    return cload_region_strs_from_meta(wafer_id, fn)
def glob_regions_exclude(wafer_id, inds, exclude_inds):
    return cglob_regions_exclude(wafer_id, inds, exclude_inds, root_raw, raw_folders_all, experiment_subfolder)
def import_exclude_regions():
    return cimport_exclude_regions(exclude_txt_fn_str, all_wafer_ids,
        root_align, experiment_subfolder, align_folders_all)
# hacky method for common function definitions for def_common_params >>>

# <<< path and filename parameters

# root directories, change depending on run location
root_raw = '/axon/axon_fs/cne-mSEM-data'
root_thumb = '/axon/scratch/pwatkins/mSEM-data-thumbnails/ds4'
root_align = '/axon/scratch/pwatkins/mSEM-proc'
root_czi = root_raw

# override or specify the voxel resolution
# for the new image acquisition format, need to specify
#scale_nm = None # to read it from the dataset
scale_nm = 4

# override or specify the number of mfov beams (images per mfov)
# for the new image acquisition format, need to specify
#nimages_per_mfov = None # to read it from the dataset (uses image_coordinates.txt in mfov)
nimages_per_mfov = 91

# the wafer ordering here is a logical wafer ordering that indexes into the folders in raw_folders_all.
# the wafer ordering defined there should be the correct physical wafer ordering for all wafers that
#   are to be used in the alignment.
# NOTE: this means that the logical wafer numbers may not match the experimental wafer numbers.
# from the viewpoint of this module, wafer_ids always goes from 1 to number of wafers, inclusive.
total_nwafers = 3 # change this appropriately for each dataset
all_wafer_ids = range(1,total_nwafers+1) # change at your own risk, may not work if unordered

# set this to true to run alignment based on (legacy) Zeiss Zen acquistion.
# False is for the new custom msem acquisition software.
legacy_zen_format = True

# common shared experimental directory subfolder
experiment_subfolder = '2019/briggman/2019-10-17-ZF_retinaa_test'

# store the processed images into single folder per wafer
# NOTE: these are for the processed images, so makes more sense for them to be in numerical order.
#   name could really be anything for each one tho, just defines top level directory for the processed images.
# index 0 defines meta folder, shared alignment folder for all wafers
align_folders_all = ['meta'] + ["wafer{:02d}".format(x) for x in range(1,total_nwafers+1)]

# defined folders underneath of shared folder
meta_subdirs = ['debug_plots', 'debug_plots/sift', 'order_plots', 'region_plots']

# for easier import if module is only using the alignment (or meta) folders
alignment_folders = [os.path.join(root_align, experiment_subfolder, x) for x in align_folders_all]
meta_folder = alignment_folders[0]

# this is for "brute-force" native resolution export.
native_subfolder = 'native'

# this is for storing all 2d alignment dills (residual and two pass info)
align_subfolder = 'alignment2D'

# subfolder for tear stitching, only to be shed individually
tears_subfolder = 'tears'

# a special hook to allow another set of region images to be saved
#  that do no include any of the tile blending / brightness matching features.
noblend_subfolder = 'noblend'

# regions for a single wafer can be stored in multiple experimental folders (yay!)
# these lists are indexed by wafer id, not wafer index
raw_folders_all = [
        None, # index 0 (zeiss does not define a wafer 0, placeholder)
        ['ZF-retina-test-ribbon_20191017_12-27-02',                  # wafer 3 part 1
         'ZF-retina-test-ribbon_20191017_22-33-07',                  # wafer 3 part 2
         'ZF-retina-test-ribbon_20191030_17-31-25',],                # wafer 3 reimage
        ['ribbon1/2019-10-20-ZF-test-ribbon1_20191021_02-24-13',     # wafer 1
         'ribbon1/2019-10-20-ZF-test-ribbon1_20191022_23-35-03',],   # wafer 1 reimage
        ['ribbon2/2019-10-18-ZF_test_ribbon2_20191018_18-20-37',     # wafer 2
         'ribbon2/2019-10-18-ZF_test_ribbon2_20191019_10-31-11',     # wafer 2 reimage 1
         'ribbon2/2019-10-18-ZF_test_ribbon2_20191019_10-37-04',],   # wafer 2 reimage 2
        ]

# NOTE: this array is not necessary at all for the new acquisition format.
# this is a workaround for what is most likely a Zeiss bug. if an error occurs on the first slice imaged
#   in an experiment folder, then the ICS2Stage matrix is not saved at all for that experimental folder.
# in the case that this happened in a "continuation experiment folder" (due to a Zeiss SW crash?),
#   and presumably nothing else happens except the SW is restarted, then the ICS2Stage matrix seems to stay
#   the same and the one from another "part" can be loaded.
# this array indicates which experimental folder to load the protocol.txt from to find the ICS2Stage matrix.
#   default of -1 means from the same experimental folder. array is parallel to raw_folders_all
#index_protocol_folders = [None] + [[-1 for x in range(len(raw_folders_all[y]))] for y in range(1,total_nwafers+1)]
index_protocol_folders = [
        None,
        [-1, 0, -1,],   # wafer 3
        [-1, -1,],      # wafer 1
        [-1, -1, -1,],  # wafer 2
        ]

# use the fullres directory to determine if this is the new acquisition data format
fullres_dir = 'fullres'

# suffix to append to raw folder name (for wafer) to get the manifest name
# indexed by wafer id, not wafer index
manifest_suffixes = [None]*(total_nwafers + 1)

# extension used for loading from image stacks
stack_ext = '.tiff'

# for the new imaging format, start value (base 1, inclusive) in the imageing order for each round of reimaging.
# NOTE: indexed by wafer_id (not index)
reimage_beg_inds = None # for no re-image information or for the legacy zen format

# defined folders underneath of processed folders, created if they do not exist by get_paths
proc_subdirs = [
        'alignment', 'alignment2D',
        'rough_alignment', os.path.join('rough_alignment', 'thumbnails'), os.path.join('rough_alignment', 'masks'),
        'native', 'tears', os.path.join('native', 'tears'),
        'noblend', os.path.join('native', 'noblend'),
        ]

# NOTE: czifiles are not used in the new acquisition format.
# index czifiles by wafer_id
czipath = os.path.join(root_czi, experiment_subfolder)
czifiles = [
    None, # index 0 (typically there is no wafer 0, placeholder)
    'limi-overview-ribbon1a-with-roi.czi',
    'limi-overview-ribbon3-with-roi.czi',
    'limi-overview-ribbon2-with-roi.czi',
    ]
czfiles = [
    None, # index 0 (typically there is no wafer 0, placeholder)
    'ZF_Retina_test_RIBBON1_ROIs.cz',
    'ZF_Retina_test_RIBBON3_ROIs.cz',
    'ZF_Retina_test_RIBBON2_ROIs.cz',
    ]

# need to "manually" define the region manifest for the legacy Zen acqisition format
def generate_manifest():
    # number of regions per wafer: [739, 959, 894] = 2592 slices
    _region_strs_all = [
        None, # index 0 (zeiss does not define a wafer 0, placeholder)
        # wafer 3 slice 526 was imaged twice and the other overlaps are only reimages,
        #   so only exclude part 1 from part 2 (and not part 2 from part 1).
        #[['S%dR%d' % (x,x) for x in _glob_regions_exclude(3,[0],[1,2])], # wafer 3 part 1
        [['S%dR%d' % (x,x) for x in glob_regions_exclude(1,[0],[2])], # wafer 3 part 1
         ['S%dR%d' % (x,x) for x in glob_regions_exclude(1,[1],[0,2])], # wafer 3 part 2
        ## to also use the duplicate of 526 (compare manually after regions aligned)
        #['S%dR%d' % (x,x) for x in _glob_regions_exclude(3,[1],[2])], # wafer 3 part 2
        ['S314R314', 'S511R511', 'S732R732']], # wafer 3 reimage
        [['S%dR%d' % (x,x) for x in range(1,960) if x not in [952,958]], # wafer 1
         ['S952R952', 'S958R958',]], # wafer 1 reimage
        [['S%dR%d' % (x,x) for x in range(1,895) if x not in [125,556,858]], # wafer 2
         ['S125R125', 'S858R858',], # wafer 2 reimage 1
         ['S556R556',]], # wafer 2 reimage 2
    ]
    return _region_strs_all

# region order and exclude strings for filesnames, needed for importing excludes
order_txt_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_region_solved_order.txt')
exclude_txt_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_region_excludes.txt')

# this is to specify regions to exclude completely from the alignment.
# NOTE: indexed by wafer_id (not index)
# NOTE: with the new region_str system, these are 1-based region indices, NOT slice numbers
#exclude_regions = None # for not excluding any additional regions
#exclude_regions = [[] for x in range(total_nwafers+1)]
# original (previous alignment)
#exclude_regions = [[], # no wafer 0, leave blank
#                   [373,528,538,546,], # wafer 3
#                   [47,171,671,950,], # wafer 1
#                   [1,80,553,557,581,759,888,889,892,893,], # wafer 2
#                  ]
# new re-ordering after initial bad matches removed
#exclude_regions = [[], # no wafer 0, leave blank
#                   [ 528, 630,], # wafer 1
#                   [  47, 941, 950,], # wafer 2
#                   [  14,  80, 201, 581, 880, 886,], # wafer 3
#                  ]
# NOTE: for the "re-ordering" test, all of the initial excludes were sucessfully inserted
# for the order solver sensitivity test
exclude_regions = import_exclude_regions()

region_strs_all = [None]*(total_nwafers + 1)
region_rotations_all = [None]*(total_nwafers + 1)
region_reimage_index = [None]*(total_nwafers + 1)
region_manifest_cnts = [None]*(total_nwafers + 1)
region_include_cnts = [None]*(total_nwafers + 1)
init_region_info(all_wafer_ids, root_raw, get_paths, raw_folders_all, alignment_folders, legacy_zen_format,
    reimage_beg_inds, fullres_dir, manifest_suffixes, stack_ext, region_strs_all, region_rotations_all,
    region_reimage_index, region_manifest_cnts, region_include_cnts, exclude_regions, generate_manifest)

# path and filename parameters >>>


# <<< parameters used by multiple scripts

# NOTE: czifiles are not used for the new acquisition format.

# different for different datasets whether scenes or ribbons were used, both indexed by wafer_id
czifile_scenes = [0, 3, 1, 2]
# which ribbon to load from the czifile, index by wafer_id, use zeros if ribbons not used
czifile_ribbons = [0]*(total_nwafers + 1) # no ribbons used in this dataset

# base rotation to add to the czifile rotations.
# defined per wafer because the template from the section mapper can change between wafers.
#czifile_rotations = [None] + [0.]*total_nwafers
czifile_rotations = [None, -11.14, 0., 13.28]
# base translation to add to the translation defined by centering the roi polygon, per wafer.
#roi_polygon_translations = [None] + [[0.,0.]]*total_nwafers
roi_polygon_translations = [None, [-725.,235.], [0.,0.], [580.,150.]]

# use the polygons from the czifile instead of the ROIs.
# this should almost always be False, except for compatibility with some datasets
#   that are only using polygons and not ROIs (!)
czifile_use_roi_polygons = False

# feature for rough run to use template zeiss polygon to calculate initial angle instead of czi angle.
# this requires the template slice to be defined, whose angle is from the czifile plus czi_rotation.
# define template slices per wafer, indexed by wafer_id.
# define template slice as -1 for a particular wafer to turn the feature off.
wafer_template_slice_inds = [None] + [-1]*total_nwafers

# amount to downsample for image processing and "full-res" exports
dsstep = 1

# this loads the zeiss thumbnails as the images to use instead of the full-resolution images.
# specify here what the downsampling factor of the thumbnails is.
#use_thumbnails_ds = 0 # to load the native resolution (do not use thumbnails)
use_thumbnails_ds = 4

# amount to downsample for exporting thumbnails used by the solver (order and rough alignment).
# this is the downsampling (on top of the downsampling used for the alignment) for the SIFT points,
#   i.e. for the order solving and rough alignment.
dsthumbnail = 8

# this suffix is used by wafer for the montaged region ending and ext
#   and also by wafer_solver for the downsampled solver image inputs.
thumbnail_suffix = '_stitched_grid_thumbnail.tiff'

# series of fine alignemnt croppings to use. the purpose is to start at small crop values and run larger ones
#   only for points that were identified as outliers by wafer_aggregator.
# this is an optimization that should save considerable compute time.
crops_um = [[32., 32.],]

# for the fine alignment, specify slice index transitions at which to apply the blurring method
slice_blur_z_indices = [] # to disable
slice_blur_factor = 3. # factor that increases the LOG sigma value when pre-procing image / template

# meta dill for saved parameters that are the same across wafers
meta_dill_fn_str = 'alignment_meta.dill'

# method of interpolating the deltas (griddata) used both by wafer_aggregator (interpolating outliers),
#   and also by wafer (interpolating deltas at all pixel locations).
#region_interp_type_deltas = 'linear'
region_interp_type_deltas = 'cubic'

# parameters used by multiple scripts >>>


# <<< run_regions parameters

# format string that prepends most of the output files
wafer_region_prefix_str = 'wafer{:02d}_{}'

# output file name format strings
slice_balance_fn_str = 'wafer{:02d}_region_brightness.txt'

# suffix to use for saving / loading the montaged region images
region_suffix = '_stitched'

# set cutoff for discarding deltas that are out of expected range, specify in nm relative to "tiled" deltas.
# there are two delta cutoffs, for horizontal adjacencies and for vertical/diagonal adjacencies (inner arrays).
# there are two sets of delta cutoffs:
#   first for image comparisons within the same mfov.
#   second for image comparisons for tiles in different mfovs.
#delta_cutoff = [[np.array([np.inf, np.inf]), np.array([np.inf, np.inf])] for x in range(2)] # for off
#delta_cutoff = [[np.array([0., 0.]), np.array([0., 0.])] for x in range(2)]  # all the way on, uses defaults (zeiss)
# for the twopass method, it is counterproductive making these too tight.
delta_cutoff = [[np.array([768,768]), np.array([768,768])],
                [np.array([15440,13440]), np.array([15440,13440])]] # can be large overlap at mfov boundaries
# ignore correlations for images with low complexity based on variance.
# be careful with this parameter, any image tiles below the cutoff rely on the zeiss coordinates entirely.
#variance_cutoff = 0.
variance_cutoff = 1.
# this is to use the correlation value soft cutoff method that is determined based on the distribution of
#   correlation values for each mfov. there are additional parameters when enabled that are not exposed.
#   see comments / default values in mfov.py
#C_cutoff_soft_nGMM = 0 # disable the soft cutoff
C_cutoff_soft_nGMM = 5 # if enabled, a good value for this parameter, worked for several datasets
# what weighting to put on the default (Zeiss) coordinates for discarded correlations and deltas
#   and also on non-discarded comparisons between mfovs (vs withing mfovs).
# description by inner index:
#   0 - weight for default delta (that replaces discarded delta) for within mfov comparisons
#   1 - weight for default delta (that replaces discarded delta) for between mfov comparisons
#   2 - weight for non-discarded delta for between mfov comparisons (relative to within mfov comparisons)
# description by outer index:
#   0 - weights for standalone mfov alignment or first region pass
#   1 - weights for second region pass
#   2 - weights for final region stitching (all tiles in the region/slice)
#weight_default = [[1., 1., 1.]]*3 # everything has equal weighting relative to valid within mfov comparisons
# dataset went back to normal stage movement pattern (not the snake pattern like MEA)
weight_default = [[0.1, 0.001, 1.], [0.5, 0.001, 0.5], [1., 0.001, 0.1]]

# use a two pass 2D alignment approach that intially computes median deltas and
#   then uses these to both determine and fill in outlier deltas.
region_stitching_twopass = True

# how many regions to average over to compute median deltas.
#   method is to use the nearest imaged regions (in the imaging order).
# NOTE: this value includes the current slice.
#region_stitching_twopass_nregions = 1 # only compute median deltas with mfovs in same slice
region_stitching_twopass_nregions = 11

# for the two pass region alignment, use these values as a tolerance on the defaults for a better method of rejecting
#   outliers in the final per mfov alignment.
# as before, first values are (xtol,ytol) for within mfov, second between mfovs. values are in nm.
#twopass_default_tol_nm = None # to disable and still use tiled deltas as rejection criteria
# important parameter for 2d alignment, see workflow for how to choose appropriate value
twopass_default_tol_nm = [[64,64], [4480,4480]]

# specify for all the brightness methods to ignore areas outside of the scaled roi polygon
use_roi_poly = True

# this fits the top-down brightness decay during run_balance_mean_mfov
# NOTE: not intended to work along with the def_brightness_use_mode_ratio feature.
def_brightness_use_fit_decay = True

# use an averaged image ratio to average image mode to try to divide all brightness artifacts out
# NOTE: recommend turning both def_overlap_correction_borders and def_brightness_use_fit_decay off if using this,
#   as it should deal with all the artifacts in a non-parametric way.
def_brightness_use_mode_ratio = False

# for def_brightness_use_mode_ratio, the block shape to filter the mode ratio image with.
mode_ratio_block_shape_um = (0.96, 0.96)

# for average_region_mfovs brightness adjustment, number of "saturated" pixels to remove at ends of slice histos
brightness_slice_histo_nsat = [5,1]

# maximum range in which to accept a full slice histogram mode, inclusive.
# this list is iterated in order until a peak that is at least a certain size relative to max (unexposed) is found.
brightness_slice_mode_limits = [[128,240],[240,255],[64,128],[32,64],[16,32]]

# defines absolute range for a heuristic for refining limits around the mode value, inclusive
#   the limits are then used in many of the brightness balancing computations in order to define
#     what "legitimate grayscale values" are. this ignores any tiles that will have an outlier effect
#     in the median tile computations for mfov brightness balancing.
# None uses a fixed percentage of the range from the slice histogram peak
#   to the min on each side of the mode respectively.
# negative values indicate to use the histogram value at a local min,
#   i.e. at the first point that the slope changes on the respective side of the mode.
#brightness_slice_absolute_rng = None # to use fixed percentage (see above)
brightness_slice_absolute_rng = [-80, 254]

# this is a final "fail-safe" for slice histograms that are wider than would be expected for a particular dataset
# it limits the cutoffs used based on the slice histogram to this maximum before and after the slice mode.
#brightness_slice_histo_max_rng = [0,0] # to disable
brightness_slice_histo_max_rng = [64,0]

# for brightness_balancing==True
# the number of tiles to use as top-n pairwise xcorrs that go as input to brightness solver.
brightness_balancing_ntop = 32

# the maximum difference between the tile histogram peaks to allow for a match
brightness_balancing_maxlag = 64

# optional parameter to essentially force single adjacency matrix connected component.
#brightness_balancing_nspan = 0 # to disable
brightness_balancing_nspan = 2

# l2 normalization parameter sent to solver for brightness balancing
brightness_balancing_L2_norm = 0. # for off

# breaks each image into this number of tiles for calculating histograms.
# then least square fit the adjust at each location and use this for each tile brightness correction.
# 1x1 can not correct any planar gradient brightness change within single images.
#   unfortunately this also occurs in the actual data.
#brightness_balancing_ntiles = [1,1] # to disable (use single scalar brightness adjustment per msem image)
#brightness_balancing_ntiles = [2,2] # avoids overfitting, does not correct quite as well.
# 3x3 can also fit to the brightness top-down decay, but implemented a seperate fitting method instead.
# Depending on the dataset 3x3 can cause overfitting.
brightness_balancing_ntiles = [3,3] # can work the best, but prone to overfitting

# what degree of polynomial to fit to the brightness balancing tiles.
# parameter is ignored if brightness_balancing_ntiles==[1,1]
brightness_balancing_degree = 2

# optionally randomly partition brightness balancing tiles into chunks that are solved separately.
# this is only needed due to memory limitations creating the "diff matrix" solved by the least squares solver.
# this is overriden if brightness_balancing_chunksize is not None
brightness_balancing_nchunks = 1
# set chunksize to calculate nchunks instead based on desired chunksize
#brightness_balancing_chunksize = None # to disable
brightness_balancing_chunksize = 5000

# for mode brightness_balance_slices_whole how many neighboring slices in the solved order to use (before and after).
# specify 1 or less to use the old mode that does all pairwise comparisons.
#brightness_balance_slices_whole_tiles_nslices = 1 # for all pairwise comparisons
brightness_balance_slices_whole_tiles_nslices = 16

# same parameters as for brightness_balancing for whole slice balancing (brightness_balance_slices_whole)
brightness_balancing_slices_maxlag = 80
# NOTE: ntop, nspan and adj_min are only used if brightness_balance_slices_whole_tiles_nslices==1
#   (for all pairwise comparisons that does not use the solved order)
brightness_balancing_slices_ntop = 128
# should not be any need to enable both nspan and label_adj_min
#brightness_balancing_slices_nspan = 0 # to disable
brightness_balancing_slices_nspan = 4
# specify min component size > -1 to use connected components centering in solver
brightness_balancing_slices_label_adj_min = -1 # to disable

# this is for backloading rois when region_stage_coords are missing (!)
# NOTE: not need for the new acquisition format.
backload_rois = False

# which borders to correct for in array are in order: top, bottom, left, right
# NOTE: works but not sure if better if used in combination
#   with def_brightness_use_mode_ratio and/or mfov decay fitting
#def_overlap_correction_borders = [0,0,0,0] # for no correction, recommand for def_brightness_use_mode_ratio
def_overlap_correction_borders = [0,1,1,1] # do not correct top, recommnend for def_brightness_use_fit_decay
#def_overlap_correction_borders = [1,1,1,1] # to correct all

# blending during the montage
#def_blending_mode = 'None'
def_blending_mode = 'feathering'

# distance in microns for feather (in feathering def_blending_mode only)
blending_mode_feathering_dist_um = 3.072

# approximate min overlap distance between tiles (in feathering def_blending_mode only), specify seperately for x/y
# NOTE: in practice this seems best set 1-2x the actual overlap, depending on the dataset.
# 0.42805 0.51081 actual max overlap for ZF
blending_mode_feathering_min_overlap_dist_um = [0.672, 0.800]

# another method for fixing tears is to try to do it with the slices individually,
#   so that they are fixed in the region images before the rough alignment.
# NOTE: indexed by wafer_id (not index)
# NOTE: with the new region_str system, these are 1-based region indices, NOT slice numbers
torn_regions = None # to disable tear fixing in individual slices / regions

# downsampling level of the tear annotation relative to the regions
tear_annotation_ds = 16

# grid density to use for correspondence points (um), beyond this (pixel dense) uses griddata
tear_grid_density = 0.512

# run_regions parameters  >>>


# <<< run_wafer parameters

# dill file names and subdirectories, these might not all be used, depending on run type
# filled in below to make actual filenames using wafer_id(s)
delta_dill_fn_str = os.path.join('alignment', '{}', 'wafer{:02d}_region_iorder{:05d}.dill')
limi_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_limi_info.dill')

# how the beginning part of the wafer string is formatted. used in most output filenames.
wafer_format_str = 'wafer{:02d}_'

# optionally translates the loaded region images so that they are centered on their zeiss roi polygon.
# for "newer" datasets where polygons were NOT done manually, recommend True.
translate_roi_center = True

# degree range about the starting rotation for delta template match
#delta_rotation_range = [0.,0.] # to disable the fine delta rotations
delta_rotation_range = [-15.,15.]

# degree steps for delta template match
delta_rotation_step = 3

# amount to crop in microns to use as the template for slice to slice template matching.
# this number can not be too large (not above maybe 20 um?) because shear/scale causes bad fits for large areas.
template_crop_um=[4.8,4.8]

# this is a method that prevents having to customize grid points for an roi shape, and more particularly
#   allows different shaped rois for the same experiment.
# grid points outside the scaled roi polygon are not used to calculate deltas.
#roi_polygon_scale = 0. # to disable
roi_polygon_scale = 1.1 # used with fine alignment

# these are for the parameters of the hex grid where the max/min specify the rough bounding box.
# parameters are number of x points, number of y points and point spacing in microns.
rough_bounding_box_xy_spc = [18, 22, 25]

# these are for the parameters of the hex grid to define the rough alignment grid.
# parameters are number of x points, number of y points and point spacing in microns.
rough_grid_xy_spc = [35, 43, 12.5]

# these are for the parameters of the hex grid to define the fine alignment grid.
# parameters are number of x points, number of y points and point spacing in microns.
fine_grid_xy_spc = [33, 41, 12]

# these are optional wafer indexed parameters for the rough bounding box
#   that allow a different rough bounding box are to be exported for the order solving only.
# default None to use rough_bounding_box_xy_spc
wafer_solver_bbox_xy_spc = [None for x in range(total_nwafers+1)]

# these are optional wafer indexed parameters for the rough bounding box
#   that allow a different rough bounding box translation for the order solving only.
# NOTE: for the non-order-solving rough exports, the translation is always zero,
#   relative to the center of the roi polygon or region center (depending on translate_roi_center).
wafer_solver_bbox_trans = [None] + [[0,0] for x in range(total_nwafers)]

# run_wafer parameters >>>


# <<< run_wafer_solver parameters

# dill file names and subdirectories, these might not all be used, depending on run type
keypoints_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_keypoints_{}.{}.dill')
matches_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_matches.{}.dill')
rough_affine_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_rough_affines_skip{}.{}.dill')
rough_rigid_affine_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_rough_affines_skip{}.{}_rigid.dill')

# subfolder underneath the main processed image folder where the thumbnails are saved
thumbnail_subfolders = [os.path.join('rough_alignment', x) for x in ['thumbnails', 'masks']]

# subfolder underneath the main processed image folder where the thumbnails are saved for order solving only
thumbnail_subfolders_order = [x + '_solve_order' for x in thumbnail_subfolders]

# various debug plots save path, relative to meta
debug_plots_subfolder = 'debug_plots'

# for importing masks that define the tissue borders in each slice
tissue_mask_path = None # to disable the tissue mask
tissue_mask_fn_str = '' # uses same name as the thumbnail exports
#tissue_mask_fn_str = 'z{:d}.tif' # if masks were generated from a stack indexed by z
tissue_mask_ds = 64
tissue_mask_min_edge_um = 30
tissue_mask_min_hole_edge_um = 20
tissue_mask_bwdist_um = 0

# <<< parameters for image matching

# sift matching parameters
# lowe_ratio must be > 1, further from one is more conservative (rejects more matches)
lowe_ratio = 1.42 # for affine run (xform fitting)
# max number of sift features to use (number specified to opencv, it returns them sorted by "prominence").
# NOTE: unlimited features really needs faiss-gpu knn_search_method (not exposed).
nfeatures = None # do not limit the number of SIFT features

# heuristic feature to remove sift correspondence points that exceed this threshold of many-to-one mappings
# with ransac method for computing percent matches, probably better to disable this.
max_npts_feature_correspondence = 0 # to disable
#max_npts_feature_correspondence = 32

# parameters for image matching >>>

# <<< parameters for rough alignment

# basically how constrained to make the affine xform fit for the rough alignment of the SIFT points.
#affine_rigid_type = 0 # full affine
affine_rigid_type = 1 # rigid, rotation and translation only
#affine_rigid_type = 2 # rigid uniform scale
#affine_rigid_type = 3 # rigid nonuniform scale

# minimum number of features fit to ransac transform to not reject match.
# also used as threshold for minimum percent matches during image matching.
min_feature_matches = 32

# for the thumbnail affine fitting (ransac parameter)
# IMPORTANT: do NOT use a different value for percent matches with ransac,
#   a loose value can result in slightly out of place slices.
rough_residual_threshold_um = 5.12 # rigid

# cutoff for heuristic to flag bad matches based on spread of the matching points
min_fit_pts_radial_std_um = 0.

# cutoff for heuristic to flag bad matches based on max allowable fitted translation
#max_fit_translation_um = None # for off
max_fit_translation_um = [196.,196.]

# parameters for rough alignment >>>

# this is a method to keep larger context around the thumbnails, but remove the sift keypoints
#   from these context areas so they do not influence the affine fits.
# scale indicates the factor by which to expand or shrink the roi polygon.
# points outside the scaled polygon are removed.
# the keypoints are computed using the largest scale in the list.
#   this means that if you increase the max, you need to recompute the keypoints!
# for the affine fits, the list is iterated as long as bad matches are encountered.
#roi_polygon_scales = [0.] # to disable
roi_polygon_scales = [1.0, 0.9, 0.7]

# which roi polygon scale (index) to use for computing the percent matches matrix.
# defined per wafer, indexed by wafer_id.
matches_iroi_polygon_scales = [None] + [0 for x in range(total_nwafers)]

# whether to compute full percent matches matrix (triu only for False)
matches_full = True

# optionally apply (median) filter before computing keypoints, size in pixels
keypoints_filter_size = 0 # to disable

# optionally rescale image to full range before keypoints.
# this is not recommended for em data, as contrast scaling / matching is part of the workflow.
keypoints_rescale = False

# <<< parallelization of sift keypoint and matches

# total independent node processes used to compute keypoints.
# NOTE: keypoints are loaded dynamically in some cases, so these values have to be controlled independently
#   from the general nprocesses/nworkers command line argument to run_wafer_solver.py
keypoints_nprocesses = 32
# keypoints_nworkers_per_process are parallel threads, each running on a subset of images
#keypoints_nworkers_per_process = 1 # default serial, but nthreads set in cv2 (with --nworkers)
#keypoints_nworkers_per_process = 12 # soma
keypoints_nworkers_per_process = 10 # axon

# parallelization of sift percent matches computations
# total number of workers is ngpus*njobs_per_gpu*nprocesses
matches_gpus = [0] # for single gpu
#matches_gpus = [0,1,2,3] # on soma cluster 4 gpus per node

# parallelization of sift keypoint and matches >>>

# run_wafer_solver parameters >>>


# <<< run_wafer_aggregator parameters

# dill file names and subdirectories, these might not all be used, depending on run type
# filled in below to make actual filenames using wafer_id(s)
fine_dill_fn_str = os.path.join('alignment', 'wafer{:02d}_solved_fine_alignment.{}.dill')
rough_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_rough_alignment.{}.dill')
rough_rigid_dill_fn_str = os.path.join('rough_alignment', 'wafer{:02d}_rough_alignment.{}_rigid.dill')

# distance cutoff between rough_alignment_grid points and original solved affine-fitted points,
#   in order to select which rough_alignment_grid points to use with which to fit reconciled affine.
rough_distance_cutoff_um = 40

# this is the minimum size of the adjacency graph connected compents for the fine solver.
#   i.e. the mimimum number of non-outlier slices required to not discard the solver component.
fine_min_valid_slice_comparisons = 11

# this is whether to reconcile the deltas as foward or reverse.
# reverse is typically what is needed for dst->src mapping required
#   by most implementations of a general coordinate remapping warping.
fine_solve_reverse = True

# minimum percentage of non-outliers (out of included points, i.e. that are inside the roi polygon)
#   remaining to go forward with delta interpolation. if value is > 1 then interpreted as a cutoff.
interp_inliers = 5

# for the interpolation, how many nearest neighbors to use when interpolating.
# this is essentially required for memory / runtime for very large grids.
interp_inlier_nneighbors = 0 # to disable, use all inliers in the slice to interpolate

# apply these relative weightings to the reconcilers that decrease the contribution
#   of the comparison to the final alignment as a function of distance to the neighbor.
rough_neighbor_dist_scale = None # to disable
fine_neighbor_dist_scale = None # to disable
#rough_neighbor_dist_scale = [1., 0.875, 0.75, 0.5]
#fine_neighbor_dist_scale = [1., 0.875, 0.75, 0.5]

# weighting to apply to interpolated deltas, relative to inliers.
fine_interp_weight = 0.5

# same as fine_neighbor_dist_scale but is ONLY applied to the interpolated deltas,
#   and on top of the default weighting for the interpolated deltas.
#fine_interp_neighbor_dist_scale = None # to disable
fine_interp_neighbor_dist_scale = [1., 1., 0.4, 0.1]

# these are for adding a 2D smoothness constraint to the solver.
rough_smoothing_radius_um = 0. # disabled
rough_smoothing_std_um = 0. # disabled, all neighbors weighted equally
rough_smoothing_weight = 1. # means 2d smoothing has equal weight as z-comparisons, if radius > 0
rough_smoothing_neighbors = [0,0] # how many pairwise neighbors, [center vertex, others]
fine_smoothing_radius_um = 0. # disabled
fine_smoothing_std_um = 0. # disabled, all neighbors weighted equally
fine_smoothing_weight = 1. # means 2d smoothing has equal weight as z-comparisons, if radius > 0
fine_smoothing_neighbors = [0,0] # how many pairwise neighbors, [center vertex, others]

# a different method for 2D smoothness that uses an "affine-filter" on the deltas.
# mistakenly used huge filter size relative to 12 um spacing on first run
#fine_filtering_shape_um = [114., 114.] # slightly larger than 8x9 hex arranged grid at 16 spacing
fine_filtering_shape_um = [61., 61.] # slightly larger than 6x6 hex arranged grid at 12 spacing

# whether to remove the bias from the solver by using linear regression (instead of mean)
#rough_regression_remove_bias = False # recommend False, not useful for rigid affine, full still overfits
rough_regression_remove_bias = True  # for some "regular geometry" datasets can help with runaway scale
fine_regression_remove_bias = True   # recommend True, really helps with "runaway" deltas

# another method along with removing bias that helps reducing low frequency solver accumulated bias.
# first is the exact radius for the deltas to solve simultaneously, second is the approximate desired
#   chunk size for the number of deltas (in z) to store for each iteration. first should be >> than second.
# only solve the deltas using (first) amount of context in z,
#   applied in both directions but number simultaneously solved same at the ends of the stack.
z_neighbors_radius = [0,0] # to disable, solve each grid point using all slice and neighbor deltas
#z_neighbors_radius = [360, 120] # == 721 neighboring slices, about 25 um at 35 nm slice thickness

# <<< parameters for the fine alignment outlier detection

# polynomial degree for the affine fit for detecting outliers, over 3 definitely not recommended
outlier_affine_degree = 2

# residual threshold for the ransac affine fit
fine_residual_threshold_um = 12.

# minimum number of non-outlying or excluded neighbors to keep as inlier
inlier_min_neighbors = 0 # to disable

# minimum size of connected components with immediate inlier neighbors to keep as inlier
inlier_min_component_size_edge_um = 0. # to disable

# hard cutoffs for xcorr values.
# indexed by n-1 where n is the distance away from current slice being compared.
C_hard_cutoff = None # to disable hard C cutoff

# iterate hard cutoffs in decreasing order until at least this percentage of included points
#   are not flagged as outliers (aka min percentage of inliers) by the C_cutoff.
# not used if C_hard_cutoff is None
min_percent_inliers_C_cutoff = 0.3

# number of inlier neighbors to calculate z-scores over to decide it "outlier deltas" should be
#   included with a lesser weight or not. second value is the total search area for inliers.
# any negative value here indicates to search for the closest deltas in the neighborhood,
#   not just the closest points. this can help at discontinuous points (e.g., big splits in the slices).
# hexagonal ring diameter [3,5,7,9,11,13] == points [7,19,37,61,91,127]
ninlier_neighhbors_cmp = [-8, 37]

# zscore to include "outlier deltas" with a lesser weight
# if value is less than zero it indicates a residual threshold for the residual (vs median nbhd vector).
#ok_outlier_zscore = 0. # to disable ok outliers, recommended off until final crop
ok_outlier_zscore = -6.

# zscore to do final rejection of some bad inliers, but include in solver with an even lesser weight
# if value is less than zero it indicates a residual threshold for the residual (vs median nbhd vector).
#not_ok_inlier_zscore = 0. # to disable final inlier rejection, recommended off until final crop
not_ok_inlier_zscore = -1.

# another option to flag outliers that are within this distance of inliers.
# this can save compute time in that only these grid points will be rerun at the next crop size.
# xxx - this is most likely no longer necessary now that we have the tissue masks, delete the feature?
#   some masks in wafer_aggregator likely consume a bit of unnecessary memory even when disabled
fine_nearby_points_um = None # to disable

# <<< merge outlier params

# these are for merging outliers for different blocks, if block processing is used for outlier detection.
# the block outlier processing is mostly intended for ultrafine alignment.

# the number of nonoverlap block points that must be included in a block to not count the whole
#   block as an outlier block.
merge_inliers_blk_cutoff = 0 # to disable

# minimum component size for block outliers, removes small clusters that are typically outside of the tissue.
merge_inliers_min_comp = 0 # to disable

# minimum component size for block outlier holes, removes small holes that are typically inside of the tissue.
merge_inliers_min_hole_comp = 0 # to disable

# merge outlier params >>>>

# parameters for the fine alignment outlier detection >>>

# run_wafer_aggregator parameters >>>
