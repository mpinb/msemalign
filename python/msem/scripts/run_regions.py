#!/usr/bin/env python3
import os
os.system('date')

"""run_regions.py

Top level command-line interface for 2D alignment and montaging of image
  tiles within a section.

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

# xxx - this got more complicated than a script, move some of the functionality into a region helper type object
#   one issue here is feature creep led to region being forced to know some things that only wafer should know,
#     the slice ordering for example... ask me about why I love object oriented.

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import argparse
import time
import dill
import logging
import tifffile

import scipy.sparse as sp
import scipy.ndimage as nd
import scipy.interpolate as interp

import multiprocessing as mp
import queue
import skimage.measure as measure
import cv2

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.neighbors import NearestNeighbors
os.system('date')

from msem import region, zimages
from msem.utils import big_img_load, big_img_save, big_img_info, big_img_init, gpfs_file_unlock, tile_nblks_to_ranges
from msem.utils import fill_outside_polygon, block_construct, make_hex_points, mls_rigid_transform, find_histo_mode

# all parameters loaded from an experiment-specific import
from def_common_params import get_paths, native_subfolder, dsstep, use_thumbnails_ds, all_wafer_ids, exclude_regions
from def_common_params import scale_nm, nimages_per_mfov, legacy_zen_format, region_manifest_cnts, reimage_beg_inds
from def_common_params import region_suffix, use_roi_poly, def_brightness_use_fit_decay
from def_common_params import def_brightness_use_mode_ratio, mode_ratio_block_shape_um, brightness_slice_histo_nsat
from def_common_params import brightness_slice_mode_limits, brightness_slice_absolute_rng
from def_common_params import brightness_balancing_ntop, brightness_balancing_maxlag, brightness_balancing_ntiles
from def_common_params import brightness_balancing_degree, brightness_balancing_nchunks, brightness_balancing_chunksize
from def_common_params import brightness_balancing_nspan, brightness_balancing_slices_nspan
from def_common_params import brightness_balance_slices_whole_tiles_nslices, brightness_slice_histo_max_rng
from def_common_params import brightness_balancing_slices_maxlag, brightness_balancing_slices_ntop
from def_common_params import brightness_balancing_slices_label_adj_min, brightness_balancing_L2_norm
from def_common_params import backload_rois, def_overlap_correction_borders, def_blending_mode
from def_common_params import blending_mode_feathering_dist_um, blending_mode_feathering_min_overlap_dist_um
from def_common_params import order_txt_fn_str, limi_dill_fn_str, wafer_region_prefix_str, slice_balance_fn_str
from def_common_params import meta_folder, debug_plots_subfolder, meta_dill_fn_str, align_subfolder
from def_common_params import delta_cutoff, variance_cutoff, weight_default, C_cutoff_soft_nGMM
from def_common_params import region_stitching_twopass, region_stitching_twopass_nregions, twopass_default_tol_nm
from def_common_params import tissue_mask_ds, tissue_mask_fn_str, tissue_mask_min_edge_um, tissue_mask_min_hole_edge_um
from def_common_params import tears_subfolder, torn_regions, tear_annotation_ds, tear_grid_density
from def_common_params import noblend_subfolder
os.system('date')


# <<< turn on stack trace for warnings
#import traceback
#import warnings
##import sys
#os.system('date')
#
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    log = file if hasattr(file,'write') else sys.stderr
#    traceback.print_stack(file=log)
#    log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#warnings.showwarning = warn_with_traceback
#warnings.simplefilter('error', UserWarning) # have the warning throw an exception
# turn on stack trace for warnings >>>


### argparse
parser = argparse.ArgumentParser(description='run_regions')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1], help='specify wafer(s) for the regions to run')
parser.add_argument('--all-wafers', dest='all_wafers', action='store_true',
    help='instead of specifying wafer id(s), include all wafers for dataset')
parser.add_argument('--region_inds', nargs='+', type=int, default=[-1],
    help='list of region indices to run (< 0 for all regions in wafer)')
parser.add_argument('--region-inds-rng', nargs=2, type=int, default=[-1,-1],
    help='if region_inds is not defined, create region_inds from this range (default to region_inds)')
parser.add_argument('--run-type', nargs=1, type=str, default=['align'],
    choices=['align', 'balance', 'export', 'balance-mean-mfov', 'slice-histos', 'slice-balance',
        'slice-brightness-adjust', 'slice-contrast-rescale', 'slice-contrast-match', 'convert-h5-to-tiff',
        'stitch-tears', 'save-target-histo', 'plot-target-histo', 'save-masks'],
    help='the type of run to choose')
parser.add_argument('--twopass_align_first_pass', dest='twopass_firstpass', action='store_true',
    help='specify this is the first pass when using 2D two pass alignment method')
parser.add_argument('--no-brightness-balancing', dest='brightness_balancing', action='store_false',
    help='disable the tile brightness balancing')
parser.add_argument('--no-blending-features', dest='blending_features', action='store_false',
    help='disable all tile balancing and blending features')
parser.add_argument('--re-brightness-balancing', dest='brightness_rebalancing', action='store_true',
    help='run brightness balancing again on top of previous balancing')
parser.add_argument('--custom-suffix', nargs=1, type=str, default=[''],
    help='override the default filename suffix, mostly for testing')
parser.add_argument('--slice-balance', dest='slice_balance', action='store_true',
    help='use saved slice brightness balancing for export')
parser.add_argument('--slice-balance-exclude', dest='use_exclude_slice_balance', action='store_true',
    help='use the exlude regions for the slice balancing (ignore slices in exclude)')
parser.add_argument('--roi-polygon-scale', nargs=1, type=float, default=[1.],
    help='amount to scale the roi poly (ignore areas outside), 0 for disable')
parser.add_argument('--tissue-masks', dest='tissue_masks', action='store_true',
    help='use (or export) the tissue masks')
parser.add_argument('--histos-compute-areas', dest='histos_compute_areas', action='store_true',
    help='when running histograms, also compute the roi/mask union area')
parser.add_argument('--slice-rescale-range', nargs=2, type=float, default=[0.5, 99.5],
    help='histogram limits (<= 1) or grayscale limits for slice-contrast-rescale')
parser.add_argument('--slice-adjust', nargs=1, type=float, default=[0.],
    help='static amount to add to the whole slice brightness adjust')
parser.add_argument('--slice-balance-heuristics', dest='slice_balance_heuristics', action='store_true',
    help='use histo heuristics like in balance-mean-mfov when slice balancing')
parser.add_argument('--native', dest='native', action='store_true',
    help='process native resolution images, not thumbnails')
parser.add_argument('--native-exchange-src', dest='native_exchange_src', action='store_true',
    help='specify to load native/thumbnails exchangable alignment files')
parser.add_argument('--native-exchange-dst', dest='native_exchange_dst', action='store_true',
    help='specify to save native/thumbnails exchangable alignment files')
parser.add_argument('--show-xcorr-plots', dest='show_xcorr_plots', action='store_true',
    help='diplay debug plots of the cross correlations')
parser.add_argument('--save-xcorr-plots', dest='save_xcorr_plots', action='store_true',
    help='save debug plots of the cross correlations')
parser.add_argument('--show-plots', dest='show_plots', action='store_true',
    help='catchall to enable all other optional plots')
parser.add_argument('--save-residuals', dest='save_residuals', action='store_true',
    help='save residuals from mfov and region stitching')
parser.add_argument('--dsexports', nargs='*', type=int, default=[32],
    help='use these downsampling levels for export to tiff')
parser.add_argument('--dsstep', nargs=1, type=int, default=[0],
    help='override dsstep, intended mostly for brightness corrections')
parser.add_argument('--nworkers', nargs=1, type=int, default=[1],
    help='split some operations into multiple threads (same-node processes)')
parser.add_argument('--mfov-id', nargs=1, type=int, default=[0],
    help='which mfov id (base 1) to process (to debug specific mfov)')
parser.add_argument('--false-color-montage', dest='false_color_montage', action='store_true',
    help='export false-color montage')
parser.add_argument('--overlay-montage', dest='overlap_overlay', action='store_true',
    help='export montage with overlay showing overlapping pixels')
parser.add_argument('--export-fg', dest='export_fg', action='store_true',
    help='export mask indicating the foreground of the region')
parser.add_argument('--stage-coords', dest='stage_coords', action='store_true',
    help='output the alignment based on the stage (zeiss) coordinates')
parser.add_argument('--overlap-radius', nargs=1, type=int, default=[-1],
    help='override mfov overlap radius, intended for mfov debugging / param adjust')
# options for blockwise processing, for when slices are memory-limited
parser.add_argument('--nblocks', nargs=2, type=int, default=[1, 1],
    help='number of partitions per dimension for blockwise processing')
parser.add_argument('--iblock', nargs=2, type=int, default=[-1, -1],
    help='which block to process for blockwise processing (zero-based)')
parser.add_argument('--block-overlap-um', nargs=2, type=float, default=[0., 0.],
    help='amount of overlap between blocks in um for blockwise processing')
# options for saving the tissue masks into the region hdf5s
parser.add_argument('--save-masks-in', nargs=1, type=str, default=[''],
    help='input path for the tissue masks')

args = parser.parse_args()
args = vars(args)


### params that are set by command line arguments

# starting at 1 (Zeiss numbering)
region_inds = args['region_inds']
if region_inds[0] < 0 and all([x > -1 for x in args['region_inds_rng']]):
    region_inds = range(args['region_inds_rng'][0],args['region_inds_rng'][1])

# wafer starting at 1, used internally for numbering processed outputs, does not map to any zeiss info
wafer_ids = args['wafer_ids']

# when using the two pass 2D stitching method that is based on comparing against median deltas,
#   and if region_stitching_twopass_nregions > 1, specify that this is the first pass.
twopass_firstpass = args['twopass_firstpass']

# specify this when running export to load the previously calculated between slices balancing
slice_balance = args['slice_balance']

# set True to enable new brightness balancing method between tiles in the entire region
brightness_balancing = args['brightness_balancing']

# set False to disable all brightness balancing and blending features
blending_features = args['blending_features']

# set True to enable loading previous brightness balancing before running brightness balancing
brightness_rebalancing = args['brightness_rebalancing']

# set True to ignore the exlude regions when doing slice balancing
use_exclude_slice_balance = args['use_exclude_slice_balance']

# amount to scale the roi polygon used in several of the brightness / contrast balancing modes
roi_polygon_scale = args['roi_polygon_scale'][0]

# histo range for slice rescale
slice_rescale_range = args['slice_rescale_range']

# static amount to add to the whole slice brightness balancing
slice_adjust = args['slice_adjust'][0]

# helps with problem slices when balancing slices without tissue masks
slice_balance_heuristics = args['slice_balance_heuristics']

# set to non-empty to override the standard output image file suffix
custom_suffix = args['custom_suffix'][0]

# option to process / export native resolution regions
native = args['native']

# options for exchanging alignment files between thumbnail and native runs
native_exchange_src = args['native_exchange_src']
native_exchange_dst = args['native_exchange_dst']

# for converting to tiff exports, use the specified downsampling levels
dsexports = args['dsexports']

# for overriding the default dsstep in which the images are processed.
# this was added mostly for running brightness corrections at different downsamplings.
dsstep_override = args['dsstep'][0]

# show debug plots of the cross correlations
show_xcorr_plots = args['show_xcorr_plots']

# path to optionally save debug plots of the cross correlations
save_xcorr_plots = args['save_xcorr_plots']

# enables several other optional / debugging plots
show_plots = args['show_plots']

# save region residuals to region dill file
save_residuals = args['save_residuals']

# for same-node process parallizations
arg_nworkers = args['nworkers'][0]

# optionally specify particular mfov to process (for debug, not production mode)
mfov_id = args['mfov_id'][0]

# false color aligned montages (for visualization of alignment).
false_color_montage = args['false_color_montage']

# when converting to tiff exports, add overlay that shows overlapping pixels
overlap_overlay = args['overlap_overlay']

# when converting to tiff exports, export foreground mask instead of image
export_fg = args['export_fg']

# optionally export region using the microscope stage coordinates (zeiss)
stage_coords = args['stage_coords']

# optionally override the mfov overlap radius
overlap_radius = args['overlap_radius'][0]

# read in tissues masks for each slice, more refined/precise that roi polygon
tissue_masks = args['tissue_masks']

# when running histos, get the area (in pixels) of the mask used for computing histos
histos_compute_areas = args['histos_compute_areas']

# options for blockwise processing, mostly intended for native
nblks = args['nblocks'][::-1]
iblk = args['iblock'][::-1]
blk_ovlp_um = args['block_overlap_um'][::-1]

# these specify what type of run this is (one of these must be set True)

run_type = args['run_type'][0]

# normal region run type, get alignment deltas for all the mfovs individually.
# then compute best 2d alignment for all tiles in the region.
run_align = run_type == 'align'

# only runs the brightness matching and does not rerun the alignment
run_brightness = run_type == 'balance'

# this mode calculates the average region brightness balancing and top-down decay, run first after alignment.
run_balance_mean_mfov = run_type == 'balance-mean-mfov'

# calculate and save slice histograms. the histograms are saved along with the data in the h5 files.
# they are used by several other features, including the slice brightness and contrast balancing features.
# this was made as a seperate run type because for large slices this can be surprisingly compute intensive.
run_slice_histos = run_type == 'slice-histos'

# after regions have been exported, iterate over all wafer regions (slices)
#   and adjust overall brightness between histograms for all slices in all wafers.
brightness_balance_slices_whole = run_type == 'slice-balance'

# does not recompute 2d alignments, instead only creates the output region images based on saved alignment coords.
# if false, computes 2d alignments and saves alignment coords, but does not export region images.
export_montage = run_type == 'export'

# because the slices are now written with hdf5, this mode is to export downsampled tiffs for easy viewing
convert_hdf5s = run_type == 'convert-h5-to-tiff'

# <<< slice contrast enhancement methods
# these methods are getting closer to something like filtering in that they are not easily reversable and
#   potentially can modify the information content of the data, but unfortunately it's another thing that
#   is not solved on the experimental side; for some datasets both the brightness and contrast in different
#   slices can be quite variable.

# enhances slice contrast by rescaling based on cdf percentages (not currently exposed as params).
run_slice_contrast_rescale = run_type == 'slice-contrast-rescale'

# enhances slice contrast by matching to a target histogram.
# the target histogram needs to have been saved in the meta dill.
run_slice_contrast_match = run_type == 'slice-contrast-match'

# this method only loads existing exported images and adjusts the slice brightness only.
# one can also do this with the export, but this is much more efficient than redoing the whole export.
run_slice_brightness_adjust = run_type == 'slice-brightness-adjust'
# slice contrast enhancement methods >>>

# this mode calculates the average region brightness balancing and top-down decay, run first after alignment.
run_stitch_tears = run_type == 'stitch-tears'

# for saving histogram of specified slice (already computed) into target histo in meta
run_save_target_histo = (run_type == 'save-target-histo' or run_type == 'plot-target-histo')

# for saving downsampled masks into the region hdf5 files
run_save_masks = run_type == 'save-masks'

# arguments for saving downsampled masks into the region hdf5 files
save_masks_in = args['save_masks_in'][0]

# set wafer_ids to contain all wafers, if specified
if args['all_wafers']:
    wafer_ids = list(all_wafer_ids)

print('run_wafer run-type is ' + run_type + ', dsstep override ' + str(dsstep_override) + ', wafer_ids:')
print(wafer_ids)
no_iblock = any([x < 0 for x in iblk]) # for debugging blocked histograms
if no_iblock: iblk = [0,0]
# special mode to initialize h5 and h5 lock files when writing out regions in blocks
init_locks = any([x < 0 for x in nblks])
if init_locks: nblks = [abs(x) for x in nblks]
single_block = all([x == 1 for x in nblks])
first_block = all([x == 0 for x in iblk])
last_block = all([x == y-1 for x,y in zip(iblk,nblks)])
tblks = np.prod(np.array(nblks))
lock = not single_block
if not single_block:
    print('iblk {} {} of {} {} ovlp {} {} um'.format(iblk[0],iblk[1],nblks[0],nblks[1],blk_ovlp_um[0],blk_ovlp_um[1]))
if mfov_id > 0:
    print('only processing mfov id {} (debug)'.format(mfov_id))

## fixed parameters not exposed in def_common_params

# this is the number of resolutions that are saved in the brightness adjustment file.
# this is a hook so that brightness adjustmensts can be applied across different downsamplings.
# the order of these resolutions is:
#   (1) mode-ratio (2) histogram range
nres_nms = 2

# The data type of the coordinates (must be prevented from type-casting to default double) has major
#   memory implications. Because the coordinate transforms work on pixel values, the precision is not
#   that important. float16 however, would not allow the range required for large slices.
# This is only used by the remapping done by run-type stitch-tears (slice tear stitching).
#coordindate_transformations_dtype = np.float32
xdtype = np.float32


### inits based on params

dsstep_import = dsstep
if dsstep_override > 0: dsstep = dsstep_override

# where to save optional cross correlation debug plots
plots_folder = os.path.join(meta_folder, debug_plots_subfolder)

# stores some info used for entire dataset
meta_dill_fn = os.path.join(meta_folder, meta_dill_fn_str)

# to avoid checking torn_regions is None below
if torn_regions is None: torn_regions = [[] for x in range(max(all_wafer_ids)+1)]

# # support for reading the tissue masks (to filter the keypoints)
# if tissue_masks:
#     assert(tissue_mask_path is not None)
# else:
#     tissue_mask_path = None

if blending_features:
    overlap_correction_borders = def_overlap_correction_borders
    blending_mode = def_blending_mode
    brightness_use_fit_decay = def_brightness_use_fit_decay
    brightness_use_mode_ratio = def_brightness_use_mode_ratio
    use_noblend_subfolder = ''
else:
    assert( not run_brightness )
    assert( not brightness_balance_slices_whole )
    overlap_correction_borders = [0,0,0,0]
    blending_mode = 'None'
    brightness_use_fit_decay = False
    brightness_use_mode_ratio = False
    brightness_balancing = False
    brightness_rebalancing = False
    slice_balance = False
    use_noblend_subfolder = noblend_subfolder

assert( not (brightness_use_mode_ratio and brightness_use_fit_decay) ) # do not work together correctly

#nwafers = len(wafer_ids)
# for lists that are indexed by wafer_id (and not by wafer index)
mwafers = max(all_wafer_ids)+1

# for annoying parts of this script that have to be aware of wafers
# xxx - how to shield the slices from knowledge of wafers / the whole dataset stack???
cum_manifest_cnts = np.cumsum(region_manifest_cnts[1:])

if overlap_radius < 0: overlap_radius = None

any_slice_histo = (run_slice_histos or brightness_balance_slices_whole or run_save_masks)
if not any_slice_histo:
    # START - enable logging
    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    # END - enable logging

any_slice_histo_adjust = (run_slice_contrast_rescale or run_slice_contrast_match or run_slice_brightness_adjust)
any_slice_histo_adjust = (any_slice_histo_adjust or run_save_target_histo)

# for parallelizing slice histogram calculations
def compute_histos_job(ind, inds, fns, rois_points, use_tissue_mask, tissue_mask_min_size, tissue_mask_min_hole_size,
        dsstep, nblks, use_iblk, result_queue, compute_area, doplots, verbose):
    if verbose: print('\tworker%d: started' % (ind,))
    nimgs = len(fns); doinit = True
    single_block = all([x==1 for x in nblks])
    for i in range(nimgs):
        histo = None; area = 0
        # xxx - somehow gpfs created a situation where everything was normal:
        #   (1) empty stderr file (2) special message printed (3) all jobs completed normally
        #   and still one of the h5 files had empty blocks that were not written.
        write_count_unique = write_count_expected = -1
        if fns[i] is not None:
            # decided to support blockwise processing serially. did not see a situation where slices become so
            #   large that this is prohibitive (diferent slices parallized by workers here and also by processes).
            tblks = np.prod(np.array(nblks))
            for b in range(tblks):
                iblk = np.unravel_index(b, nblks)
                if use_iblk is not None and not all([x == y for x,y in zip(use_iblk, iblk)]): continue
                if verbose: print('Loading {}, block {} {}'.format(fns[i],iblk[0],iblk[1])); t = time.time()
                attrs={'write_mask':None} if write_count_unique < 0 else write_count_unique
                img, img_shape, blk_rng = big_img_load(fns[i], nblks=nblks, iblk=iblk, return_rng=True, attrs=attrs)
                if write_count_unique < 0:
                    write_count_unique = int((attrs['write_mask'] > 0).sum())
                    write_count_expected = int(np.prod(np.array(attrs['write_mask'].shape)))
                if verbose:
                    print('\tdone loading in %.4f s' % (time.time() - t, ))
                    print('img blk is {}x{}'.format(img.shape[1],img.shape[0]))
                    print('full img is {}x{}'.format(img_shape[1],img_shape[0]))
                # NOTE: do the downsampling before zero'ing outside the polygon,
                #   because the roi has already been downsampled by region.
                if dsstep > 1:
                    if verbose: print('Downsampling {}'.format(dsstep)); t = time.time()
                    pad = (dsstep - np.array(img.shape) % dsstep) % dsstep
                    img = measure.block_reduce(np.pad(img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                            block_size=(dsstep, dsstep), func=zimages.blkrdc_func).astype(img.dtype)
                    if verbose: print('\tdone downsampling in %.4f s' % (time.time() - t, ))
                cpts = rois_points[i] if rois_points is not None else None
                if not single_block and cpts is not None:
                    cpts = cpts - np.array([x[0] for x in blk_rng])[::-1]/dsstep
                if cpts is not None:
                    if verbose: print('Fill 0 outside polygon'); t = time.time()
                    img = fill_outside_polygon(img, cpts, docrop=(single_block and not use_tissue_mask))
                    if verbose: print('\tdone filling in %.4f s' % (time.time() - t, ))
                if use_tissue_mask:
                    if verbose: print('Fill 0 outside mask {}'); t = time.time()
                    attrs={'ds':0}; bw, _ = big_img_load(fns[i], dataset='tissue_mask', attrs=attrs)
                    if attrs['ds'] < dsstep:
                        rel_ds = dsstep // attrs['ds']
                        pad = (rel_ds - np.array(bw.shape) % rel_ds) % rel_ds
                        bw = measure.block_reduce(np.pad(bw, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                block_size=(rel_ds, rel_ds), func=zimages.blkrdc_func).astype(bw.dtype)
                        bw_ds = rel_ds * attrs['ds']
                        rel_ds = 1
                    else:
                        bw_ds = attrs['ds']
                        rel_ds = attrs['ds'] // dsstep

                    if tissue_mask_min_size > 0:
                        # remove small components
                        labels, nlbls = nd.label(bw, structure=nd.generate_binary_structure(2,2))
                        if nlbls > 0:
                            sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                            rmv = np.nonzero(sizes < tissue_mask_min_size)[0] + 1
                            if rmv.size == nlbls:
                                # keep the largest label if they are all smaller than cutoff
                                rmv = rmv[rmv != np.argmax(sizes) + 1]
                            if rmv.size > 0:
                                bw[np.isin(labels, rmv)] = 0

                    if tissue_mask_min_hole_size > 0:
                        # remove small holes
                        labels, nlbls = nd.label(np.logical_not(bw),
                            structure=nd.generate_binary_structure(2,1))
                        if nlbls > 0:
                            sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                            add = np.nonzero(sizes < tissue_mask_min_hole_size)[0] + 1
                            if add.size > 0:
                                bw[np.isin(labels, add)] = 1

                    # save memory by inverting the mask before upsampling.
                    bw = np.logical_not(bw)
                    # have to crop the image down because of the padding for cubing the regions.
                    shp = np.array(img_shape)
                    crp = (shp + (rel_ds - shp % rel_ds) % rel_ds) // rel_ds
                    bw = bw[:crp[0],:crp[1]]

                    if not single_block:
                        if verbose: print('full bw tm is {}x{}'.format(bw.shape[1],bw.shape[0]))
                        # crop is based on original image load before downsampling
                        rng = [[int(np.floor(x[0]/bw_ds)), int(np.ceil(x[1]/bw_ds))] for x in blk_rng]
                        bw = bw[rng[0][0]:rng[0][1],rng[1][0]:rng[1][1]]
                    if verbose: print('bw tm blk is {}x{}'.format(bw.shape[1],bw.shape[0]))

                    if rel_ds > 1:
                        if verbose: print('Upsampling mask {}'.format(rel_ds)); t = time.time()
                        bw = block_construct(bw, rel_ds)
                        if verbose: print('\tdone upsampling in %.4f s' % (time.time() - t, ))
                        if verbose: print('bw tm is {}x{}'.format(bw.shape[1],bw.shape[0]))
                    # need to crop again after upsampling because of padding for downsampling
                    bw = bw[:img.shape[0], :img.shape[1]]

                    # zero out everything outside of mask (as with polygon)
                    img[bw] = 0
                    if compute_area:
                        nbw = np.logical_not(bw)
                        nbw = fill_outside_polygon(nbw, cpts)
                        area = nbw.sum(dtype=np.int64)
                #if use_tissue_mask:

                if verbose: print('Img is {}x{}'.format(img.shape[1],img.shape[0]))

                # for validation, keeping for reference / debug
                if doplots:
                    plt.figure(); plt.gcf().clf()
                    plt.imshow(img, cmap='gray')
                    plt.show()

                if doinit:
                    # this is just to prevent passing or hard-coding the image datatype.
                    imgmax = np.iinfo(img.dtype).max; doinit = False
                if b==0:
                    histo = np.zeros((imgmax+1,),dtype=np.int64)
                if verbose: print('Compute histo'); t = time.time()
                img = np.ravel(img); chisto = np.bincount(img, minlength=imgmax+1); del img
                histo += chisto
                if verbose: print('\tdone with histo in %.4f s' % (time.time() - t, ))
            #for b in range(tblks):
        #if fns[i] is not None:
        res = {'ind':inds[i], 'histo':histo, 'area':area, 'write_count_unique':write_count_unique,
                'write_count_expected':write_count_expected, 'iworker':ind}
        result_queue.put(res)
    #for i in range(nimgs):
    if verbose: print('\tworker%d: completed' % (ind,))
#def compute_histos_job(

def load_brightness_other(fn, strkey, ftype):
    print("load_brightness_other '{}'".format(fn))
    vals = None
    strkey = '#'+strkey
    with open(fn, 'r') as f:
        for line in f:
            sline = line.strip()
            if not sline: continue
            line = sline.split()
            if line[0] == strkey:
                vals = np.array([ftype(x) for x in line[1:]])
                break
    return vals

# utility function for supporting mode ratio at lower downsamplings
def downsample_mode_ratio(imgs, ds):
    if ds > 1:
        print('Downsampling mode-ratios at ds=%d' % (ds,)); t = time.time()
        pad = (ds - np.array(imgs.shape[-2:]) % ds) % ds
        for i in range(imgs.shape[0]):
            oimg = measure.block_reduce(np.pad(imgs[i,:], ((0,pad[0]), (0,pad[1])), mode='reflect'),
                    block_size=(ds, ds), func=zimages.blkrdc_func).astype(imgs.dtype)
            if i==0: oimgs = np.zeros((imgs.shape[0], oimg.shape[0], oimg.shape[1]), dtype=imgs.dtype)
            oimgs[i,:] = oimg
        imgs = oimgs
        print('\tdone in %.4f s' % (time.time() - t, ))
    return imgs


### precompute the histo_dill_fns names
# this is for modes that use the order information for brightness balancing
load_order = brightness_balance_slices_whole and brightness_balance_slices_whole_tiles_nslices > 1
histo_dill_fns = [None]*mwafers
cum_missing_region_inds = np.zeros((0,), dtype=np.int64)
if load_order:
    region_ind_to_solved_order = [None]*mwafers
    missing_region_inds = [None]*mwafers
    cum_nsolved_order = 0
    cum_nregions = 0
    cum_to_solved_order = np.zeros((0,), dtype=np.int64)
    cum_from_solved_order = np.zeros((0,), dtype=np.int64)
for wafer_id in all_wafer_ids:
    experiment_folders, thumbnail_folders, protocol_folders, alignment_folder, _, region_strs = get_paths(wafer_id)
    nregions = sum([len(x) for x in region_strs])
    # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
    region_strs_flat = [item for sublist in region_strs for item in sublist]

    # this is outside the region loop so that regions do not have to be loaded for brightness_balance_slices_tiles
    histo_dill_fns[wafer_id] = [os.path.join(alignment_folder, wafer_region_prefix_str.format(wafer_id,
            region_strs_flat[x]) + '_tile_histograms.dill') for x in range(nregions)]
    order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))

    if load_order:
        # save the tile histograms for each region in the solved order.
        solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        # get the inverse lookup for the solved order, put -1's for missing regions
        region_ind_to_solved_order[wafer_id] = -np.ones(nregions, np.int64)
        region_ind_to_solved_order[wafer_id][solved_order] = np.arange(solved_order.size)
        sel = (region_ind_to_solved_order[wafer_id] >= 0)
        region_ind_to_solved_order[wafer_id][sel] = region_ind_to_solved_order[wafer_id][sel] + cum_nsolved_order
        missing_region_inds[wafer_id] = np.nonzero(np.logical_not(sel))[0]
        cum_nsolved_order += solved_order.size

        # create cumulative forward and inverse mappings, including missing regions.
        to_solved_order = np.concatenate((solved_order, missing_region_inds[wafer_id]))
        assert( to_solved_order.size == nregions )
        from_solved_order = np.zeros_like(to_solved_order)
        from_solved_order[to_solved_order] = np.arange(nregions)
        cum_to_solved_order = np.concatenate((cum_to_solved_order, to_solved_order+cum_nregions))
        cum_missing_region_inds = np.concatenate((cum_missing_region_inds,
                missing_region_inds[wafer_id]+cum_nregions))
        cum_nregions += nregions


### the iteration over wafers is intended mostly for intraslice brightness balancing.
slices_histos_doinit = True
slice_balance_fns = [None]*mwafers
for wafer_id in wafer_ids:
    experiment_folders, thumbnail_folders, protocol_folders, alignment_folder, _, region_strs = get_paths(wafer_id)
    nregions = sum([len(x) for x in region_strs])
    slice_histos_fns = [None]*nregions
    slice_histos_roi_polys = [None]*nregions
    slice_histos_include_fns = [None]*nregions
    #slice_histos_mask_fns = [None]*nregions
    # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
    region_strs_flat = [item for sublist in region_strs for item in sublist]

    # skip the region loop entirely for slice brightness balancing
    # in this mode region_ind is interpreted as a 1-based index
    if run_stitch_tears and region_inds[0] < 0:
        cregion_inds = torn_regions[wafer_id]
    else:
        cregion_inds = region_inds if region_inds[0] > -1 else range(1,nregions+1)

    if any_slice_histo:
        print('slice histos: enumerating %d regions in wafer %d' % (len(cregion_inds), wafer_id,))
        t = time.time()
    if backload_rois:
        print('Using backloaded reconstructed roi polygons if region_stage_coords is missing')
        limi_dill_fn = os.path.join(alignment_folder, limi_dill_fn_str.format(wafer_id))
        with open(limi_dill_fn, 'rb') as f: limi_dict = dill.load(f)
        backload_roi_polys = limi_dict['imaged_order_region_recon_roi_poly_raw']
    elif len(cregion_inds) > 0:
        backload_roi_polys = [None]*max(cregion_inds)
    for region_ind in cregion_inds:
        if stage_coords:
            to_native_scale = 1./use_thumbnails_ds/dsstep
            use_use_thumbnails_ds = use_thumbnails_ds
            use_tissue_mask_ds = tissue_mask_ds
            use_tear_annotation_ds = tear_annotation_ds
        else:
            if native:
                to_native_scale = use_thumbnails_ds / dsstep
                use_use_thumbnails_ds = 0
                use_tissue_mask_ds = tissue_mask_ds * use_thumbnails_ds
                use_tear_annotation_ds = tear_annotation_ds * use_thumbnails_ds
            else:
                to_native_scale = 1./dsstep
                use_use_thumbnails_ds = use_thumbnails_ds
                use_tissue_mask_ds = tissue_mask_ds
                use_tear_annotation_ds = tear_annotation_ds
        mfov_align_init = not any_slice_histo and not any_slice_histo_adjust
        cregion = region(experiment_folders, protocol_folders, region_strs, region_ind,
            thumbnail_folders=thumbnail_folders, dsstep=dsstep, brightness_balancing=brightness_balancing,
            overlap_correction_borders=overlap_correction_borders, nimages_per_mfov=nimages_per_mfov,
            blending_mode_feathering_dist_um=blending_mode_feathering_dist_um, overlap_radius=overlap_radius,
            blending_mode_feathering_min_overlap_dist_um=blending_mode_feathering_min_overlap_dist_um,
            use_thumbnails_ds=use_use_thumbnails_ds, mfov_align_init=mfov_align_init,
            backload_roi_poly_raw=backload_roi_polys[region_ind-1], mfov_ids=([mfov_id] if mfov_id > 0 else None),
            false_color_montage=false_color_montage, D_cutoff=delta_cutoff, V_cutoff=variance_cutoff,
            W_default=weight_default, legacy_zen_format=legacy_zen_format, scale_nm=scale_nm,
            tissue_mask_ds=use_tissue_mask_ds, tissue_mask_min_edge_um=tissue_mask_min_edge_um,
            tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um, C_cutoff_soft_nGMM=C_cutoff_soft_nGMM,
            verbose=True)
        if cregion.imfov_diameter <= 1:
            print('WARNING: skipping region {}, bad number of mfovs {}'.format(cregion.region_str,
                cregion.imfov_diameter))
            assert( legacy_zen_format ) # fatal error for the new format
            continue
        if mfov_align_init:
            tmp = cregion.images_shape[::-1] - cregion.max_delta_zeiss
            tmp /= (1e3/(cregion.scale_nm*cregion.dsstep))
            print('max overlap um %g %g' % (tmp[0], tmp[1]))
        # this is for the x-scale on anything that fits to pixel coordinates (brightness matching).
        # for indexing details, see comment for nres_nms (fixed parameters).
        res_nm = cregion.scale_nm*dsstep; res_nms = [res_nm]*nres_nms
        # for any run-types that are using blockwise processing
        blk_ovlp_pix = np.round(np.array(blk_ovlp_um) / cregion.scale_nm / cregion.dsstep * 1000).astype(np.int64)
        # used for two pass region (2D) alignment
        if twopass_default_tol_nm is not None:
            twopass_default_tol = [[x / cregion.scale_nm / cregion.dsstep for x in twopass_default_tol_nm[y]] \
                for y in range(2)]

        # output / input files per-region depending on run-type
        prefix = wafer_region_prefix_str.format(wafer_id, cregion.region_str)
        mfov_str = '' if mfov_id < 1 else ('_mfov{}'.format(mfov_id))
        use_region_suffix = (custom_suffix if custom_suffix else region_suffix) #\
        #    if blending_features else region_suffix_noblend
        if stage_coords:
            coords_fn = os.path.join(cregion.region_folder, "full_image_coordinates.txt")
        else:
            coords_fn = os.path.join(alignment_folder, prefix + mfov_str + '_coords.txt')
        slice_mean_mfovs_fn = os.path.join(alignment_folder, prefix + '_mmfov.tiff')
        thumb_brightness_fn = os.path.join(alignment_folder, prefix + '_brightness.txt')
        thumb_slice_mode_ratio_fn = os.path.join(alignment_folder, prefix + '_scale.tiff')
        slice_twopass_dill_fn = os.path.join(alignment_folder, align_subfolder,
                prefix + mfov_str + '_alignment_deltas.dill')
        slice_alignment_dill_fn = os.path.join(alignment_folder, align_subfolder,
                prefix + mfov_str + '_alignment.dill')
        slice_alignment_nomfov_dill_fn = os.path.join(alignment_folder, align_subfolder, prefix + '_alignment.dill')
        tear_annotation_fn = os.path.join(alignment_folder, tears_subfolder, prefix + '_annotation.dill')
        # basically all the alignment files for native are saved into a subfolder,
        #   with the exception that some files can be exchanged between thumbnail (normal) and native alignment.
        native_alignment_folder = os.path.join(alignment_folder, native_subfolder)
        native_coords_fn = os.path.join(native_alignment_folder, prefix + '_coords.txt')
        native_brightness_fn = os.path.join(native_alignment_folder, prefix + '_brightness.txt')
        native_slice_mode_ratio_fn = os.path.join(native_alignment_folder, prefix + '_scale.tiff')
        if native:
            slice_balance_fns[wafer_id] = os.path.join(native_alignment_folder, slice_balance_fn_str.format(wafer_id))
            slice_image_bfn = os.path.join(native_alignment_folder, use_noblend_subfolder, prefix + use_region_suffix)
            stitched_slice_bfn = os.path.join(native_alignment_folder, tears_subfolder, prefix + use_region_suffix)
            src_brightness_fn = dst_brightness_fn = native_brightness_fn
            src_slice_mode_ratio_fn = dst_slice_mode_ratio_fn = native_slice_mode_ratio_fn

            if native_exchange_src:
                src_brightness_fn = thumb_brightness_fn
                src_slice_mode_ratio_fn = thumb_slice_mode_ratio_fn
            if native_exchange_dst:
                dst_brightness_fn = thumb_brightness_fn
                dst_slice_mode_ratio_fn = thumb_slice_mode_ratio_fn
        else:
            slice_balance_fns[wafer_id] = os.path.join(alignment_folder, slice_balance_fn_str.format(wafer_id))
            slice_image_bfn = os.path.join(alignment_folder, use_noblend_subfolder, prefix + use_region_suffix)
            stitched_slice_bfn = os.path.join(alignment_folder, tears_subfolder, prefix + use_region_suffix)
            src_brightness_fn = dst_brightness_fn = thumb_brightness_fn
            src_slice_mode_ratio_fn = dst_slice_mode_ratio_fn = thumb_slice_mode_ratio_fn
            if native_exchange_src:
                src_brightness_fn = native_brightness_fn
                src_slice_mode_ratio_fn = native_slice_mode_ratio_fn
            if native_exchange_dst:
                dst_brightness_fn = native_brightness_fn
                dst_slice_mode_ratio_fn = native_slice_mode_ratio_fn
        #microscope_coords_fn = os.path.join(cregion.region_folder, "full_image_coordinates.txt")
        slice_image_fn = slice_image_bfn + '.h5'
        stitched_slice_fn = stitched_slice_bfn + '.h5'
        # SO 45211650/very-slow-writing-of-a-slice-into-an-existing-hdf5-datased-using-h5py
        slice_image_rewrite_fn = slice_image_bfn + '.new.h5'

        if not run_stitch_tears and region_ind in torn_regions[wafer_id]:
            slice_image_fn = stitched_slice_bfn + '.h5'
            slice_image_rewrite_fn = stitched_slice_bfn + '.new.h5'

        if use_roi_poly and cregion.roi_poly is not None and roi_polygon_scale > 0:
            if region_ind == cregion_inds[0]:
                print('Using roi poly with scale factor {}'.format(roi_polygon_scale))
            # for the options that use the polygons, scale it by the specified factor parameter
            pts = cregion.roi_poly; ctr = cregion.roi_poly_ctr
            croi_poly = (pts - ctr)*roi_polygon_scale + ctr
        else:
            if use_roi_poly and roi_polygon_scale > 0:
                print('WARNING: no roi polygon (maybe missing coords file?)')
            croi_poly = None

        if slice_balance or run_slice_brightness_adjust:
            loaded_slice_adjust = zimages.load_slice_balance_file(slice_balance_fns[wafer_id], cregion.region_slcstr)
            print('Loaded overall slice brightness adjust of %.3f' % (loaded_slice_adjust,))
            print('Adding static slice brightness adjust of %.3f' % (slice_adjust,))
            slice_adjust += loaded_slice_adjust

        # decided to always export so the slices are sorted by the manifest index.
        # this allows easy conversion from a dataset stored as a z-slice stack.
        export_prefix = 'wafer{:02d}_manifest{:05d}_{}'.format(wafer_id, cregion.region_ind, cregion.region_str)
        slice_dsimage_pfn = export_prefix + use_region_suffix + '.tiff'

        if export_montage:
            ## hack to skip already exported files (for example wrong wafer counts in swarm)
            ## normally comment this
            #if os.path.isfile(slice_image_fn): continue

            if init_locks:
                print('Initializing h5 and h5 locks files for region export')
                big_img_init(slice_image_fn)
                print('exiting')
                print('Twas brillig, and the slithy toves') # for --check-msg not to report failure
                sys.exit(0)

            print('Montaging region from saved coords at:')
            print(coords_fn)
            # load the stitched coords and format properly for montage
            cregion.load_stitched_region_image_coords(coords_fn, scale=to_native_scale, rmv_thb=native,
                add_thb=stage_coords)

            decay_params = None
            if (cregion.brightness_balancing or slice_balance) and os.path.isfile(src_brightness_fn):
                print('Brightness balancing from saved brightness adjustments at:')
                print(src_brightness_fn)
                param_dict = {'resolution_nm':float}
                if brightness_use_fit_decay: param_dict['tile-decay']=float
                param_dict = cregion.load_stitched_region_image_brightness_adjustments(src_brightness_fn,
                    param_dict=param_dict, rmv_thb=native and native_exchange_src,
                    add_thb=not native and native_exchange_src)
                if not cregion.brightness_balancing:
                    print('Zeroing brightness balancing for slice_balance only')
                    cregion.mfov_adjust.fill(0)
                # see comment regarding resolution_nm under fixed parameters.
                #old_res_nms = load_brightness_other(src_brightness_fn, 'resolution_nm', float)
                old_res_nms = param_dict['resolution_nm']
                rescales = [int(x // y) for x,y in zip(res_nms, old_res_nms)]

                # optionally montage using the fitted top-down decay from run_balance_mean_mfov
                if brightness_use_fit_decay:
                    print('Using per-tile decay params')
                    decay_params = param_dict['tile-decay'].reshape((-1,2))
                    cregion.mfov_decay_params = decay_params
            if slice_adjust > 0: cregion.mfov_adjust[:,:,0] += slice_adjust

            if brightness_use_mode_ratio:
                print('Using image scaled brightness adjustments at:')
                print(src_slice_mode_ratio_fn)
                assert(rescales[0] >= 1) # only implemented downsampling for mode-ratio

                # new method, different adjust for each mfov tile
                mean_mfovs_mode_ratio = tifffile.imread(src_slice_mode_ratio_fn)
                # applied during montage instead of tiling over all mfovs (tiling is huge memory waste).
                cregion.mfov_scale_adjust = downsample_mode_ratio(mean_mfovs_mode_ratio, rescales[0])

            # nmfovs, tiles per mfov, ndims
            cregion.create_region_coords_from_inner_rect_coords(cregion.region_coords)
            if native and first_block:
                # write out image coordinates to text file in zeiss format
                # this is mostly here for ease for wafer, so that wafer itself does not have to be aware of native,
                #   but only to point it to the native alignment folder.
                zimages.write_image_coords(native_coords_fn, cregion.stitched_region_fns,
                    cregion.stitched_region_coords)

            # generate the stitched image (the "montage") for the whole region (slice) using the coordindate and
            #   brightness information loaded above.
            image, corners, crop_info, overlap_sum = cregion.montage(dsstep=dsstep, blending_mode=blending_mode,
                    get_overlap_sum=True, res_nm=res_nm, nblks=nblks, iblk=iblk, novlp_pix=blk_ovlp_pix)

            if mfov_id < 1:
                img_shape = crop_info['noncropped_size'][::-1]
                print('Saving image {}x{}, background (and empty histogram)'.format(img_shape[0],img_shape[1]))
                t = time.time()
                write_count, f1, f2 = big_img_save(slice_image_fn, image, img_shape, nblks=nblks, iblk=iblk,
                        novlp_pix=blk_ovlp_pix, compression=True, recreate=True, lock=lock, keep_locks=lock, wait=True)
                big_img_save(slice_image_fn, overlap_sum==0, img_shape, nblks=nblks, iblk=iblk, novlp_pix=blk_ovlp_pix,
                        dataset='background', compression=True, recreate=True, f1=f1, f2=f2)
                big_img_save(slice_image_fn, overlap_sum, img_shape, nblks=nblks, iblk=iblk, novlp_pix=blk_ovlp_pix,
                        dataset='overlap_count', compression=True, recreate=True, f1=f1, f2=f2)
                if first_block:
                    # create the histogram dataset now. it will be computed more than once after image transformations.
                    histo = np.zeros((np.iinfo(image.dtype).max+1), dtype=np.int64)
                    big_img_save(slice_image_fn, histo, histo.shape, dataset='histogram', recreate=True)
                    # xxx - hacky, save slots for downsampled image histograms
                    for ds in [2,4,8,16,32,64]:
                        big_img_save(slice_image_fn, histo, histo.shape, dataset='histogram_'+str(ds), recreate=True)
                if lock: gpfs_file_unlock(f1,f2)
                print('\tdone in %.4f s' % (time.time() - t, ))
            else:
                print('Saving single mFOV for debug:')
                fn = os.path.join(plots_folder, prefix + '_mfov{}.tiff'.format(mfov_id))
                print(fn)
                tifffile.imwrite(fn, image)

        elif run_balance_mean_mfov: # run-type if/elif
            print('Averaging and exporting slice mean mfov'); t = time.time()

            mean_mfovs_mode_ratio = decay_params = None
            if brightness_rebalancing:
                print('\tComputing mode balancing on top of mode ratio or decay')
                print(src_slice_mode_ratio_fn)
                print(src_brightness_fn)
                assert(brightness_use_mode_ratio or brightness_use_fit_decay)

                if brightness_use_mode_ratio:
                    # new method, different adjust for each mfov tile
                    mean_mfovs_mode_ratio = tifffile.imread(src_slice_mode_ratio_fn)

                if brightness_use_fit_decay:
                    decay_params = load_brightness_other(src_brightness_fn, 'tile-decay', float).reshape((-1,2))

            # NOTE: overlap corrections used in zimages are not loaded in mfov load_images, only done during montage.
            #   this is set on or off with overlap_correction_borders parameter above
            # options for mean_mfov_return_type: None, 'mean', 'mode', 'mode-ratio', 'counts'
            mean_mfov_return_type=None
            mean_mfov, mfov_adjusts, mfov_fns, histo_rng, fitted_decay_params, mean_mfovs_mode_ratio = \
                    cregion.average_region_mfovs(use_heuristics=True, scale_tiled_coords_factor=1.05, res_nm=res_nm,
                    mode_ratio_block_shape_um=mode_ratio_block_shape_um, fit_decay=brightness_use_fit_decay,
                    bwdist_heuristic_um=0.032, mean_mfov_return_type=mean_mfov_return_type,
                    histo_nsat=brightness_slice_histo_nsat, mode_limits=brightness_slice_mode_limits,
                    absolute_rng=brightness_slice_absolute_rng, mean_mfovs_mode_ratio=mean_mfovs_mode_ratio,
                    decay_params=decay_params, offset=slice_adjust if slice_adjust != 0 else None,
                    slice_histo_max_rng=brightness_slice_histo_max_rng, doplots=show_plots)
            # this is only for display / debug / validation.
            # in production set mean_mfov_return_type=None so that this file is not saved.
            if mean_mfov is not None: tifffile.imwrite(slice_mean_mfovs_fn, mean_mfov)
            if brightness_use_mode_ratio and not brightness_rebalancing:
                # put mfovs modes into a 3d ndarray, so they can be saved as a tiff stack.
                mean_mfovs_mode_ratio = np.concatenate([x[:,:,None] for x in mean_mfovs_mode_ratio], axis=2)
                # save as single precision (do not need double anyways) and as an image stack.
                mean_mfovs_mode_ratio = mean_mfovs_mode_ratio.astype(np.single).transpose((2,0,1))
                tifffile.imwrite(dst_slice_mode_ratio_fn, mean_mfovs_mode_ratio)

            # save other associated parameters as comments in brightness adjust file
            # always save the resolution that this brightness adjustment was run at.
            #   this matters for the polynomial fit brightness as they are a function of pixel coordinates.
            # always save the "histogram range" that estimates the instensity range that contains
            #   most of the information in this slice.
            #   see comment regarding resolution_nm under fixed parameters.
            param_dict = {'resolution_nm':res_nms, 'histogram_range':histo_rng}
            # optionally append the decay parameters to the brightness file
            if brightness_use_fit_decay:
                param_dict['tile-decay'] = (decay_params if decay_params is not None else fitted_decay_params).\
                        reshape(-1)

            # these can be saved as brightness adjusts first
            #   and optionally loaded before the normal brightness matching algorith is run.
            zimages.write_image_coords(dst_brightness_fn, mfov_fns, mfov_adjusts, param_dict=param_dict,
                rmv_thb=not native and native_exchange_dst, add_thb=native and native_exchange_dst)

            print('\tdone in %.4f s' % (time.time() - t, ))

        elif run_align or run_brightness: # run-type if/elif
            if not run_brightness:
                doplots = (show_xcorr_plots or save_xcorr_plots)
                dosave_path = plots_folder if save_xcorr_plots else ''

                # xxx - expose as option? it is the method for deciding which xcorr to use
                #   in the case of duplicates due to mfov neighbor overlap.
                cmp_method='res' # seems to work better
                #cmp_method='C'

                # whether this is two pass and if so, which pass we are running.
                cregion_stitching_twopass = region_stitching_twopass and (cregion.nmfov_ids > 1)
                ctwopass_multiple_regions = cregion_stitching_twopass and region_stitching_twopass_nregions > 1
                ctwopass_firstpass = twopass_firstpass and ctwopass_multiple_regions

                if ctwopass_firstpass:
                    mfovDx, mfovDy, mfovC, delta_coords = cregion.align_and_stitch_mfovs(twopass=True,
                            twopass_firstpass=True, cmp_method=cmp_method, default_tol=twopass_default_tol,
                            nworkers=arg_nworkers, doplots=doplots, dosave_path=dosave_path)

                    d = {
                        'mfovDx':mfovDx, 'mfovDy':mfovDy, 'mfovC':mfovC, 'delta_coords':delta_coords
                        }
                    with open(slice_twopass_dill_fn, 'wb') as f: dill.dump(d, f)
                else:
                    if ctwopass_multiple_regions:
                        # xxx - this block is even uglier than usual...
                        #   if the need for information from multiple slices is really ultimately
                        #     required for 2D alignment then we need to rethink the whole hierarchy

                        # first load the first pass dill just for this region
                        with open(slice_twopass_dill_fn, 'rb') as f: d = dill.load(f)
                        mfovDx = d['mfovDx']; mfovDy = d['mfovDy']; mfovC = d['mfovC']
                        delta_coords = d['delta_coords']

                        # next iterate the regions that were imaged closest to this one,
                        #   and combine all the deltas.
                        n = region_stitching_twopass_nregions
                        all_mfovDx = [None]*n; all_mfovDy = [None]*n
                        all_mfovDx[0] = mfovDx; all_mfovDy[0] = mfovDy
                        region_strs_fn = [os.path.split(x)[1] for x in region_strs_flat]
                        if legacy_zen_format:
                            # NOTE: for the legacy format slices are not compared in the imaged order.
                            #   one could use the subdirectories as "reimages" but some are quite.
                            #   there did not seem to be a need for this based on the 2d alignments,
                            #     and this code paht (legacy_zen_format) is mostly deprecated.
                            region_nums = [int(x.split('_')[1][1:].split('R')[0]) for x in region_strs_fn]
                            region_num = int(cregion.region_str.split('_')[1][1:].split('R')[0])
                        else:
                            region_nums = [int(x.split('_')[0]) for x in region_strs_fn]
                            region_num = int(cregion.region_numstr)
                        if reimage_beg_inds is not None:
                            lims_reimage_beg_inds = np.concatenate(([0], reimage_beg_inds[wafer_id],
                                    [nregions + reimage_beg_inds[wafer_id][-1]]))
                            ireimage = np.nonzero(region_num < lims_reimage_beg_inds)[0][0]
                            region_nums = [x for x in region_nums if x >= lims_reimage_beg_inds[ireimage-1]]
                            region_nums = [x for x in region_nums if x < lims_reimage_beg_inds[ireimage]]
                        region_nums = np.array(region_nums)
                        region_num = np.array(region_num)
                        assert( region_nums.shape == np.unique(region_nums).shape )
                        knbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(region_nums.reshape((-1,1)))
                        knnbrs = knbrs.kneighbors(region_num.reshape((-1,1)), return_distance=False).reshape(-1)
                        knnbrs = np.sort(knnbrs); knnbrs = knnbrs[region_nums[knnbrs] != region_num]
                        for i in range(n-1):
                            cslice_twopass_dill_fn = os.path.join(alignment_folder, align_subfolder,
                                    wafer_region_prefix_str.format(wafer_id, region_strs_fn[knnbrs[i]]) + \
                                    '_alignment_deltas.dill')
                            print(cslice_twopass_dill_fn)
                            with open(cslice_twopass_dill_fn, 'rb') as f: d = dill.load(f)
                            all_mfovDx[i+1] = d['mfovDx']; all_mfovDy[i+1] = d['mfovDy']
                        # get medians over all mfovs for multiple slices
                        mfovDx_med = np.nanmedian(np.concatenate(all_mfovDx, axis=1), axis=1)
                        mfovDy_med = np.nanmedian(np.concatenate(all_mfovDy, axis=1), axis=1)
                    else:
                        mfovDx = mfovDy = mfovC = delta_coords = mfovDx_med = mfovDy_med = None

                    if save_residuals:
                        # comment for debug see region.py comments - xxxalgnbypass
                        mfov_residuals, mfov_residuals_xy, mfov_residuals_triu, mfov_residuals_orig, xcorrs, \
                                imgvars, mfov_deltas, mfov_deltas_xy, mfov_deltas_triu, mfovDx, mfovDy, \
                                sel_btw_mfovs = cregion.align_and_stitch_mfovs(get_residuals=True,
                                twopass=cregion_stitching_twopass, default_tol=twopass_default_tol,
                                cmp_method=cmp_method, mfovDx=mfovDx, mfovDy=mfovDy, mfovC=mfovC,
                                delta_coords=delta_coords, mfovDx_med=mfovDx_med, mfovDy_med=mfovDy_med,
                                nworkers=arg_nworkers, doplots=doplots, dosave_path=dosave_path)
                        residuals, residuals_xy, residuals_triu = cregion.mfov_stitch(get_residuals=True)

                        d = {
                            'residuals':residuals, 'residuals_xy':residuals_xy, 'residuals_triu':residuals_triu,
                            'mfov_residuals':mfov_residuals, 'mfov_residuals_xy':mfov_residuals_xy,
                            'mfov_residuals_triu':mfov_residuals_triu, 'mfov_residuals_orig':mfov_residuals_orig,
                            'mfov_deltas':mfov_deltas, 'mfov_deltas_xy':mfov_deltas_xy,
                            'mfov_deltas_triu':mfov_deltas_triu, 'xcorrs':xcorrs, 'img_variances':imgvars,
                            'mfovDx':mfovDx, 'mfovDy':mfovDy, 'sel_btw_mfovs':sel_btw_mfovs
                            }
                        with open(slice_alignment_dill_fn, 'wb') as f: dill.dump(d, f)
                    else:
                        # comment for debug see region.py comments - xxxalgnbypass
                        cregion.align_and_stitch_mfovs(twopass=region_stitching_twopass, cmp_method=cmp_method,
                            default_tol=twopass_default_tol,  mfovDx=mfovDx, mfovDy=mfovDy, mfovC=mfovC,
                            delta_coords=delta_coords, mfovDx_med=mfovDx_med, mfovDy_med=mfovDy_med,
                            nworkers=arg_nworkers, doplots=doplots, dosave_path=dosave_path)
                        cregion.mfov_stitch()
                    # write out image coordinates to text file in zeiss format
                    if mfov_id < 1:
                        zimages.write_image_coords(coords_fn, cregion.stitched_region_fns,
                            cregion.stitched_region_coords)
                    else:
                        zimages.write_image_coords(coords_fn, cregion.mfov_filenames[mfov_id-1],
                            cregion.mfov_coords_independent[mfov_id-1])
                # else - if ctwopass_firstpass:
            else: # if not run_brightness:
                cregion.load_stitched_region_image_coords(coords_fn, scale=to_native_scale, rmv_thb=native)

            if cregion.brightness_balancing:
                adjust = 0. # default, no "pre"-adjust
                # can optionally load previous slice brightness balancing and run on top of that result
                # NOTE: this is only supported for one balance per image, can not run on top of a previous
                #   tiled brightness adjustment.
                decay_params = histo_rng = None
                if brightness_rebalancing:
                    print('Brightness (re-)balancing from saved brightness adjustments at:')
                    print(src_brightness_fn)
                    param_dict = {'resolution_nm':float, 'histogram_range':int}
                    if brightness_use_fit_decay: param_dict['tile-decay']=float
                    param_dict = cregion.load_stitched_region_image_brightness_adjustments(src_brightness_fn,
                        param_dict=param_dict, rmv_thb=native and native_exchange_src,
                        add_thb=not native and native_exchange_src)
                    #old_res_nms = load_brightness_other(src_brightness_fn, 'resolution_nm', float)
                    old_res_nms = param_dict['resolution_nm']
                    rescales = [int(x // y) for x,y in zip(res_nms, old_res_nms)]
                    # use decay params when brightness balancing, if enabled
                    if brightness_use_fit_decay:
                        print('Using per-tile decay params')
                        decay_params = param_dict['tile-decay'].reshape((-1,2))
                        cregion.mfov_decay_params = decay_params

                    #histo_rng = load_brightness_other(src_brightness_fn, 'histogram_range', int)
                    histo_rng = param_dict['histogram_range']
                    if brightness_use_mode_ratio:
                        print('Using image scaled brightness adjustments at:')
                        print(src_slice_mode_ratio_fn)
                        assert(rescales[0] >= 1) # only implemented downsampling for mode-ratio

                        # different adjust for each mfov tile
                        mean_mfovs_mode_ratio = tifffile.imread(src_slice_mode_ratio_fn)
                        # NOTE: do not tile for all mfovs in region, waste of memory
                        cregion.mfov_scale_adjust = downsample_mode_ratio(mean_mfovs_mode_ratio, rescales[0])

                # this has be to be called after BOTH loads (coords and brightnesses)
                cregion.create_region_coords_from_inner_rect_coords(cregion.region_coords)
                if brightness_rebalancing:
                    adjust = cregion.stitched_region_adjusts
                    adjust[np.logical_not(np.isfinite(adjust))] = 0 # remove the nans b/c of rect ordering

                ihistos = cregion.tile_brightness_balancing(dsstep=dsstep, ntop=brightness_balancing_ntop,
                    ntiles_per_img=brightness_balancing_ntiles, degree=brightness_balancing_degree,
                    nchunks=brightness_balancing_nchunks, chunksize=brightness_balancing_chunksize,
                    maxlag=brightness_balancing_maxlag, offset=adjust, histo_roi_polygon=croi_poly,
                    xcorrs_nworkers=arg_nworkers, res_nm=res_nm, nspan=brightness_balancing_nspan,
                    L2_norm=brightness_balancing_L2_norm, doplots=show_plots)

                # allow downsampling exchange only of the mode ratio. computing it at a lower resolution
                # did not work as well empirically, so do not see any good reason to support upsampling.
                if brightness_use_mode_ratio and not native_exchange_dst and not native and not brightness_rebalancing:
                    old_res_nms[0] = res_nm
                    tifffile.imwrite(dst_slice_mode_ratio_fn, cregion.mfov_scale_adjust)
                # save other associated parameters as comments in brightness adjust file
                # always save the resolution that this brightness adjustment was run at.
                # always save the "histogram range" that estimates the instensity range that contains
                #   most of the information in this slice.
                param_dict = {'resolution_nm':old_res_nms, 'histogram_range':histo_rng}
                # optionally append the decay parameters to the brightness file
                if brightness_use_fit_decay: param_dict['tile-decay'] = decay_params.reshape(-1)
                # write out brightness adjustments to text file in zeiss format
                zimages.write_image_coords(dst_brightness_fn, cregion.stitched_region_fns,
                    cregion.stitched_region_adjusts, param_dict=param_dict,
                    rmv_thb=not native and native_exchange_dst, add_thb=native and native_exchange_dst)

        elif any_slice_histo: # run-type if/elif
            slice_histos_fns[region_ind-1] = slice_image_fn
            slice_histos_roi_polys[region_ind-1] = croi_poly
            if not use_exclude_slice_balance or region_ind not in exclude_regions[wafer_id]:
                slice_histos_include_fns[region_ind-1] = slice_image_fn

            if run_save_masks:
                # create the full filename for the masks
                # have to do this if you did not sort the region exports by manifest index before cubing.
                #use_region_ind = argsort(region_strs_flat).index(region_ind-1)
                # or if they were
                use_region_ind = region_ind-1
                tind = use_region_ind if wafer_id < 2 else use_region_ind + cum_manifest_cnts[wafer_id-2]
                pfn = tissue_mask_fn_str.format(tind) if tissue_mask_fn_str else slice_dsimage_pfn
                fn = os.path.join(save_masks_in, pfn)
                bw = tifffile.imread(fn).astype(bool)
                # have to crop the image down because of the padding for cubing the regions.
                rel_ds = use_tissue_mask_ds // dsstep
                img_shape, _ = big_img_info(slice_image_fn)
                shp = np.array(img_shape)
                crp = (shp + (rel_ds - shp % rel_ds) % rel_ds) // rel_ds
                bw = bw[:crp[0],:crp[1]]

                print("Saving mask '{}' to '{}'".format(fn, slice_image_fn))
                big_img_save(slice_image_fn, bw, img_shape=bw.shape, dataset='tissue_mask',
                    compression=True, recreate=True, overwrite_dataset=True, attrs={'ds':use_tissue_mask_ds})

        elif any_slice_histo_adjust:
            if init_locks:
                print('Initializing h5 and h5 locks files for region histo adjustments')
                big_img_init(slice_image_rewrite_fn)
                print('exiting')
                print('Twas brillig, and the slithy toves') # for --check-msg not to report failure
                sys.exit(0)

            print('Loading slice, background and overlap'); t = time.time()
            if not run_save_target_histo:
                img, img_shape = big_img_load(slice_image_fn, nblks=nblks, iblk=iblk)
                bg, _ = big_img_load(slice_image_fn, nblks=nblks, iblk=iblk, dataset='background')
                #overlap, _ = big_img_load(slice_image_fn, nblks=nblks, iblk=iblk, dataset='overlap_count')
            histo, histo_shape = big_img_load(slice_image_fn, dataset='histogram')
            print('\tdone in %.4f s' % (time.time() - t, ))

            rescale_range_histo = all([x <= 1 for x in slice_rescale_range])
            if (run_slice_contrast_rescale and rescale_range_histo) or \
                    run_slice_contrast_match or run_save_target_histo:
                assert(histo.sum() > 0) # you probably forgot to run slice-histos, or (xxx) this is an excluded slice

                # remove saturated pixels at ends.
                histo_nsat = brightness_slice_histo_nsat
                # replace with the next bin
                histo[:histo_nsat[0]] = histo[histo_nsat[0]]
                histo[-histo_nsat[1]:] = histo[-histo_nsat[1]-1]
                # # replace with zeros
                # histo[:histo_nsat[0]] = 0; histo[-histo_nsat[1]:] = 0

                if run_slice_contrast_match:
                    # load the pre-saved target histogram
                    with open(meta_dill_fn, 'rb') as f: d = dill.load(f)
                    target_histogram = d['target_histogram' + ('_native' if native else '')]
                # optionally and heuristically deal with double peaked histograms.
                # this can happen for example if part of the slice is missing and bare wafer
                #   is imaged where the slice is missing.
                if run_slice_contrast_match and slice_balance_heuristics:
                    # xxx - expose these? in average_region_mfovs they are defaults, probably should standardize?
                    mode_rel = 0.1
                    histo_smooth_size = 5
                    mode_limits = brightness_slice_mode_limits
                    mode, smoothed_histo = find_histo_mode(histo, mode_limits, mode_rel, histo_smooth_size)
                    if mode < 0:
                        print('WARNING: slice_balance_heuristics specified and really ugly region, mode {}'.\
                            format(np.argmax(histo)))
                    else:
                        # look for a low end second peak and remove it from the histo
                        dsmoothed_histo = np.diff(smoothed_histo)
                        ilo_increase = np.nonzero(dsmoothed_histo[mode-1::-1] < 0)[0]
                        if ilo_increase.size > 0:
                            ilo_increase = mode - ilo_increase[0]
                            rel_height = histo[ilo_increase::-1].max() - histo[ilo_increase]
                            if rel_height / histo[mode] > mode_rel:
                                # replace the second peak with just a linear ramp
                                histo = histo / histo.sum()
                                histo[:ilo_increase+1] = np.linspace(target_histogram[0],
                                        histo[ilo_increase], ilo_increase+1)
                slice_cdf = np.cumsum(histo) / histo.sum()
            # if (run_slice_contrast_rescale and rescale_range_histo) or \
            #         run_slice_contrast_match or run_save_target_histo:

            if run_slice_contrast_rescale: # run-type if/elif
                img_dtype = img.dtype; imgmax = np.iinfo(img_dtype).max
                rng = slice_rescale_range
                if rescale_range_histo:
                    # interpret as histogram cdf limits
                    rng = [np.nonzero(slice_cdf > rng[0])[0][0], np.nonzero(slice_cdf > rng[1])[0][0]]
                else:
                    # interpret as grayscale limits
                    assert( rng[0] >= 0 and rng[1] <= imgmax ) # specified grayscale limits out of image dtype range
                if show_plots:
                    print(rng); plt.subplot(1,2,1); plt.plot(histo); plt.subplot(1,2,2); plt.plot(slice_cdf); plt.show()
                assert( rng[0] != rng[1] ) # unbelieveably bad or single gray value slice???

                print('Applying slice rescaling'); t = time.time()
                img = img.astype(np.float32)
                img -= rng[0]; img *= imgmax/(rng[1] - rng[0])
                img = np.round(img); img = np.clip(img, 0, imgmax); img = img.astype(img_dtype); img[bg] = 0
                print('\tdone in %.4f s' % (time.time() - t, ))

            elif run_slice_contrast_match: # run-type if/elif
                hsum = target_histogram.sum()
                assert(hsum > 0) # all zero target histogram???
                target_cdf = np.cumsum(target_histogram) / hsum

                print('Applying slice contrast matching'); t = time.time()
                img_dtype = img.dtype; imgmax = np.iinfo(img_dtype).max; bins = np.arange(imgmax+1)
                # histogram matching using cdfs, interp method pulled from skimage source.
                img = np.round(np.interp(slice_cdf, target_cdf, bins)).astype(img_dtype)[img]; img[bg] = 0
                print('\tdone in %.4f s' % (time.time() - t, ))

            elif run_slice_brightness_adjust:
                print('Applying slice adjustment {}'.format(slice_adjust)); t = time.time()
                img_dtype = img.dtype; imgmax = np.iinfo(img_dtype).max; img = img.astype(np.float32)
                img += slice_adjust
                img = np.round(img); img = np.clip(img, 0, imgmax); img = img.astype(img_dtype); img[bg] = 0
                print('\tdone in %.4f s' % (time.time() - t, ))

            elif run_save_target_histo:
                pdf = histo/histo.sum()
                print(histo.sum()); plt.figure(); plt.plot(histo)
                print(pdf.sum()); plt.figure(); plt.plot(pdf)
                if os.path.isfile(meta_dill_fn):
                    with open(meta_dill_fn, 'rb') as f: d = dill.load(f)
                else:
                    d = {}
                target_histo_key = 'target_histogram' + ('_native' if native else '')
                target_histo_slice_key = target_histo_key + '_slice'
                if run_type == 'plot-target-histo':
                    plt.plot(d[target_histo_key], 'r')
                    print('Current saved histo template slice ' + d[target_histo_slice_key])
                plt.show()
                if run_type != 'plot-target-histo':
                    print('Saving histo template slice "' + target_histo_slice_key + '"')
                    d[target_histo_key] = pdf
                    d[target_histo_slice_key] = prefix
                    with open(meta_dill_fn, 'wb') as f: dill.dump(d, f)
                    print('target histo saved to ' + meta_dill_fn)
                print('exiting')
                print('Twas brillig, and the slithy toves') # for --check-msg not to report failure
                sys.exit(0)

            # SO 45211650/very-slow-writing-of-a-slice-into-an-existing-hdf5-datased-using-h5py
            # have to write adjusted data to a new hdf5.
            print('Saving image, background (and empty histogram)'); t = time.time()
            write_count, f1, f2 = big_img_save(slice_image_rewrite_fn, img, img_shape, nblks=nblks, iblk=iblk,
                compression=True, recreate=True, lock=lock, keep_locks=lock, wait=True)
            big_img_save(slice_image_rewrite_fn, bg, img_shape, nblks=nblks, iblk=iblk, dataset='background',
                compression=True, recreate=True, f1=f1, f2=f2)
            # overlap not used above, save memory by loading it right before re-saving it
            overlap, _ = big_img_load(slice_image_fn, nblks=nblks, iblk=iblk, dataset='overlap_count')
            big_img_save(slice_image_rewrite_fn, overlap, img_shape, nblks=nblks, iblk=iblk,
                    dataset='overlap_count', compression=True, recreate=True, f1=f1, f2=f2)
            if first_block:
                # create the histogram dataset now, as it will be computed more than once after image transformations.
                histo = np.zeros(histo_shape, dtype=np.int64)
                big_img_save(slice_image_rewrite_fn, histo, histo.shape, dataset='histogram', recreate=True)
                # xxx - hacky, save slots for downsampled image histograms
                for ds in [2,4,8,16,32,64]:
                    big_img_save(slice_image_rewrite_fn, histo, histo.shape, dataset='histogram_'+str(ds),
                        recreate=True)
                # copy the tissue mask if it has been saved
                try:
                    bw, _ = big_img_load(slice_image_fn, dataset='tissue_mask')
                    save_tissue_mask = True
                except:
                    save_tissue_mask = False
                if save_tissue_mask:
                    big_img_save(slice_image_rewrite_fn, bw, img_shape=bw.shape, dataset='tissue_mask',
                        compression=True, recreate=True, overwrite_dataset=True, attrs={'ds':use_tissue_mask_ds})
            if lock: gpfs_file_unlock(f1,f2)
            print('\tdone in %.4f s' % (time.time() - t, ))
            # if we just wrote the last block, delete the original h5 and move the new h5 to the original h5.
            #if last_block: # not guaranteed when blocks are not written serially
            if write_count == tblks:
                os.remove(slice_image_fn)
                os.rename(slice_image_rewrite_fn, slice_image_fn)

        elif convert_hdf5s:
            print('Loading slice'); t = time.time()
            # xxx - did not enable blockwise processing here
            if tissue_masks:
                attrs = {'ds':0}
                bw, _ = big_img_load(slice_image_fn, dataset='tissue_mask', attrs=attrs)
                rel_ds = attrs['ds'] // dsstep
                img_shape, img_dtype = big_img_info(slice_image_fn)
                img_shape = np.array(img_shape)
            else:
                image, _ = big_img_load(slice_image_fn, dataset='background' if export_fg else 'image')
                overlay_count = True # overlay scaled count on red channel, False for boolean overlap only
                if overlap_overlay:
                    overlap, _ = big_img_load(slice_image_fn, dataset='overlap_count')

            print('\tdone in %.4f s' % (time.time() - t, ))

            export_root = os.path.join(meta_folder, 'region_exports')
            if export_fg:
                export_root = os.path.join(export_root, 'foreground')
                image = np.logical_not(image)
            if tissue_masks:
                export_root = os.path.join(export_root, 'masks')
            native_str = 'native' if native else ''
            noblend_str = '' if blending_features else '_noblend'
            for j in range(len(dsexports)):
                dsthumbnail = dsexports[j]
                export_str = 'wafer{:02d}{}{}_{}ds{:d}'.format(wafer_id, use_region_suffix, noblend_str, native_str,
                        dsthumbnail)
                export_path = os.path.join(export_root, export_str)
                os.makedirs(export_path, exist_ok=True)
                slice_dsimage_fn = os.path.join(export_path, slice_dsimage_pfn)

                print('Downsampling / exporting thumbnail at ds=%d' % (dsthumbnail,)); t = time.time()
                if tissue_masks:
                    if dsthumbnail > rel_ds:
                        assert(dsthumbnail % rel_ds == 0)
                        ds = dsthumbnail // rel_ds
                        pad = (ds - np.array(bw.shape) % ds) % ds
                        bw = np.round(measure.block_reduce(np.pad(bw, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                block_size=(ds, ds), func=zimages.blkrdc_func)).astype(img.dtype)
                    elif dsthumbnail < rel_ds:
                        assert(rel_ds % dsthumbnail == 0)
                        ds = rel_ds // dsthumbnail
                        bw = block_construct(bw, ds)
                    else:
                        bw = bw.copy()
                    # modify shape to match expected shape of downsampled image
                    cimg_shape = np.ceil(img_shape / dsthumbnail).astype(np.uint64)
                    oimg = bw[:cimg_shape[0], :cimg_shape[1]]
                    if (np.array(bw.shape) != cimg_shape).any():
                        oimg = np.zeros(cimg_shape, dtype=bw.dtype)
                        oimg[:bw.shape[0], :bw.shape[1]] = bw
                elif dsthumbnail > 1:
                    pad = (dsthumbnail - np.array(image.shape) % dsthumbnail) % dsthumbnail
                    oimg = np.round(measure.block_reduce(np.pad(image, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                            block_size=(dsthumbnail, dsthumbnail), func=zimages.blkrdc_func)).astype(image.dtype)
                    if overlap_overlay:
                        ovlp = measure.block_reduce(np.pad(overlap, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                block_size=(dsthumbnail, dsthumbnail), func=np.max).astype(image.dtype)
                else:
                    oimg = image
                    if overlap_overlay: ovlp = overlap
                if overlap_overlay:
                    print('\tadding overlap overlay')
                    oimg = cv2.cvtColor(oimg, cv2.COLOR_GRAY2RGB)
                    tmp = np.zeros(ovlp.shape + (3,), dtype=np.uint8)
                    if overlay_count:
                        ovlp[ovlp > 0] -= 1
                        ovlp = (ovlp / ovlp.max() * 255).astype(np.uint8)
                        print(ovlp.min(), ovlp.max(), np.unique(ovlp))
                        tmp[:,:,0] = ovlp
                    else:
                        ovlp = (ovlp > 1); tmp[ovlp,0] = 255
                    ovlp = tmp
                    oimg = cv2.addWeighted(ovlp, 0.6 if overlay_count else 0.25, oimg, 1, 0.0)
                    #oimg = cv2.addWeighted(ovlp, 1.2, oimg, 1, 0.0)

                    # also overlay the roi polygon
                    line_thickness = 3 #; circle_rad = 5
                    iroi_points = np.round(croi_poly/dsthumbnail).astype(np.int32)
                    cv2.polylines(oimg, [iroi_points.reshape((-1,1,2))], True, (255,0,0), line_thickness)

                # xxx - add another parameter?
                ## export foreground by also applying roi polygon
                #if export_fg: oimg = fill_outside_polygon(oimg, croi_poly/dsthumbnail)

                tifffile.imwrite(slice_dsimage_fn, oimg)
            print('\tdone in %.4f s' % (time.time() - t, ))

        elif run_stitch_tears:
            if region_ind not in torn_regions[wafer_id]:
                print('Region {} not torn'.format(region_ind))
                continue

            if init_locks:
                print('Initializing h5 and h5 locks files for stitched tear fixes')
                print(stitched_slice_fn)
                big_img_init(stitched_slice_fn)
                print('exiting')
                print('Twas brillig, and the slithy toves') # for --check-msg not to report failure
                sys.exit(0)

            img_shape, img_dtype = big_img_info(slice_image_fn)
            img_shape = np.array(img_shape)

            dpix = tear_grid_density / cregion.scale_nm / cregion.dsstep * 1000
            nx = int(np.floor((img_shape[1] + dpix/2) / dpix))
            ny = int(np.floor(img_shape[0] / (dpix*np.sqrt(3)/2))) + 1
            # make grid points
            grid = make_hex_points(nx, ny, dpix)
            grid -= grid.min(0)
            # add evenly points around the edges so grid fully covers the image shape
            # right
            pts = np.arange(0,img_shape[0],dpix*np.sqrt(3)/2)
            pts = np.concatenate((np.ones((pts.size,1), dtype=np.double)*(img_shape[1]-1), pts[:,None]), axis=1)
            grid = np.concatenate((grid, pts), axis=0)
            # bottom
            pts = np.arange(0,img_shape[1],dpix)
            pts = np.concatenate((pts[:,None], np.ones((pts.size,1), dtype=np.double)*(img_shape[0]-1)), axis=1)
            grid = np.concatenate((grid, pts), axis=0)
            # left
            pts = np.arange(dpix*np.sqrt(3)/2,img_shape[0],dpix*np.sqrt(3))
            pts = np.concatenate((np.zeros((pts.size,1), dtype=np.double), pts[:,None]), axis=1)
            grid = np.concatenate((grid, pts), axis=0)
            # bottom right corner
            grid = np.concatenate((grid, (np.array(img_shape[::-1])-1)[None,:]), axis=0)

            #plt.figure(1010); plt.gcf().clf()
            #plt.scatter(grid[:,0], grid[:,1], c='g', s=12, marker='.')
            #plt.gca().invert_yaxis()
            #plt.gca().set_aspect('equal')
            #plt.show()

            with open(tear_annotation_fn, 'rb') as f: cdict = dill.load(f)
            #d = {'correspondence':nodes_middles, 'tear_segments_labels':tear_segments}
            nsegments = cdict['tear_segments_labels'].max()
            rel_ds = use_tear_annotation_ds // dsstep

            # interpolate the grid points first with MLS.
            # xxx - do this with downsampled image, how well would MLS interp work using blocks???
            cgrid = [None]*nsegments; deltas = [None]*nsegments
            dst_pts = cdict['correspondence'][:,2,:]*rel_ds
            tear_segments = cdict['tear_segments_labels']
            igrid = np.round(grid/rel_ds).astype(np.int64)
            assert( (igrid.max(0) < np.array(tear_segments.shape)[::-1] + 2).all() )
            # floor any points right along the edge of the image
            igrid[igrid[:,0] >= tear_segments.shape[1],0] = tear_segments.shape[1]-1
            igrid[igrid[:,1] >= tear_segments.shape[0],1] = tear_segments.shape[0]-1
            for i in range(nsegments):
                pts = cdict['correspondence'][:,i,:]*rel_ds
                # MLS "interpolate" grid points in current segment
                print('Grid interpolating segment {}'.format(i+1)); t = time.time()
                # select grid points in current segment
                sel_pts = np.logical_or(tear_segments == i+1, tear_segments == 0)
                cgrid[i] = grid[sel_pts[igrid[:,1],igrid[:,0]], :]
                # always include the correspondence points
                cgrid[i] = np.concatenate((cgrid[i], dst_pts), axis=0)
                deltas[i] = mls_rigid_transform(cgrid[i], dst_pts, pts) - cgrid[i]
                print('\tdone in %.4f s' % (time.time() - t, ))
            del igrid, sel_pts

            #tmpp = np.concatenate(cgrid, axis=0); tmpd = np.concatenate(deltas, axis=0)
            #plt.figure(4321); plt.gcf().clf()
            #plt.scatter(tmpp[:,0], tmpp[:,1], c=tmpd[:,0], s=36, marker='.')
            #plt.gca().invert_yaxis()
            #plt.gca().set_aspect('equal')
            #plt.colorbar()
            #plt.figure(4322); plt.gcf().clf()
            #plt.scatter(tmpp[:,0], tmpp[:,1], c=tmpd[:,1], s=36, marker='.')
            #plt.gca().invert_yaxis()
            #plt.gca().set_aspect('equal')
            #plt.colorbar()
            #plt.show()

            # upsample only the block being processed
            if not single_block:
                blk_ovlp_pix_ds = np.ceil(blk_ovlp_pix/rel_ds).astype(np.int64)
                _, _, _, rng = tile_nblks_to_ranges(tear_segments.shape, nblks, blk_ovlp_pix_ds, iblk)
                tear_segments = tear_segments[rng[0][0]:rng[0][1],rng[1][0]:rng[1][1]]
            print('Upsampling segments {}'.format(rel_ds)); t = time.time()
            tear_segments = block_construct(tear_segments, rel_ds)
            print('\tdone upsampling in %.4f s' % (time.time() - t, ))

            if single_block:
                blk_shape = img_shape
                blk_corner = np.array([0,0])
            else:
                _, _, _, rng = tile_nblks_to_ranges(img_shape, nblks, blk_ovlp_pix, iblk)
                blk_shape = [x[1] - x[0] for x in rng]
                blk_corner = np.array([rng[1][0], rng[0][0]])
            assert( all([np.abs(x-y) < 4*rel_ds for x,y in zip(tear_segments.shape, blk_shape)]) )
            # might have to crop the label image down to match shape
            tear_segments = tear_segments[:blk_shape[0],:blk_shape[1]]
            # because of pixels lost from downsampling, might have to pad
            if not all([x==y for x,y in zip(tear_segments.shape, blk_shape)]):
                pad = [y - x for x,y in zip(tear_segments.shape, blk_shape)]
                tear_segments = np.pad(tear_segments, ((0,pad[0]),(0,pad[1])), 'edge')
            assert( all([x==y for x,y in zip(tear_segments.shape, blk_shape)]) )

            print('Allocating grid points'); t = time.time()
            grid_pts = np.indices(blk_shape, dtype=xdtype)
            grid_pts = grid_pts[::-1,:,:] # swap x/y for coords
            grid_pts_flat = grid_pts.reshape(2,-1)
            print('\tdone in %.4f s' % (time.time() - t, ))

            # do the dense interpolation only on the block.
            # griddata has problems (probably pointer sizes issue) for very large grid sizes.
            vx = np.zeros(grid_pts_flat.shape[1], dtype=xdtype)
            vy = np.zeros(grid_pts_flat.shape[1], dtype=xdtype)
            for i in range(nsegments):
                print('Dense interpolating segment {}'.format(i+1))
                sel_pts_flat = np.logical_or(tear_segments == i+1, tear_segments == 0).reshape(-1)
                sel_grid_pts_flat = grid_pts_flat[:,sel_pts_flat].T
                if not single_block:
                    if sel_grid_pts_flat.shape[0] < 2:
                        print('\tSkipping, no overlap')
                        continue
                    rng = sel_grid_pts_flat.max(0) - sel_grid_pts_flat.min(0)
                    if (rng < blk_ovlp_pix).any():
                        print('\tSkipping, segment range {} {} < overlap pix {} {}'.format(rng[0], rng[1],
                            blk_ovlp_pix[0], blk_ovlp_pix[1]))
                        continue
                    cgrid[i] -= blk_corner
                    sel = np.logical_and(cgrid[i] >= 0, cgrid[i] < np.array(blk_shape)[None,::-1]).all(1)
                    cgrid[i] = cgrid[i][sel,:]; deltas[i] = deltas[i][sel,:]
                    if cgrid[i].shape[0] < 1:
                        print('\tSkipping, no block overlap')
                        continue
                print('Interpolating with griddata'); t = time.time()
                vx[sel_pts_flat] = interp.griddata(cgrid[i], deltas[i][:,0], sel_grid_pts_flat,
                    fill_value=0., method='cubic')
                vy[sel_pts_flat] = interp.griddata(cgrid[i], deltas[i][:,1], sel_grid_pts_flat,
                    fill_value=0., method='cubic')
                print('\tdone in %.4f s' % (time.time() - t, ))
                del sel_pts_flat, sel_grid_pts_flat
            del grid_pts_flat, cgrid, deltas
            # too large deltas here always seems to stem from the MLS divide by zero tolerance
            if (np.abs(vx) >= img_shape[1]).any() or (np.abs(vy) >= img_shape[0]).any():
                print('unexpectedly large deltas, adjust MLS tolerance')
                print(img_shape)
                print(vx.min(), vx.max(), vy.min(), vy.max())
                assert(False)

            # use the same method as in the wafer fine alignment transformation coordinate xform version.
            # get a bounding box on the transformed coords and use this to load from the original region image.
            vx = vx.reshape(blk_shape); vy = vy.reshape(blk_shape)
            if blk_corner[0] != 0:
                grid_pts[0,:,:] += (vx + blk_corner[0].astype(xdtype))
            if blk_corner[1] != 0:
                grid_pts[1,:,:] += (vy + blk_corner[1].astype(xdtype))
            del vx, vy
            grid_pts = grid_pts[::-1,:,:] # swap x/y for remap

            # only load from the bounding box on the source image coords
            coords_min = grid_pts.reshape(2,-1).min(1)
            coords_max = grid_pts.reshape(2,-1).max(1)
            coords_imin = np.floor(coords_min).astype(np.int64)
            coords_imin[coords_imin < 0] = 0
            coords_imax = np.ceil(coords_max).astype(np.int64)
            sel = (coords_imax >= img_shape); coords_imax[sel] = img_shape[sel] - 1
            custom_rng = [[coords_imin[x], coords_imax[x]] for x in range(2)]
            # shift coordinates to match the loaded source image bounding box
            grid_pts -= coords_imin[:,None,None]

            # if the coordinates are all out of bounds, then just query for the datatype.
            # xxx - just setting a single pixel so that the _region_load_load_imgs code path
            #   that determines whether the background / tissue mask should be loaded or not is presevered.
            if (coords_max <= 0).any() or (coords_min >= img_shape).any():
                out_of_bounds = True
                custom_rng = [[0,1], [0,1]]
            else:
                out_of_bounds = False

            print('Loading image, applying remap and saving'); t = time.time()
            img, _ = big_img_load(slice_image_fn, custom_rng=custom_rng)
            if not out_of_bounds:
                img = nd.map_coordinates(img, grid_pts, order=1, mode='constant', cval=0.0, prefilter=False)
            else:
                img = np.zeros(blk_shape, dtype=img.dtype)
            write_count, f1, f2 = big_img_save(stitched_slice_fn, img, img_shape, nblks=nblks, iblk=iblk,
                    novlp_pix=blk_ovlp_pix, compression=True, recreate=True, lock=lock, keep_locks=lock, wait=True)
            print('\tdone in %.4f s' % (time.time() - t, ))
            print('Loading background, applying remap and saving'); t = time.time()
            img, _ = big_img_load(slice_image_fn, custom_rng=custom_rng, dataset='background')
            if not out_of_bounds:
                img = nd.map_coordinates(img, grid_pts, order=1, mode='constant', cval=0.0, prefilter=False)
            else:
                img = np.zeros(blk_shape, dtype=img.dtype)
            big_img_save(stitched_slice_fn, img, img_shape, nblks=nblks, iblk=iblk, novlp_pix=blk_ovlp_pix,
                dataset='background', compression=True, recreate=True, f1=f1, f2=f2)
            print('\tdone in %.4f s' % (time.time() - t, ))
            print('Loading overlap, applying remap and saving'); t = time.time()
            img, _ = big_img_load(slice_image_fn, custom_rng=custom_rng, dataset='overlap_count')
            if not out_of_bounds:
                img = nd.map_coordinates(img, grid_pts, order=1, mode='constant', cval=0.0, prefilter=False)
            else:
                img = np.zeros(blk_shape, dtype=img.dtype)
            del grid_pts
            big_img_save(stitched_slice_fn, img, img_shape, nblks=nblks, iblk=iblk, novlp_pix=blk_ovlp_pix,
                dataset='overlap_count', compression=True, recreate=True, f1=f1, f2=f2)
            print('\tdone in %.4f s' % (time.time() - t, ))
            if first_block:
                histo_shape, histo_dtype = big_img_info(slice_image_fn, dataset='histogram')
                # create the histogram dataset now, as it will be computed more than once after image transformations.
                histo = np.zeros(histo_shape, dtype=histo_dtype)
                big_img_save(stitched_slice_fn, histo, histo.shape, dataset='histogram', recreate=True)
                # xxx - hacky, save slots for downsampled image histograms
                for ds in [2,4,8,16,32,64]:
                    big_img_save(stitched_slice_fn, histo, histo.shape, dataset='histogram_'+str(ds), recreate=True)
            if lock: gpfs_file_unlock(f1,f2)
        # run-type if/elif
    #for region_ind in cregion_inds:

    if run_slice_histos:
        print('\tdone in %.4f s' % (time.time() - t, ))
        print('run_slice_histos: starting jobs for wafer %d' % (wafer_id,))
        cregion_inds = (np.array(region_inds)-1) if region_inds[0] > -1 else np.arange(nregions)
        cnregions = cregion_inds.size
        nworkers = min([arg_nworkers, cnregions])
        workers = [None]*nworkers
        result_queue = mp.Queue(cnregions)
        inds = np.array_split(cregion_inds, nworkers)
        assert( not show_plots or nworkers == 1 )
        for i in range(nworkers):
            workers[i] = mp.Process(target=compute_histos_job, daemon=True,
                    args=(i, inds[i], slice_histos_include_fns[inds[i][0]:inds[i][-1]+1],
                        slice_histos_roi_polys[inds[i][0]:inds[i][-1]+1],
                        tissue_masks, cregion.tissue_mask_min_size, cregion.tissue_mask_min_hole_size, dsstep, nblks,
                        None if no_iblock else iblk, result_queue, histos_compute_areas, show_plots, True))
            workers[i].start()
        # NOTE: only call join after queue is emptied

        dt = time.time()
        print_every = int(1e6) if native else 25
        dsstr = ('_'+str(dsstep)) if dsstep > 1 else ''
        worker_cnts = np.zeros((nworkers,), dtype=np.int64); doinit = True
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for i in range(cnregions):
        i = 0
        while i < cnregions:
            if i>0 and i%print_every==0:
                print('%d through q in %.3f s, worker_cnts:' % (print_every, time.time()-dt,)); dt = time.time()
                print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=zimages.queue_timeout)
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

            # the size of the histo is already set in the hdf5, so the -1 gets broadcast
            histo = res['histo'] if res['histo'] is not None else -np.ones((1,), dtype=np.int64)
            if res['write_count_unique'] < res['write_count_expected']:
                print('write_count_unique {} < write_count_expected {}: {}'.format(
                    res['write_count_unique'], res['write_count_expected'], slice_histos_fns[res['ind']]))
                assert(False)
            big_img_save(slice_histos_fns[res['ind']], histo, histo.shape, dataset='histogram'+dsstr,
                attrs={'area':res['area']})
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]
    elif brightness_balance_slices_whole: # if run_slice_histos:
        print('loading precomputed histograms from image hdf5s'); t = time.time()
        dsstr = ('_'+str(dsstep)) if dsstep > 1 else ''
        for i in range(nregions):
            histo, _ = big_img_load(slice_histos_fns[i], dataset='histogram'+dsstr)
            if i==0:
                histos = np.empty((nregions, histo.size), dtype=np.int64); histos.fill(-1)
            histos[i,:] = histo
        if slices_histos_doinit:
            slices_histos = histos; slices_histos_doinit = False
        else:
            slices_histos = np.vstack((slices_histos, histos))
        print('\tdone in %.4f s' % (time.time() - t, ))
    #if run_slice_histos:
#for wafer_id in wafer_ids:

# apply brightness balancing across all the specified wafers
if brightness_balance_slices_whole:
    n = brightness_balance_slices_whole_tiles_nslices
    if n > 1:
        print('Whole slice brightness adjust with %d neighboring slice comparisons' % (n,))

        # create the adjacency matrix directly
        adj_matrix = sp.diags([np.ones(cum_nregions,dtype=np.double)]*2*n, [x for x in range(-n,n+1) if x != 0],
                shape=(cum_nregions,cum_nregions), format='lil').astype(bool)

        # reordered the banded adjacency matrix based on solved order into region order.
        adj_matrix = adj_matrix[cum_to_solved_order,:][:,cum_to_solved_order]
        adj_matrix[:,cum_missing_region_inds] = 0; adj_matrix[cum_missing_region_inds,:] = 0
        # xxx - long comment here because it was a bug and it was essentially a dependency-betwen-steps trap
        # the histograms to be ignored in img_brightness_balancing are determinted by slices_histos
        #   being all negative for a particular image. for this to work properly the exclude regions
        #   MUST have been excluded (--slice-balance-exclude) when the histos were calculated.
        # added this, which forces exclude regions to be ignored here, even if the histos were calculated
        #   for them, because otherwise this is an annoying dependency and did not see a use case for
        #   keeping exluded regions in the balancing when the slice ordering is being used.
        slices_histos[cum_missing_region_inds,:] = -1
        brightness_adjusts = region.img_brightness_balancing(slices_histos, adj_matrix=adj_matrix, regr_bias=True,
                                 maxlag=brightness_balancing_slices_maxlag, nworkers=arg_nworkers, verbose=True)
    else:
        print('Whole slice brightness adjust with with all pairwise comparisons')
        slices_histos[cum_missing_region_inds,:] = -1
        assert( (slices_histos.sum(1) != 0).all() ) # some histos were not computed properly
        brightness_adjusts = region.img_brightness_balancing(slices_histos, ntop=brightness_balancing_slices_ntop,
                maxlag=brightness_balancing_slices_maxlag, nspan=brightness_balancing_slices_nspan,
                nworkers=arg_nworkers, label_adj_min=brightness_balancing_slices_label_adj_min, verbose=True)
    print(brightness_adjusts.min(), brightness_adjusts.max())

    # write out the brightness adjustments over all slices individiually in wafer alignment folders.
    cum_nregions = 0
    for wafer_id in wafer_ids:
        experiment_folders, thumbnail_folders, _, alignment_folder, _, region_strs = get_paths(wafer_id)
        # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
        region_strs = [item for sublist in region_strs for item in sublist]
        nregions = len(region_strs)
        coords_file_names = [None]*nregions
        for i in range(nregions):
            coords_file_names[i] = os.path.join('wafer%d_%s' % (wafer_id, region_strs[i]))

        zimages.write_image_coords(slice_balance_fns[wafer_id], coords_file_names,
            brightness_adjusts[cum_nregions:cum_nregions+nregions,None])
        cum_nregions += nregions


# do not delete this, is used at a minimum for addressing "timed-out" when running on the cluster (to be re-run).
# can also grep to count how many jobs of a particular run have completed without fatal error.
print('JOB FINISHED: run_regions.py')
print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
