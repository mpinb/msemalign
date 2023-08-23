#!/usr/bin/env python3
"""run_wafer.py

Top level command-line interface for the running the matching for the fine
  alignment and for exporting sections for a whole wafer at different
  stages of the alignment.

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

## <<< so figures can be saved without X11, uncomment
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
## so figures can be saved without X11, uncomment >>>

import numpy as np

import dill
import argparse
import os

from msem import wafer

from msem.utils import make_hex_points, PolyCentroid
from msem.utils import dill_lock_and_load, dill_lock_and_dump
from msem.utils import big_img_load, big_img_info
from msem.zimages import msem_input_data_types

# all parameters loaded from an experiment-specific import
from def_common_params import get_paths, native_subfolder, dsstep, use_thumbnails_ds, dsthumbnail, exclude_regions
from def_common_params import czifiles, czfiles, czipath, czifile_ribbons, czifile_scenes, czifile_rotations
from def_common_params import scale_nm, nimages_per_mfov, legacy_zen_format, region_rotations_all
from def_common_params import czifile_use_roi_polygons, roi_polygon_translations, crops_um
from def_common_params import fine_dill_fn_str, delta_dill_fn_str, rough_dill_fn_str, limi_dill_fn_str
from def_common_params import order_txt_fn_str, wafer_format_str, meta_dill_fn_str
from def_common_params import wafer_template_slice_inds, translate_roi_center
from def_common_params import region_manifest_cnts, region_include_cnts, total_nwafers
from def_common_params import delta_rotation_range, delta_rotation_step, template_crop_um
from def_common_params import roi_polygon_scale, thumbnail_suffix
from def_common_params import region_suffix, region_interp_type_deltas
from def_common_params import rough_bounding_box_xy_spc, rough_grid_xy_spc, fine_grid_xy_spc
from def_common_params import wafer_solver_bbox_xy_spc, wafer_solver_bbox_trans
from def_common_params import thumbnail_subfolders, thumbnail_subfolders_order, debug_plots_subfolder
from def_common_params import tissue_mask_path, tissue_mask_fn_str, tissue_mask_ds
from def_common_params import tissue_mask_min_edge_um, tissue_mask_min_hole_edge_um, tissue_mask_bwdist_um
from def_common_params import tears_subfolder, torn_regions, slice_blur_z_indices, slice_blur_factor
from def_common_params import noblend_subfolder

# <<< turn on stack trace for warnings
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
#warnings.simplefilter("always")
# turn on stack trace for warnings >>>

## uncomment to enable all the logging from region / mfov. typically too verbose for wafer.
#import logging
#LOGGER = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
#                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


## argparse

parser = argparse.ArgumentParser(description='run_wafer.py')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
                    help='wafer to run OR two neighboring wafers to run cross-wafer alignment')
parser.add_argument('--solved-order-ind', nargs='+', type=int, default=[0],
                    help='compare region order ind to ind+1, use range to init fine dills')
parser.add_argument('--invert-order', dest='invert_order', action='store_true',
                    help='go in the backwards direction of the region order')
parser.add_argument('--skip-slices', dest='skip_slices', nargs=1, type=int, default=[0],
                    help='how many slices beyond the next slice to template match to')
parser.add_argument('--crops-um-ind', nargs=1, type=int, default=[0],
                    help='index into crops_um (non-CL parameter) to use')
parser.add_argument('--fine-blur-only', dest='fine_blur_only', action='store_true',
                    help='only rerun comparisons at slice transitions to be blurred')
parser.add_argument('--export-region-beg', nargs=1, type=int, default=[1],
                    help='for any of the export runs, region to start with (base 1)')
parser.add_argument('--export-region-end', nargs=1, type=int, default=[0],
                    help='for any of the export runs, region to stop on (base 1)')
parser.add_argument('--validate-region-grid', dest='validate_region_grid', action='store_true',
                    help='debug feature to show grids overlaid on image')
parser.add_argument('--run-type', nargs=1, type=str, default=['wafer_stats'],
                    choices=['fine', 'fine_export', 'rough_export', 'wafer_stats',
                             'export_rough_dills', 'update_meta_dill', ],
                    help='the type of run to choose')
parser.add_argument('--overlays', dest='overlays', action='store_true',
                    help='do overlays (rough export only)')
parser.add_argument('--rough-hdf5', dest='rough_hdf5', action='store_true',
                    help='store/load reloadable rough alignment (typically for native)')
parser.add_argument('--no-order', dest='load_solved_order', action='store_false',
                    help='specify not to load solved order')
parser.add_argument('--no-rough-alignment', dest='load_rough_alignment', action='store_false',
                    help='specify to not use the rough alignment')
parser.add_argument('--no-blending-features', dest='blending_features', action='store_false',
                    help='disable all tile balancing and blending features')
parser.add_argument('--tissue-masks', dest='tissue_masks', action='store_true',
                    help='use the tissue masks')
parser.add_argument('--save-masks-in', nargs=1, type=str, default=[''],
                    help='input path for the tissue masks')
parser.add_argument('--use-coordinate-based-xforms', dest='use_coordinate_based_xforms', action='store_true',
                    help='use coordinate xforms instead of image xforms when loading regions')
parser.add_argument('--crop-to-grid', dest='crop_to_grid', action='store_true',
                    help='option for exports that crops output to grid min/max')
parser.add_argument('--zero-outside', dest='zero_outside', action='store_true',
                    help='option for exports that zeros outside of polygon (mask if --tissue-masks)')
parser.add_argument('--contrast-filter', dest='contrast_filter', action='store_true',
                    help='use the slices that were exported with the contrast filter')
parser.add_argument('--rough-export-solve-order', dest='rough_export_solve_order', action='store_true',
                    help='option for rough_export that can specify manual rough bbox for order solving')
parser.add_argument('--rough-run-str', nargs=1, type=str, default=['none'],
                    help='string to differentiate rough alignments with different parameters')
parser.add_argument('--fine-run-str', nargs=1, type=str, default=['none'],
                    help='string to differentiate fine alignments with different parameters')
parser.add_argument('--delta-run-str', nargs=1, type=str, default=['none'],
                    help='string to differentiate fine alignment deltas with different parameters')
parser.add_argument('--thumbs-run-str', nargs=1, type=str, default=[''],
                    help='string to differentiate different thumbnails for order solving')
parser.add_argument('--custom-suffix', nargs=1, type=str, default=[''],
                    help='override the default filename suffix, mostly for testing')
parser.add_argument('--dsexports', nargs='*', type=int, default=[],
                    help='use these downsampling levels for export, leave empty for defaults')
parser.add_argument('--show-xcorr-plots', dest='show_xcorr_plots', action='store_true',
                    help='diplay debug plots of the cross correlations')
parser.add_argument('--save-xcorr-plots', dest='save_xcorr_plots', action='store_true',
                    help='save debug plots of the cross correlations')
parser.add_argument('--native', dest='native', action='store_true',
                    help='process native resolution images, not thumbnails')
parser.add_argument('--override-stack', nargs=1, type=str, default=[''],
                    help='use this location for input stacks instead of alignment folder')
parser.add_argument('--custom-crop-um', nargs=4, type=float, default=[-1,-1,-1,-1],
                    help='specify top left and size for crop (overrides normal grid)')
parser.add_argument('--range-write_order', dest='write_order', action='store_true',
                    help='hacky hook to export order (if slices provided in order)')
parser.add_argument('--nworkers', nargs=1, type=int, default=[1],
                    help='to control some operations threads/subprocesses')
# options for blockwise processing, for when slices are memory-limited
parser.add_argument('--nblocks', nargs=2, type=int, default=[1, 1],
                    help='number of partitions per dimension for blockwise processing')
parser.add_argument('--iblock', nargs=2, type=int, default=[0, 0],
                    help='which block to process for blockwise processing (zero-based)')
parser.add_argument('--block-overlap-um', nargs=2, type=float, default=[0., 0.],
                    help='amount of overlap between blocks in um for blockwise processing')
parser.add_argument('--block-overlap-grid-um', nargs=2, type=float, default=None,
                    help='specify different overlap for the grid points (for interpolation)')
parser.add_argument('--convert-h5-to-tiff', dest='convert_hdf5s', action='store_true',
                    help='take previously exported blockwise hdf5s and convert to tiffs')
parser.add_argument('--export-h5', dest='export_h5', action='store_true',
                    help='forgo the other logic and force export in hdf5 format')
args = parser.parse_args()
args = vars(args)


## params that are set by command line arguments

# this script is setup to only run neighboring alignments in a solved order.
# this specifies which solved-order index of the alignment to run.
iorder = args['solved_order_ind']

# if set, this means to align the previous slice in the order to the current.
# default is to the align the next slice in the solved order to the current.
invert_order = args['invert_order']

# use this to specify to template match against the i+n slice instead of i+1 in the region order
skip_slices = args['skip_slices'][0]

# for any of the export runs, optionally specify a starting and stopping indices
export_region_beg = args['export_region_beg'][0]-1
export_region_end = args['export_region_end'][0]-1

# for any of the export runs, use the specified downsampling levels, instead of the defaults
# do NOT use this when exporting for the solver.
dsexports = args['dsexports']

# wafer starting at 1, used only for specifying directories and filenames, does not map to any zeiss info.
#   in zeiss terminology, a single wafer is an experiment, however multiple wafers make up an actual tissue block.
# NOTE: only used as an array so that alignments can be calculated across wafer borders.
#   only specify one wafer, or two neighboring wafers.
#wafer_ids = [1] # one wafer
#wafer_ids = [4, 5] # two neighboring wafers
wafer_ids = args['wafer_ids']

# index into the crops_um array, amount to crop out in microns for slice to slice template matching.
crops_um_ind = args['crops_um_ind'][0]

# only rerun comparisons that cross transitions defined in slice_blur_z_indices
fine_blur_only = args['fine_blur_only']

# show the grid overlaid on the images after loading (sequentially).
# to validate region grids overlaid on the transformed slice (region_load).
validate_region_grid = args['validate_region_grid']

# this is an identifier so that multiple rough alignemnts can be exported / loaded easily.
rough_run_str = args['rough_run_str'][0]

# this is an identifier so that multiple fine alignemnts can be exported / loaded easily.
fine_run_str = args['fine_run_str'][0]

# this is an identifier so that multiple fine alignemnt deltas can be exported / loaded easily.
delta_run_str = args['delta_run_str'][0]

# this is an identifier so that multiple thumbnails can be exported / loaded easily.
# this can be useful during the order solving for regions generated with different
#   contrast / brightness balancing, for example.
thumbs_run_str = args['thumbs_run_str'][0]

# this is whether to export in the solved order or not for rough_export
load_solved_order = args['load_solved_order']

# this is whether to export with the rough alignment or not for rough_export
load_rough_alignment = args['load_rough_alignment']

# this is for rough exports, to optionally use manually defined rough bounding box
rough_export_solve_order = args['rough_export_solve_order']

# set False to disable all brightness balancing and blending features.
# in run_wafer this is only used to set the region suffix to load regions
#   that were stitched without using the brightness matching / blending features.
blending_features = args['blending_features']

# xxx - basically an old option that is really no longer useful now that tissue
#   masks have been introduced. crops to the min/max of the grid during rough or fine exports.
crop_to_grid = args['crop_to_grid']

# option to zero areas outside of the scaled polygon during rough or fine exports.
zero_outside = args['zero_outside']

# set to non-empty to override the standard region image file suffix
custom_suffix = args['custom_suffix'][0]

# enables the per-slice contrast and/or local contrast enhancement.
# in run_wafer this is only used to set the region suffix to load regions
#   that were exported using the contrast enhancment filter.
contrast_filter = args['contrast_filter']

# option to process / export native resolution regions
native = args['native']

# override the alignment folder as the input stack location and use this instead.
# this is used for exporting UF alignment at native that was computed at a downsampled level.
override_stack = args['override_stack'][0]

# override the normal grid by specifying top left and size of a cropping box.
# useful for only expoorting a specific area for the rough and fine exports.
custom_crop = args['custom_crop_um']

# show debug plots of the cross correlations
show_xcorr_plots = args['show_xcorr_plots']

# path to optionally save debug plots of the cross correlations
save_xcorr_plots = args['save_xcorr_plots']

# this is a hacky special hook to write the order out.
write_order = args['write_order']

# specify to export rough alignment at a higher downsampling and
#   overlaid with ROI polygon, image center, rough bounding box, etc.
overlays = args['overlays']

# specify a special rough_export mode that writes rough aligment in hdf5
#   or for fine export, loads from this hdf5 (thereby skipping the rough xforms).
# recommend only using this for the native resolution fine export.
rough_hdf5 = args['rough_hdf5']

# read in tissues masks for each slice, more refined/precise that roi polygon
tissue_masks = args['tissue_masks']

# arguments for saving downsampled masks into the region hdf5 files.
# this is for processing masks that have been saved in some alignment step after the regions.
save_masks_in = args['save_masks_in'][0]

# this enables the new mode that allows for full block processing of the rough alignment.
# all the transformations are then based on the coordinates. the image transformation is
# then applied as a single remap based on the coordinate xforms.
use_coordinate_based_xforms = args['use_coordinate_based_xforms']

# control some operations threads/subprocesses separate from MSEM_NUM_THREADS
arg_nworkers = args['nworkers'][0]

# options for blockwise processing, mostly intended for native
nblks = args['nblocks'][::-1]
iblk = args['iblock'][::-1]
blk_ovlp_um = args['block_overlap_um'][::-1]
convert_hdf5s = args['convert_hdf5s']
export_h5 = args['export_h5']
# optionally specify a different overlap amount for the grid points.
# this allows for keeping more grid points in for context when doing dense pixel interpolation,
#   but without actually interpolating all of the overlap area.
blk_ovlp_grid_um = args['block_overlap_grid_um']
blk_ovlp_grid_um = blk_ovlp_um if blk_ovlp_grid_um is None else blk_ovlp_grid_um[::-1]

# these specify what type of run this is (one of these must be set True)

run_type = args['run_type'][0]

# specify to only print statistics for the requested wafer
get_wafer_stats = run_type == 'wafer_stats'

# specify to only run the deltas / warping fine alignment
fine_run = run_type == 'fine'

# export using saved fine angle and delta accumulated alignments
fine_export_run = run_type == 'fine_export'

# specify to update the meta dill (with pixel resolution...)
update_meta_dill = run_type == 'update_meta_dill'

# specify to export the dills required for rough alignemnt.
# this mode loads the coordinate files for all the regions in a wafer,
#   which in the worst case of many instances running in parallel on GFPS can be quite slow.
# run in this mode when exporting the grid dill also.
rough_dills_run = (run_type == 'export_rough_dills' or run_type == 'update_meta_dill')

# specify to export rough alignment.
rough_export_run = run_type == 'rough_export'

print('run_wafer run-type is ' + run_type + (', solved_order_ind %d,' % (iorder[0],)) + \
      (' skip slices is %d,' % (skip_slices,)) + ' load solved order is ' + str(load_solved_order) + \
      ', load rough alignment is ' + str(load_rough_alignment) + ', native is ' + str(native) + \
      ', rough hdf5 (save/load) is ' + str(rough_hdf5))
print('running with wafer_ids:')
print(wafer_ids)

init_locks = any([x < 0 for x in nblks])
if init_locks: nblks = [abs(x) for x in nblks]

if not all([x==1 for x in nblks]):
    print('iblk {} {} of {} {}'.format(iblk[0],iblk[1],nblks[0],nblks[1]))
    print('ovlp {} {} um, grid ovlp {} {} um'.format(blk_ovlp_um[0],blk_ovlp_um[1],
        blk_ovlp_grid_um[0],blk_ovlp_grid_um[1]))


## fixed parameters not exposed in def_common_params

# this is the subfolder used for the special h5 rough alignment export.
load_rough_xformed_img_subfolder = 'rough'
# this is the suffix used for the special h5 rough alignment export.
rough_aligned_suffix = '_rough_aligned'


## parameters that are determined based on above parameters

# special mode to initialize empty fine alignment dills
if len(iorder) > 1:
    fine_init_dills = True
    fine_init_dills_rng = iorder
else:
    fine_init_dills = False
iorder = iorder[0]

assert( not (convert_hdf5s and export_h5) )

nwafer_ids = len(wafer_ids)
assert(nwafer_ids < 3)
assert(iorder < 0 or nwafer_ids == 1) # use negative iorder to specify cross-wafer comparisons
assert(not fine_init_dills or nwafer_ids==1) # use negative --solved-order-ind[0] with one wafer for cross init

assert( not (fine_run or fine_export_run) or load_solved_order ) # fine alignment needs order

# set the rough bounding box and grid locations depending on the run.
fine_grid_locations = make_hex_points(*fine_grid_xy_spc)
rough_grid_locations = make_hex_points(*rough_grid_xy_spc)
normal_rough_bounding_box = make_hex_points(*rough_bounding_box_xy_spc, bbox=True)
if fine_run or fine_export_run:
    rough_bounding_box = normal_rough_bounding_box
    grid_locations = fine_grid_locations
    griddist_um = fine_grid_xy_spc[2]
else:
    if rough_export_run and rough_export_solve_order and wafer_solver_bbox_xy_spc[wafer_ids[0]] is not None:
        bbox = wafer_solver_bbox_xy_spc[wafer_ids[0]]
        trans = wafer_solver_bbox_trans[wafer_ids[0]]
        rough_bounding_box = make_hex_points(*bbox, trans=trans, bbox=True)
        grid_locations = make_hex_points(*bbox, trans=trans)
        griddist_um = bbox[2]
    else:
        rough_bounding_box = normal_rough_bounding_box
        grid_locations = rough_grid_locations
        griddist_um = rough_grid_xy_spc[2]

use_custom_crop = all([x > -1 for x in custom_crop])
if use_custom_crop:
    x = np.array([[custom_crop[0],custom_crop[1]],
        [custom_crop[0]+custom_crop[2],custom_crop[1]],
        [custom_crop[0],custom_crop[1]+custom_crop[3]],
        [custom_crop[0]+custom_crop[2],custom_crop[1]+custom_crop[3]]])
    # custom cropping box is typically coming from knossos coordinates,
    #    so always assume coordinates are relative to fine grid.
    if fine_export_run and convert_hdf5s:
        cnr = rough_bounding_box[0]
    else:
        cnr = fine_grid_locations.min(0)
    grid_locations = x + cnr
    print('Custom grid relative to corner ({:.4f},{:.4f}) at {:.4f},{:.4f} size {:.4f}x{:.4f} (all um)'\
        .format(cnr[0],cnr[1], grid_locations[0,0], grid_locations[0,1], custom_crop[2], custom_crop[3]))

# amount to crop out in microns for slice to slice template matching.
crop_um = crops_um[crops_um_ind]

# # support for reading the tissue masks (to filter the keypoints)
# if tissue_masks:
#     assert(tissue_mask_path is not None)
# else:
#     tissue_mask_path = None

experiment_folders_all = [None]*nwafer_ids
thumbnail_folders = [None]*nwafer_ids
alignment_folders = [None]*nwafer_ids; use_alignment_folders = [None]*nwafer_ids
region_strs_all = [None]*nwafer_ids
protocol_folders_all = [None]*nwafer_ids
for i,j in zip(wafer_ids, range(nwafer_ids)):
    experiment_folders_all[j], thumbnail_folders[j], protocol_folders_all[j], alignment_folders[j], meta_folder, \
        region_strs_all[j] = get_paths(i)
    use_alignment_folders[j] = os.path.join(alignment_folders[j], native_subfolder) if native else alignment_folders[j]

rough_dill_fns = [None]*nwafer_ids; fine_dill_fns = [None]*nwafer_ids
limi_dill_fns = [None]*nwafer_ids; order_txt_fns = [None]*nwafer_ids
for i in range(nwafer_ids):
    rough_dill_fns[i] = os.path.join(alignment_folders[i], rough_dill_fn_str.format(wafer_ids[i], rough_run_str))
    fine_dill_fns[i] = os.path.join(alignment_folders[i], fine_dill_fn_str.format(wafer_ids[i], fine_run_str))
    limi_dill_fns[i] = os.path.join(alignment_folders[i], limi_dill_fn_str.format(wafer_ids[i]))
    order_txt_fns[i] = os.path.join(alignment_folders[i], order_txt_fn_str.format(wafer_ids[i]))

if rough_dills_run:
    meta_dill_fn = os.path.join(meta_folder, meta_dill_fn_str)
    if czifiles is None or czifiles[wafer_ids[0]] is None:
        czifile = None
    else:
        czifile = os.path.join(czipath, czifiles[wafer_ids[0]])
    czifile_ribbon = czifile_ribbons[wafer_ids[0]]
    czifile_scene = czifile_scenes[wafer_ids[0]]
    if czfiles is None or czfiles[wafer_ids[0]] is None:
        czfile = None
    else:
        czfile = os.path.join(czipath, czfiles[wafer_ids[0]])
    czifile_rotation = czifile_rotations[wafer_ids[0]]
    template_slice_ind = wafer_template_slice_inds[wafer_ids[0]]
    region_rotation = region_rotations_all[wafer_ids[0]]

    # no need to instantiate all the slices for new acquisition format
    init_regions = (legacy_zen_format and not update_meta_dill)
else:
    init_regions = True

# where to save optional cross correlation debug plots
plots_folder = os.path.join(meta_folder, debug_plots_subfolder)

# this is a "pre-translation" to try and counter sharp transitions between wafers.
# this happens on the experimental side because they reset the roi template between wafers.
roi_polygon_translations = np.concatenate([np.array(roi_polygon_translations[x]).reshape(-1,2) \
    for x in wafer_ids], axis=0)

# whether there are fixed tear saved regions for this dataset / wafer
wafer_tears = (torn_regions is not None and len(torn_regions[wafer_ids[0]]) > 0)

# xxx - this block is a particular !@^# show
load_stitched_coords = True
solved_order = None
region_inds = None
backload_roi_polys = None
get_export_solved_order = False
zorder = None
istorn = False
load_img_subfolder = override_stack
is_excluded = False
if rough_dills_run or get_wafer_stats:
    assert(nwafer_ids == 1) # no multiple wafer rough dills or stats runs

    if get_wafer_stats:
        alignment_folders = None
        load_stitched_coords = False
else:
    rough_dicts = [None]*2 # duplicated below for single wafer case, was easier
    for i in range(nwafer_ids):
        with open(limi_dill_fns[i], 'rb') as f: limi_dict = dill.load(f)
        if load_rough_alignment:
            with open(rough_dill_fns[i], 'rb') as f: rough_dicts[i] = dill.load(f)
            rough_dicts[i]['imaged_order_limi_rotation'] = limi_dict['imaged_order_limi_rotation']
            rough_dicts[i]['template_roi_points'] = limi_dict['template_roi_points']
            rough_dicts[i]['imaged_order_region_recon_roi_poly_raw'] = \
                limi_dict['imaged_order_region_recon_roi_poly_raw']
        else:
            rough_dicts[i] = limi_dict
            if load_solved_order:
                rough_dicts[i]['solved_order'] = \
                    np.fromfile(order_txt_fns[i], dtype=np.uint32, sep=' ')-1 # saved order is 1-based
    if nwafer_ids==1:
        rough_dicts[1] = rough_dicts[0]
    elif invert_order:
        rough_dicts = rough_dicts[::-1]
        # have to invert the arrays pertaining to the regions also in the opposite wafer order
        experiment_folders_all = experiment_folders_all[::-1]
        thumbnail_folders = thumbnail_folders[::-1]
        protocol_folders_all = protocol_folders_all[::-1]
        use_alignment_folders = use_alignment_folders[::-1]
        region_strs_all = region_strs_all[::-1]
        roi_polygon_translations = roi_polygon_translations[::-1,:]
        wafer_ids = wafer_ids[::-1]

    do_export_solved_order = load_solved_order
    export_solved_order = None
    order_name_str = 'order' if load_solved_order else 'manifest'
    cum_include_cnts = np.cumsum(region_include_cnts[1:])
    if rough_export_run or fine_export_run:
        assert(nwafer_ids == 1) # no multiple wafer export runs
        if export_region_end > 0:
            assert(not wafer_tears or ((export_region_end - export_region_beg) == 1)) # not implemented
            # setting region_inds to None is very costly because then wafer has to load all the image
            #   coordinates for all the regions in the wafer, and since these are small text files
            #   this can be very slow on a distributed file system like GPFS.
            region_inds = np.array(range(export_region_beg,export_region_end))
            do_export_solved_order = False
            export_solved_order = region_inds
            if load_solved_order:
                # zorder was put in later for order in full dataset
                w = 0 if wafer_ids[0] < 2 else cum_include_cnts[wafer_ids[0]-2]
                zorder = region_inds + w
                if region_inds < rough_dicts[0]['solved_order'].size:
                    # use the export inds as solved order inds, get corresponding region_inds
                    region_inds = rough_dicts[0]['solved_order'][region_inds]
                else:
                    # this confusing codepath allows excluded regions to be exported or to have masks saved.
                    is_excluded = True
                    i = region_inds - rough_dicts[0]['solved_order'].size
                    region_inds = np.array([exclude_regions[wafer_ids[0]][x] for x in i]) - 1
                    # xxx - gah, for ease of viewing moved all exclude slices to the end of the stack.
                    #   this makes the z-order "backcalculation" not so easy.
                    cum_exclude_cnts = np.cumsum(np.array([len(exclude_regions[x+1]) for x in range(total_nwafers)]))
                    w = 0 if wafer_ids[0] < 2 else cum_exclude_cnts[wafer_ids[0]-2]
                    zorder = cum_include_cnts[-1] - cum_exclude_cnts[-1] + w + i
                istorn = wafer_tears and ((region_inds[0] + 1) in torn_regions[wafer_ids[0]])
            else:
                istorn = wafer_tears and ((export_region_beg + 1) in torn_regions[wafer_ids[0]])
            export_region_end = export_region_end-export_region_beg; export_region_beg = 0
        else:
            assert(not wafer_tears) # not implemented for multiple region export
            region_inds = None
            if load_solved_order:
                solved_order = rough_dicts[0]['solved_order']
            else:
                # deals with a legacy workflow (no region range specified), allow manifest index to still be added
                get_export_solved_order = True
                do_export_solved_order = False

        # this is in case the region_stage_coords.csv were not saved during acquisition. (!)
        backload_roi_polys = rough_dicts[0]['imaged_order_region_recon_roi_poly_raw']
        if region_inds is not None:
            backload_roi_polys = [backload_roi_polys[x] for x in region_inds]
    else:
        # integers for which regions to try to align in z, None for all in the experiement folder
        #region_inds = None  # for all in the experiment folder, order is arbitrary
        if invert_order:
            region_inds = [rough_dicts[0]['solved_order'][iorder+1+skip_slices],
                          rough_dicts[1]['solved_order'][iorder]]
        else:
            region_inds = [rough_dicts[0]['solved_order'][iorder],
                          rough_dicts[1]['solved_order'][iorder+1+skip_slices]]

        # the order is always 0 to 1 since the regions were set above in order, based on the solved order.
        solved_order = [0,1]

        # this is in case the region_stage_coords.csv were not saved during acquisition. (!)
        backload_roi_polys = [rough_dicts[0]['imaged_order_region_recon_roi_poly_raw'][region_inds[0]],
                              rough_dicts[1]['imaged_order_region_recon_roi_poly_raw'][region_inds[1]]]

        # need to be able to load either as tear repairs for the fine alignment
        w = [wafer_ids[0], wafer_ids[0]]
        if nwafer_ids > 1: w[1] = wafer_ids[1]
        istorn = [wafer_tears and ((region_inds[0] + 1) in torn_regions[w[0]]),
                  wafer_tears and ((region_inds[1] + 1) in torn_regions[w[1]])]
        load_img_subfolder = [tears_subfolder if istorn[0] else '',
                              tears_subfolder if istorn[1] else '']
        istorn = any(istorn) # leaving as array causes problems with the logic below

        # zorder was put in later for order in full dataset
        if nwafer_ids > 1:
            w = cum_include_cnts[max(wafer_ids)-2] # bigger wafer has to always be > 1
            # for cross-wafer iorder relies on python negative indexing above
            zorder0 = iorder + cum_include_cnts[min(wafer_ids)-1]
        else:
            w = 0 if wafer_ids[0] < 2 else cum_include_cnts[wafer_ids[0]-2]
            zorder0 = iorder + w
        zorder1 = iorder+1+skip_slices + w
        zorder = [zorder1, zorder0] if invert_order else [zorder0, zorder1]

# in the new region indexing mode, the region_inds are 1-based
if region_inds is not None: region_inds = [x+1 for x in region_inds]

# only need exclude regions if all region_inds are specified (set to None)
# can not use exclude_regions along with exporting rough dills
exclude_regions = exclude_regions[wafer_ids[0]] if (exclude_regions is not None and region_inds is None \
        and (not rough_dills_run or write_order)) else None

# load images without blending features if specified
use_region_suffix = (custom_suffix if custom_suffix else region_suffix) #\
#    if blending_features else region_suffix_noblend

# to support native resolution export
if native:
    to_native_scale = use_thumbnails_ds
    use_use_thumbnails_ds = 0
else:
    to_native_scale = 1
    use_use_thumbnails_ds = use_thumbnails_ds

# changed this to the regular alignment folders so that stitched coords are always
#   loaded from that location (even for native).
# xxx - we always use the thumbnail alignment coords for native, will there ever be a reason not to?
load_stitched_coords = use_alignment_folders if load_stitched_coords else None

load_rough_xformed_img = False
# xxx - hacky, but easily allows for block processing of stacks
#   this got ugly because load_img_subfolder supports loading the rough hdf5, loading tears
#     and loading no blending features because the modes are mutually exclusive.
valid_experiment_folders = any([x for x in experiment_folders_all])
if len(load_img_subfolder) == 0 or all([not x for x in load_img_subfolder]):
    if rough_hdf5 and (fine_run or fine_export_run):
        load_rough_xformed_img = True
        if valid_experiment_folders:
            load_img_subfolder = load_rough_xformed_img_subfolder
            use_region_suffix = rough_aligned_suffix
    elif istorn:
        load_img_subfolder = tears_subfolder
    elif not blending_features:
        load_img_subfolder = noblend_subfolder


cwafer = wafer(experiment_folders_all, protocol_folders_all, use_alignment_folders, region_strs_all,
        wafer_ids=wafer_ids, region_inds=region_inds, dsstep=dsstep, template_crop_um=template_crop_um,
        crop_um=crop_um, grid_locations=grid_locations, rough_bounding_box=rough_bounding_box,
        region_suffix=use_region_suffix, solved_order=solved_order, wafer_format_str=wafer_format_str,
        delta_rotation_step=delta_rotation_step, delta_rotation_range=delta_rotation_range,
        load_stitched_coords=load_stitched_coords, exclude_regions=exclude_regions,
        translate_roi_center=translate_roi_center, use_thumbnails_ds=use_use_thumbnails_ds,
        thumbnail_folders=thumbnail_folders, roi_polygon_scale=roi_polygon_scale, scale_nm=scale_nm,
        backload_roi_polys=backload_roi_polys, nblocks=nblks, iblock=iblk, block_overlap_um=blk_ovlp_um,
        region_ext=thumbnail_suffix, region_interp_type_deltas=region_interp_type_deltas,
        griddist_um=griddist_um, load_rough_xformed_img=load_rough_xformed_img, legacy_zen_format=legacy_zen_format,
        load_img_subfolder=load_img_subfolder, nimages_per_mfov=nimages_per_mfov,
        use_tissue_masks=tissue_masks, tissue_mask_path=tissue_mask_path, tissue_mask_ds=tissue_mask_ds,
        tissue_mask_fn_str=tissue_mask_fn_str, tissue_mask_min_edge_um=tissue_mask_min_edge_um,
        tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um, tissue_mask_bwdist_um=tissue_mask_bwdist_um,
        init_regions=init_regions, region_manifest_cnts=region_manifest_cnts, region_include_cnts=region_include_cnts,
        zorder=zorder, use_coordinate_based_xforms=use_coordinate_based_xforms, block_overlap_grid_um=blk_ovlp_grid_um,
        verbose=True)


# deals with a legacy workflow (no region range specified), allow manifest index to still be added
if get_export_solved_order: export_solved_order = wafer.region_inds


## run-type specific program-flow
# everything from here on depends on the run-type, some code might be shared by multiple run types.
# nothing except JOB FINISHED should be outside of a conditional block from here on.

if get_wafer_stats:
    cwafer.print_wafer_stats(full_print=False) # xxx - add params for full print? prints all nmfovs in each region

if update_meta_dill:
    if os.path.isfile(meta_dill_fn):
        with open(meta_dill_fn, 'rb') as f: d = dill.load(f)
    else:
        d = {}
    # the scale is ultimately read from the zeiss config based on the pixel resolution.
    # xxx - theoretically it might be easier to just define the resolution in def_common_params,
    #   since it's constant per experiment... this was done, could remove meta dill, but keeping
    #   for now because thinking of using it also for use cases that need to coordinate processes.
    d['scale_um_to_pix'] = cwafer.scale_um_to_pix
    # use this also as a way to reset some dict items
    d['process_uuids'] = {}
    d['order_solving'] = {}
    with open(meta_dill_fn, 'wb') as f: dill.dump(d, f)
    print('meta information saved to ' + meta_dill_fn)
    print('exiting')
    print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
    sys.exit(0)

if write_order:
    # this is a hacky special hook to write the order out.
    tmp = np.arange(1,cwafer.nregions+1)
    tmp = np.delete(tmp, [x-1 for x in exclude_regions])
    tmp.tofile(order_txt_fns[0], sep=' ')
    print('exported order as range')
    sys.exit(0)

if rough_dills_run:
    # special hook to export range as solved order in order to avoid a bunch of special code for solved order
    if cwafer.input_data_type == msem_input_data_types.image_stack or \
            cwafer.input_data_type == msem_input_data_types.hdf5_stack:
        tmp = np.arange(1,cwafer.nregions+1)
        tmp.tofile(order_txt_fns[0], sep=' ')

    if legacy_zen_format:
        if czifile:
            # set doplots=True to show the template matching that creates the limi roi to region mapping.
            # xxx - expose use_full_affine as param? full affine is maybe a bit better.
            # remove_duplicates==True removes any very nearby limi roi centers.
            #   this should be set False when first running with new czi/cz files to determine
            #     if there are any duplicates. once they are removed from def_common_params, then set True.
            cwafer.set_region_rotations_czi(czifile, scene=czifile_scene, ribbon=czifile_ribbon,
                    rotation=czifile_rotation, czfile=czfile, use_roi_polygons=czifile_use_roi_polygons,
                    use_full_affine=True, remove_duplicates=True, doplots=False)
    else:
        if region_rotation is not None:
            rotations = [x + czifile_rotation for x in region_rotation]
        else:
            rotations = [czifile_rotation]
        cwafer.set_region_rotations_manual(rotations=rotations)

    # save the dict items here incase of reinstantiation below, meh.
    d = {'region_to_limi_roi':cwafer.region_to_limi_roi,
         'imaged_order_limi_rotation':cwafer.region_rotations/np.pi*180,
         'imaged_order_region_recon_roi_poly_raw':cwafer.region_recon_roi_poly_raw,
         'limi_to_region_affine':cwafer.limi_to_region_affine, # just for reference, not used
         'template_roi_points':None,
         }
    # this rotatation is made instead of the czi file rotation for any slices where the angle
    # that is stored in the czifile is wrong. it is dependent on the roi points (template)
    # remaing the same, at least within a wafer, but this is the typical experimental workflow.
    # specify -1 to disable and only use the czifile (czfile) angles.
    if template_slice_ind >= 0:
        print('Loading template slice %d to get czifile rotated roi points' % (template_slice_ind,))
        # this workflow is ugly because it's a combination of workarounds for two separate acquisition problems.
        # in order to support the backloaded reconstructed rois, wafer neeeds to be reinstantiated here.
        cwafer = wafer(experiment_folders_all, protocol_folders_all, use_alignment_folders, region_strs_all,
                wafer_ids=wafer_ids, region_inds=region_inds, dsstep=dsstep, template_crop_um=template_crop_um,
                crop_um=crop_um, grid_locations=grid_locations, rough_bounding_box=rough_bounding_box,
                region_suffix=use_region_suffix, solved_order=solved_order, wafer_format_str=wafer_format_str,
                delta_rotation_step=delta_rotation_step, delta_rotation_range=delta_rotation_range,
                load_stitched_coords=load_stitched_coords, exclude_regions=exclude_regions,
                translate_roi_center=translate_roi_center, use_thumbnails_ds=use_use_thumbnails_ds,
                thumbnail_folders=thumbnail_folders, roi_polygon_scale=roi_polygon_scale,
                scale_nm=scale_nm, region_interp_type_deltas=region_interp_type_deltas,
                legacy_zen_format=legacy_zen_format, backload_roi_polys=cwafer.region_recon_roi_poly_raw,
                tissue_mask_path=tissue_mask_path, tissue_mask_ds=tissue_mask_ds,
                tissue_mask_fn_str=tissue_mask_fn_str, tissue_mask_min_edge_um=tissue_mask_min_edge_um,
                tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um,
                tissue_mask_bwdist_um=tissue_mask_bwdist_um, nimages_per_mfov=nimages_per_mfov,
                region_manifest_cnts=region_manifest_cnts, region_include_cnts=region_include_cnts, zorder=zorder,
                nblocks=nblks, iblock=iblk, block_overlap_um=blk_ovlp_um, block_overlap_grid_um=blk_ovlp_grid_um,
                use_coordinate_based_xforms = use_coordinate_based_xforms)
        cwafer.set_region_rotations_manual(d['imaged_order_limi_rotation'])

        # loading the whole slice just to get the roi points is way too heavy weight,
        #   and also forces the regions to have been exported already before the rough dills.
        #   the rough dills have to have been created in order to use the backloaded rois in run_regions.
        #   the only other workaround would be to disable backloaded rois for regions until after
        #     all the regions are exported, then go back and redo the regions... unideal.
        #_, d['template_roi_points'] = cwafer._region_load(template_slice_ind-1) # this is what not to do
        ind = template_slice_ind-1
        roi_points = cwafer.region_roi_poly[ind]; ang_rad = cwafer.region_rotations[ind]
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R_backwards = np.array([[c, s], [-s, c]]) # rotate backwards for points
        # _region_load also applies a translation based on the image size, but this template is ONLY
        #   used to get a matching angle for other slices, so just rotate around polygonal center instead.
        ctr = PolyCentroid(roi_points[:,0], roi_points[:,1])
        d['template_roi_points'] = np.dot(roi_points-ctr, R_backwards.T) + ctr

    # save region to limi roi (czifile) mapping, rotations and other limi-based info
    with open(limi_dill_fns[0], 'wb') as f: dill.dump(d, f)
    print('rough dills exported, exiting')
    print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
    sys.exit(0) # uncomment to only re-export the rough dills

if not (get_wafer_stats or rough_dills_run):
    # set the limi angles (from the czi file and region polygon alignment).
    # NOTE: imaged_order_limi_rotation is only actually stored in the limi dicts.
    #   it's copied over to the rough_dict above if this is not rough_dills_run.
    if rough_export_run or fine_export_run:
        rotations = rough_dicts[0]['imaged_order_limi_rotation'][cwafer.region_inds-1]
    else:
        rotations = [d['imaged_order_limi_rotation'][r-1] for d,r in zip(rough_dicts, cwafer.region_inds)]
    cwafer.set_region_rotations_manual(rotations) # these rotations are specified in degrees

    # optionally set the angle based on rigid point matching of the roi points to a template slice.
    if rough_dicts[0]['template_roi_points'] is not None:
        if nwafer_ids == 1:
            roi_points = rough_dicts[0]['template_roi_points']*to_native_scale
        else:
            roi_points = [rough_dicts[0]['template_roi_points']*to_native_scale,
                    rough_dicts[1]['template_roi_points']*to_native_scale]
        # set doplots=True to show a scatter of the roi points and fit.
        cwafer.set_region_rotations_roi(roi_points, diff_warn_deg=0.1, index_roi_points=nwafer_ids>1, doplots=False)

    # this is to counter sudden changes that can occur in roi between wafers.
    # this happens because roi template is typically reset between wafers.
    cwafer.set_region_translations_manual(roi_polygon_translations*to_native_scale)

    # unless this is the rough_order export that does not include rough alignment, set rough affines
    if load_rough_alignment:
        # scale translations if native
        imaged_order_affines = [None]*len(rough_dicts)
        for i in range(len(rough_dicts)):
            if i > 0 and nwafer_ids==1: break # for single wafer there is only one rough_dict copy
            for j in range(len(rough_dicts[i]['imaged_order_affines'])):
                if rough_dicts[i]['imaged_order_affines'][j] is not None:
                    rough_dicts[i]['imaged_order_affines'][j][:2,2] *= to_native_scale
        if rough_export_run or fine_export_run:
            for i in range(cwafer.nregions):
                cwafer.region_affines[i] = rough_dicts[0]['imaged_order_affines'][cwafer.region_inds[i]-1]
        else:
            cwafer.region_affines = [d['imaged_order_affines'][r-1] for d,r in zip(rough_dicts, cwafer.region_inds)]

if rough_export_run:
    if validate_region_grid:
        cwafer._validate_region_grid(start=export_region_beg, per_figure=1, use_solved_order=do_export_solved_order,
            show_grid=True, show_patches=False)
        sys.exit(0)

    ndsexports = len(dsexports)
    if not rough_hdf5:
        # specifies the "normal" rough export used by wafer_solver
        solver_thumbs_export = (not load_solved_order and not load_rough_alignment and \
                ndsexports == 0 and blending_features and not overlays)
        export_str = ('overlays' if overlays else 'thumbnails') + \
                ('_native' if native else '') + ('_solved_order' if load_solved_order else '') + \
                (('_' + rough_run_str + '_aligned') if load_rough_alignment else '') + \
                ('_noblend' if not blending_features else '')
        if solver_thumbs_export:
            i = 1 if tissue_masks else 0
            if rough_export_solve_order:
                export_path = os.path.join(cwafer.alignment_folders[0], thumbnail_subfolders_order[i])
            else:
                tmp = thumbnail_subfolders[i]
                if thumbs_run_str: tmp += ('-' + thumbs_run_str)
                export_path = os.path.join(cwafer.alignment_folders[0], tmp)
            save_roi_points = not tissue_masks
            order_name_str = ''
        else:
            export_path = os.path.join(meta_folder, 'rough_alignment_exports', export_str)
            if use_custom_crop: export_path += '_custom_crop'
            if tissue_masks: export_path += '_masks'
            save_roi_points = False
        if ndsexports > 0:
            use_dssteps = dsexports
            export_paths = [export_path + '_ds{}'.format(y) for y in dsexports]
        else:
            use_dssteps = [tissue_mask_ds] if tissue_masks else [dsthumbnail]
            export_paths = [export_path]

        # force disable crop and zero for some export types
        enable_crop_and_zero = (not overlays and (load_rough_alignment or use_custom_crop))
        _crop_to_grid = crop_to_grid and enable_crop_and_zero
        _zero_outside = zero_outside and enable_crop_and_zero

        img_suffix = thumbnail_suffix
        _use_solved_order = do_export_solved_order
        _export_solved_order = export_solved_order
        _do_overlays = overlays
        _save_roi_points = save_roi_points
        save_h5 = export_h5
    else:
        export_paths = [os.path.join(cwafer.alignment_folders[0], load_rough_xformed_img_subfolder)]
        img_suffix = rough_aligned_suffix
        use_dssteps = [1]
        _use_solved_order = _do_overlays = _crop_to_grid = _zero_outside = _save_roi_points = False
        _export_solved_order = None
        save_h5 = True

    cwafer.export_regions(export_paths, [img_suffix]*len(export_paths), save_h5=save_h5, dssteps=use_dssteps,
            use_solved_order=_use_solved_order, export_solved_order=_export_solved_order,
            do_overlays=_do_overlays, crop_to_grid=_crop_to_grid, zero_outside_grid=_zero_outside,
            save_roi_points=_save_roi_points, start=export_region_beg, stop=export_region_end,
            order_name_str=order_name_str, tissue_masks=tissue_masks, is_excluded=is_excluded,
            save_masks_in=save_masks_in, verbose_load=native)

# load the previous fine alignment results for exporting aligned tiffs
if fine_export_run and not init_locks:
    assert(nwafer_ids == 1) # no multiple wafer export runs

    with open(fine_dill_fns[0], 'rb') as f: fine_dict = dill.load(f)

    cwafer.deformation_points = fine_dict['deformation_points']*to_native_scale
    # The deformations need to be mapping from dst->src. This is what is required
    #   by basically all coordinate remapping library functions that apply general warping xforms.
    if 'imaged_order_forward_deformations' in fine_dict:
        key_str = 'imaged_order_forward_deformations'; sign = -1.
    else:
        key_str = 'imaged_order_reverse_deformations'; sign = 1.
    dataset = 'imaged_order_deltas'
    h5fn = fine_dill_fns[i] + '.h5'
    if os.path.isfile(h5fn):
        shp, dtype = big_img_info(h5fn, dataset=dataset)
        shp = np.array(shp); shp[0] = 1
        data = np.empty(shp, dtype=dtype)
        for i in range(cwafer.nregions):
            ind = cwafer.region_inds[i]-1
            big_img_load(h5fn, img_blk=data, dataset=dataset, custom_slc=np.s_[ind:ind+1,:,:])
            cwafer.imaged_order_deformation_vectors[i] = sign * data.reshape(-1,2) * to_native_scale
    else:
        # legacy mode, before fine reslice implemented, where deltas are also stored in the dill
        for i in range(cwafer.nregions):
            cwafer.imaged_order_deformation_vectors[i] = \
                sign*fine_dict[key_str][cwafer.region_inds[i]-1]*to_native_scale
        fine_dict = None; del fine_dict # with lots of grid points, this can be quite large

if validate_region_grid and (fine_export_run or fine_run):
    cwafer._validate_region_grid(start=0, per_figure=1, use_solved_order=True, bg_fill_type='noise',
        show_patches=False)

if fine_export_run:
    export_path = os.path.join(meta_folder, 'fine_alignment_exports', fine_run_str)
    img_suffix = '_fine_aligned' # xxx - paramterize?
    if native and not convert_hdf5s:
        dsexports = [1]
        export_paths = [export_path + '_native']
    else:
        if len(dsexports) == 0: dsexports = [1]
        export_paths = [export_path + ('_native' if native else '') + ('_custom_crop' if use_custom_crop else '') \
                + '_ds{}'.format(y) for y in dsexports]

        if convert_hdf5s:
            # for the rare cases where you need ds1 tiffs, so does not write to same dir as h5 export
            if dsexports[0] == 1: export_paths[0] += '_tiffs'
            # use the first slot to store the ds1 native hdf5 location
            dsexports = [1] + dsexports
            export_paths = [export_path + ('_native' if native else '_ds1')] + export_paths

    # force disable crop and zero for some export types
    enable_crop_and_zero = (not convert_hdf5s or use_custom_crop)
    _crop_to_grid = (crop_to_grid or use_custom_crop) and enable_crop_and_zero
    _zero_outside = zero_outside and enable_crop_and_zero

    cwafer.export_regions(export_paths, [img_suffix]*len(export_paths), dssteps=dsexports, init_locks=init_locks,
            use_solved_order=do_export_solved_order, export_solved_order=export_solved_order,
            crop_to_grid=_crop_to_grid, zero_outside_grid=_zero_outside, start=export_region_beg,
            stop=export_region_end, convert_hdf5s=convert_hdf5s, save_h5=export_h5, verbose_load=True)

if fine_run:
    direction_str = 'reverse' if invert_order else 'forward'
    # NOTE: confusing, but the cross wafer alignments are always stored in the lower wafer (folder and dill name),
    #   which is the reason for this ind conditional.
    ind = 1 if nwafer_ids > 1 and invert_order else 0
    dill_fn = os.path.join(use_alignment_folders[ind], delta_dill_fn_str.format(delta_run_str, wafer_ids[ind], iorder))
    os.makedirs(os.path.dirname(dill_fn), exist_ok=True)
    assert(fine_init_dills or os.path.isfile(dill_fn)) # use fine-export-dills to predump empty dills

    if crops_um_ind > 0:
        subkey = 'crop{}-skip{}-{}'.format(crops_um_ind-1, skip_slices, direction_str)
        d = dill_lock_and_load(dill_fn)
        grid_selects = np.zeros((grid_locations.shape[0],), dtype=bool)
        #grid_selects[d[subkey]['outliers']] = 1
        grid_selects[d[subkey]['nearby_outliers']] = 1
        print('Rerunning at crop {},{} for {} remaining outliers'.format(crop_um[0], crop_um[1], grid_selects.sum()))
    else:
        grid_selects = None

    ## to run delta validation
    #with open(dill_fn, 'rb') as f: d = dill.load(f)
    #cwafer.wafer_grid_deltas = d['wafer_grid_deltas']
    #cwafer.wafer_grid_Cvals = d['wafer_grid_Cvals']
    #cwafer._validate_align_regions(gridnums=range(5))

    doplots = (show_xcorr_plots or save_xcorr_plots)
    dosave_path = plots_folder if save_xcorr_plots else ''
    # save different crop, skip and direction runs in the same dills to keep file count down.
    subkey = 'crop{}-skip{}-{}'.format(crops_um_ind, skip_slices, direction_str)
    if not fine_init_dills:
        # if this comparison bridges the specified blur indices, then apply further blurring
        iblur = -1
        for i in range(len(slice_blur_z_indices)):
            if any([(slice_blur_z_indices[i][0] - x) >= 0 for x in zorder]) and \
                    any([(x - slice_blur_z_indices[i][1]) >= 0 for x in zorder]):
                iblur = i; break
        if iblur > -1:
            print('Comparison between slices {} and {}, zorders {} and {}, crosses blur indices {} to {}'.format(\
                region_inds[0], region_inds[1], zorder[0], zorder[1],
                slice_blur_z_indices[iblur][0], slice_blur_z_indices[iblur][1]))
            cwafer._proc_whiten_sigma *= slice_blur_factor

        if not fine_blur_only or iblur > -1:
            # uncomment for hacky way to save xcorr inputs and outputs
            #cwafer.export_xcorr_comps_path = # export path
            cwafer.align_regions(grid_selects=grid_selects, doplots=doplots, dosave_path=dosave_path,
                nworkers=arg_nworkers)
            d, f1, f2 = dill_lock_and_load(dill_fn, keep_locks=True)
            if subkey in d:
                sel = np.isfinite(cwafer.wafer_grid_Cvals)
                d[subkey]['wafer_grid_deltas'][sel,:] = cwafer.wafer_grid_deltas[sel,:]
                d[subkey]['wafer_grid_Cvals'][sel] = cwafer.wafer_grid_Cvals[sel]
            else:
                d[subkey] = {'wafer_grid_deltas':cwafer.wafer_grid_deltas, 'wafer_grid_Cvals':cwafer.wafer_grid_Cvals,}
            d[subkey]['iblur'] = iblur
            dill_lock_and_dump(dill_fn, d, f1, f2)
        else:
            print('--fine-blur-only specified and not a blur comparison, skipping')
    else:
        # for convenience always use the solved order size to know how many dills to init.
        # the lower value should be based on the max skipped planned to be run.
        for i in range(fine_init_dills_rng[0],rough_dicts[0]['solved_order'].size-1):
            dill_fn = os.path.join(use_alignment_folders[ind], delta_dill_fn_str.format(delta_run_str,
                wafer_ids[ind], i))
            print(dill_fn)
            assert( not os.path.isfile(dill_fn) ) # safety feature, delete by hand first if you want to start over
            if not os.path.isfile(dill_fn):
                d = {'False->True':True}
                with open(dill_fn, 'wb') as f: dill.dump(d, f)



# do not delete this, is used at a minimum for addressing "timed-out" jobs when running on the cluster (to be re-run).
# can also grep to count how many jobs of a particular run have completed without fatal error.
print('JOB FINISHED: run_wafer.py')
print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
