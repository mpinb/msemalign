#!/usr/bin/env python3
"""run_wafer_aggregator.py

Top level command-line interface for aggregating the alignments computed
  for the rough and fine alignments by applying the LSS.

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
import dill
import os
import sys
import argparse
import time

from msem import wafer_aggregator
from msem.utils import make_hex_points, dill_lock_and_load, dill_lock_and_dump
from msem.utils import l2_and_delaunay_distance_select, big_img_save

# all parameters loaded from an experiment-specific import
from def_common_params import get_paths, crops_um, all_wafer_ids, exclude_regions
from def_common_params import fine_dill_fn_str, delta_dill_fn_str, meta_dill_fn_str
from def_common_params import rough_dill_fn_str, rough_rigid_dill_fn_str
from def_common_params import rough_affine_dill_fn_str, rough_rigid_affine_dill_fn_str
from def_common_params import rough_distance_cutoff_um, fine_min_valid_slice_comparisons, outlier_affine_degree
from def_common_params import fine_residual_threshold_um, inlier_min_neighbors, inlier_min_component_size_edge_um
from def_common_params import C_hard_cutoff, min_percent_inliers_C_cutoff, ninlier_neighhbors_cmp
from def_common_params import ok_outlier_zscore, not_ok_inlier_zscore, fine_solve_reverse, region_interp_type_deltas
from def_common_params import rough_bounding_box_xy_spc, rough_grid_xy_spc, fine_grid_xy_spc
from def_common_params import fine_neighbor_dist_scale, rough_neighbor_dist_scale, fine_nearby_points_um
from def_common_params import rough_smoothing_radius_um, rough_smoothing_std_um, rough_smoothing_weight
from def_common_params import fine_smoothing_radius_um, fine_smoothing_std_um, fine_smoothing_weight
from def_common_params import rough_smoothing_neighbors, fine_smoothing_neighbors, fine_filtering_shape_um
from def_common_params import merge_inliers_blk_cutoff, merge_inliers_min_comp, merge_inliers_min_hole_comp
from def_common_params import fine_interp_weight, fine_interp_neighbor_dist_scale, slice_blur_z_indices
from def_common_params import interp_inliers, interp_inlier_nneighbors, affine_rigid_type
from def_common_params import rough_regression_remove_bias, fine_regression_remove_bias, z_neighbors_radius

# <<< turn on stack trace for warnings
import traceback
import warnings

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
#warnings.simplefilter("always")
# turn on stack trace for warnings >>>


## argparse

parser = argparse.ArgumentParser(description='run_wafer_aggregator.py')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
                    help='wafers to accumulate alignments across (can be single or multiple)')
parser.add_argument('--all-wafers', dest='all_wafers', action='store_true',
                    help='instead of specifying wafer id(s), include all wafers for dataset')
parser.add_argument('--run-type', nargs=1, type=str, default=['rough'],
                    choices=['fine', 'fine_outliers', 'fine_filter', 'fine_affine_raw', 'fine_affine', 'fine_interp',
                             'fine_reslice', 'rough', 'rough_merge', 'rough_status'],
                    help='the type of run to choose')
parser.add_argument('--run-str-in', nargs='+', type=str, default=['none'],
                    help='string to differentiate alignments with different parameters')
parser.add_argument('--run-str-out-rough', nargs=1, type=str, default=['none'],
                    help='string to differentiate alignments with different parameters')
parser.add_argument('--run-str-out-fine', nargs=1, type=str, default=['none'],
                    help='string to differentiate alignments with different parameters')
parser.add_argument('--filtered-fine-deltas', dest='filtered_fine_deltas', action='store_true',
                    help='for fine delta accumulation, use the filtered deltas as input / filter output')
parser.add_argument('--use-fine-reslice', dest='use_fine_reslice', action='store_true',
                    help='for fine delta accumulation, load resliced data created by fine_reslice mode')
parser.add_argument('--keep-xcorrs', dest='keep_xcorrs', action='store_true',
                    help='for fine reslice, specify to also save the reslices xcorr values')
parser.add_argument('--order-range', nargs=2, type=int, default=[0, 0],
                    help='specify to only run a subset of the aggretation using slice orders (1-based)')
parser.add_argument('--max-skips', nargs='+', type=int, default=[0],
                    help='maximum skip aligments to consider, for all (rough/fine) or each (fine) crop')
parser.add_argument('--max-crops', nargs=1, type=int, default=[1],
                    help='maximum number of crops in the crop series to consider')
parser.add_argument('--use-rough-skip', nargs=1, type=int, default=[-1],
                    help='if required, which rough alignment skip to use (default 0)')
parser.add_argument('--L1_norm', nargs=1, type=float, default=[0.0],
                    help='L1 norm to use for reconciler, use 0. for off')
parser.add_argument('--L2_norm', nargs=1, type=float, default=[0.0],
                    help='L2 norm to use for reconciler, use 0. for off')
parser.add_argument('--plot-deltas-debug', action='store_true',
                    help='show debug quiver plots of deltas before and after replacing points')
parser.add_argument('--save-rough-sequences', nargs=1, type=str, default=[''],
                    help='for rough_status file to save only the clean sequences')
parser.add_argument('--percent-matches-topn', nargs=1, type=int, default=[0],
                    help='preprocess percent matches for solver by only keeping top n')
parser.add_argument('--iprocess', nargs=1, type=int, default=[0],
                    help='process index for jobs that are divided into multiple processes')
parser.add_argument('--nprocesses', nargs='+', type=int, default=[1],
                    help='split some operations into multiple processes')
parser.add_argument('--nworkers', nargs=1, type=int, default=[1],
                    help='split some operations into multiple threads (same-node processes)')
parser.add_argument('--merge', dest='merge', action='store_true',
                    help='merge results from appropriate multiple process runs into a single dill')
parser.add_argument('--zero-blur', dest='zero_blur', action='store_true',
                    help='zero out the deltas at jump "blur" locations')
parser.add_argument('--verbose', action='store_true', help='print messages for each accumulation iteration')
# make ransac iterations easily controlable from command line
parser.add_argument('--ransac-repeats', nargs=1, type=int, default=[1],
                    help='if ransac is run (fine outliers), parallized by workers')
parser.add_argument('--ransac-max', nargs=1, type=int, default=[1000],
                    help='if ransac is run (fine outliers)')

# options for blockwise processing, for when grid sizes are very large (in number of points)
parser.add_argument('--nblocks', nargs=2, type=int, default=[1, 1],
                    help='number of partitions per dimension for blockwise processing')
parser.add_argument('--iblock', nargs=2, type=int, default=[0, 0],
                    help='which block to process for blockwise processing (zero-based)')
parser.add_argument('--block-overlap-um', nargs=2, type=float, default=[0., 0.],
                    help='amount of overlap between blocks in um for blockwise processing')
args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

# which wafers to accumulate across
wafer_ids = args['wafer_ids']

# the max number of skip alignments to consider.
# for fine/rough aggregation, dill files up to this skip value have to have been saved.
range_skips = [x + 1 for x in args['max_skips']]
max_range_skips = max(range_skips)

# for rough_status and rough (legacy accumulator), which skip index to look at (must be less than max).
use_rough_skip = args['use_rough_skip'][0]

# for the fine alignment only, how many crops in the crop sequence to load.
# higher sized crops are run only for outliers remaining after previous crops considered.
range_crops = args['max_crops'][0]

# for rough and fine reconciliation, weight to use for L1 and L2 normalizations
#reconcile_L1_norm = 0. # for off
#reconcile_L2_norm = 0. # for off
reconcile_L1_norm = args['L1_norm'][0]
reconcile_L2_norm = args['L2_norm'][0]

# for debug, show quiver plots of the deltas
plot_deltas = args['plot_deltas_debug']

# this is an identifier so that multiple rough/fine alignemnts can be exported / loaded easily.
run_str_in = args['run_str_in'][0]
run_strs_in = args['run_str_in'] # some modes can take multiple inputs
run_str_out_rough = args['run_str_out_rough'][0]
run_str_out_fine = args['run_str_out_fine'][0]

# specify subset of slices to accumulate by specifying range in slice order
order_range = np.array(args['order_range'])-1

# preprocess percent matches before order solved to only use top n
percent_matches_topn = args['percent_matches_topn'][0]

# number of total processes for parallizing some of the run types.
nprocesses = args['nprocesses'][0]
nprocesses_all = args['nprocesses'] # for a few "double merge" scenarios

# which process this is for parallized run types.
iprocess = args['iprocess'][0]

# for same-node process parallizations
arg_nworkers = args['nworkers'][0]

# some multiple process runs will need outputs merged into single dills
run_merge = args['merge']

# specify to use "afffine-filtered" deltas as input to the fine accumulation
filtered_fine_deltas = args['filtered_fine_deltas']

# specify to load the resliced data from fine_reslice as input to the fine accumulation.
# this workflow is necessary for very large grids that must use (lots of) x/y blocking.
use_fine_reslice = args['use_fine_reslice']

# optional file to save the clean rough sequences for use in manually creating final solved order
save_rough_sequences = args['save_rough_sequences'][0]

# for fine_reslice, optionally also save the actual xcorr values in the reslice file
keep_xcorrs = args['keep_xcorrs']

# number of (parallized) ransac repeats and iterations per repeat
# NOTE: for ZF ultrafine used 4 and 250000 respectively (used to be set above init in wafer_aggregator)
ransac_repeats = args['ransac_repeats'][0]
ransac_max = args['ransac_max'][0]

# options for blockwise processing, currently only supported for (ultrafine) outlier detection
# these are not inverted like in wafer because they work on the grid points,
#   not on image shape like in wafer.
nblks = args['nblocks']
iblk = args['iblock']
blk_ovlp_um = args['block_overlap_um']

# optionally just set the deltas to zero at the "blur" locations.
# typically used during the UF alignment.
zero_blur = args['zero_blur']

# run type string is used for some filenames / paths
run_type = args['run_type'][0]

# these specify what type of run this is (one of these must be set True)

# specify to accumulate the deltas
fine_accumulate = run_type == 'fine'

# specify to run the fine outlier detection
fine_outliers = run_type == 'fine_outliers'

# run interpolation on inliers, must be run after outlier detection
fine_interp = run_type == 'fine_interp'

# specify to filter fine deltas with affines
fine_filter = run_type == 'fine_filter'

# specify to fit fine deltas with affines in and export as input to rough recon
fine_affine_raw = run_type == 'fine_affine_raw'

# specify to fit solved fine deltas with affines and export to rough affine
fine_affine = run_type == 'fine_affine'

# mode that writes out deltas computed for reconcile to h5 file(s).
# this is an optimization so that they can be loaded as blocks without loading all of them.
fine_reslice = run_type == 'fine_reslice'

# specify to print information regarding rough alignment (bad matches)
print_rough_status = run_type == 'rough_status'

# specify to "reconcile" rough alignments by recomputing affines across multiple i+n alignments
rough_reconcile = (run_type in ['rough', 'rough_merge'])

# this is to merge the rough affine refits, that are also parallelized
rough_reconcile_merge = run_type == 'rough_merge'

# set defaults differently depending on fine or rough alignemnt
any_rough = (print_rough_status or rough_reconcile)
fine_accumulate = fine_accumulate or fine_reslice # fine reslice is a special mode for accumulate
any_fine = (fine_accumulate or fine_outliers or fine_interp or fine_affine or fine_filter)
rerough = ((any_rough or fine_affine) and len(run_strs_in) > 1)

run_str_out = run_str_out_rough if any_rough else run_str_out_fine

# status prints from each accumulate iteration (force on for deltas or angles)
verbose_iterations = args['verbose']

print(('run_wafer_aggregator run-type==%s run-str-in==%s run-str-out-rough==%s run-str-out-fine==%s ' + \
       'range-skips==%d L1_norm==%g L2_norm==%g') % (run_type, run_str_in, run_str_out_rough,
        run_str_out_fine, max_range_skips, reconcile_L1_norm, reconcile_L2_norm))
if rerough:
    print('\trough on top of previous rough {}'.format(run_strs_in[1]))

# set wafer_ids to contain all wafers, if specified
if args['all_wafers']:
    wafer_ids = list(all_wafer_ids)
print('Aggregating wafers:'); print(wafer_ids)



## fixed parameters not exposed in def_common_params

# meta files contain the data across all wafers
if fine_affine:
    agg_dill_fn_str = 'accum_meta_rough' + '.' + run_str_out + '.dill'
    agg_rigid_dill_fn_str = 'accum_meta_rough' + '.' + run_str_out + '_rigid.dill'
    agg_dill_in_fn_str = 'accum_meta_' + run_type.split('_')[0] + '.' + run_str_in + '.dill'
else:
    agg_dill_fn_str = 'accum_meta_' + run_type.split('_')[0] + '.' + run_str_out + '.dill'
    agg_rigid_dill_fn_str = 'accum_meta_' + run_type.split('_')[0] + '.' + run_str_out + '_rigid.dill'

# option to use interpolated deltas during fine accumulation, probably leave this always true
load_interpolated_deltas = True

# name of the hdf5 file to save the "resliced" items necessary for the fine recon
fine_reslice_fn = 'fine_reslice.h5'


### INITS

use_slice_blur_z_indices = slice_blur_z_indices if zero_blur else []

# default < 0 means that no rough skip was specified, default to skip 0
irough_skip = use_rough_skip if use_rough_skip >= 0 else 0
use_rough_skip = (use_rough_skip >= 0)
assert( irough_skip < max_range_skips )
assert( not any_rough or len(range_skips)==1 )

assert(range_crops > 0)
use_crops_um = [crops_um[x] for x in range(range_crops)]

# this is to support reducing the number of skips at higher crop sizes.
# xxx - possible something could break here if more skips were specified at higher crop sizes.
if not any_rough:
    if len(range_skips)==1:
        range_skips = [range_skips[0] for x in range(range_crops)]
    else:
        assert( len(range_skips)==range_crops )
        assert( all(x>=y for x, y in zip(range_skips, range_skips[1:])) ) # meant for less skips at larger crops

## parameters that are determined based on above parameters

rough_bounding_box = make_hex_points(*rough_bounding_box_xy_spc, bbox=True)
rough_alignment_grid_um = make_hex_points(*rough_grid_xy_spc)
grid_locations_um = make_hex_points(*fine_grid_xy_spc)
griddist_um = fine_grid_xy_spc[2]

# several modes allow block by block processing in the case of very large number of grid points.
single_block = all([abs(x)==1 for x in nblks])
single_proc = (nprocesses==1 and single_block)


nwafer_ids = len(wafer_ids)
experiment_folders_all = [None]*nwafer_ids
#thumbnail_folders = [None]*nwafer_ids
alignment_folders = [None]*nwafer_ids
region_strs_all = [None]*nwafer_ids
region_strs = [None]*nwafer_ids
#protocol_folders_all = [None]*nwafer_ids
fine_dill_fns = [None]*nwafer_ids
rough_dill_fns = [None]*nwafer_ids
rough_dill_out_fns = [None]*nwafer_ids
rough_rigid_dill_fns = [None]*nwafer_ids
rough_dicts = [None]*nwafer_ids
rough_affine_skip_dill_fns = [[None]*max_range_skips for x in range(nwafer_ids)]
rough_affine_skip_dicts = [[None]*max_range_skips for x in range(nwafer_ids)]

# outputs for fine_affine_raw mode
rough_out_affine_skip_dill_fns = [[None]*max_range_skips for x in range(nwafer_ids)]
rough_out_rigid_affine_skip_dill_fns = [[None]*max_range_skips for x in range(nwafer_ids)]

# get all the dill filenames, and load input dill dicts
for j,i in zip(wafer_ids, range(nwafer_ids)):
    #experiment_folders_all[i], thumbnail_folders[i], protocol_folders_all[i], alignment_folders[i], meta_folder, \
    #    region_strs_all[i] = get_paths(j)
    experiment_folders_all[i],_,_, alignment_folders[i], meta_folder, region_strs_all[i] = get_paths(j)
    # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
    region_strs[i] = [item for sublist in region_strs_all[i] for item in sublist]

    # this is for msem_input_data_types.image_stack, xxx - instwafer
    valid_experiment_folders = any([x for x in experiment_folders_all])

    fine_dill_fns[i] = os.path.join(alignment_folders[i], fine_dill_fn_str.format(j, run_str_out_fine))
    rough_dill_fns[i] = os.path.join(alignment_folders[i], rough_dill_fn_str.format(j, run_str_out_rough))
    rough_rigid_dill_fns[i] = os.path.join(alignment_folders[i], rough_rigid_dill_fn_str.format(j,run_str_out_rough))
    if fine_affine:
        rough_dill_out_fns[i] = os.path.join(alignment_folders[i], rough_dill_fn_str.format(j, run_str_out_fine))

    if any_rough:
        for k in range(max_range_skips):
            if use_rough_skip and k != irough_skip: continue
            rough_affine_skip_dill_fns[i][k] = os.path.join(alignment_folders[i],
                    rough_affine_dill_fn_str.format(j, k, run_str_in))
            with open(rough_affine_skip_dill_fns[i][k], 'rb') as f: d = dill.load(f)
            rough_affine_skip_dicts[i][k] = d
    if (valid_experiment_folders and (any_fine or rerough)) and not print_rough_status:
        rough_dill_fn = rough_dill_fns[i] if not rerough else \
            os.path.join(alignment_folders[i], rough_dill_fn_str.format(j, run_strs_in[1]))
        with open(rough_dill_fn, 'rb') as f: rough_dicts[i] = dill.load(f)

    if fine_affine_raw:
        for k in range(max_range_skips):
            rough_out_affine_skip_dill_fns[i][k] = os.path.join(alignment_folders[i],
                    rough_affine_dill_fn_str.format(j, k, run_str_out_fine))
            rough_out_rigid_affine_skip_dill_fns[i][k] = os.path.join(alignment_folders[i],
                    rough_rigid_affine_dill_fn_str.format(j, k, run_str_out_fine))

# meta dill stores aggregated alignments unrolled over wafers
agg_dill_fn = os.path.join(meta_folder, agg_dill_fn_str)
agg_rigid_dill_fn = os.path.join(meta_folder, agg_rigid_dill_fn_str)

# h5 files that temporarily stores the fine recone information, and is easily slice-able by blocks
fine_reslice_fn = os.path.join(meta_folder, fine_reslice_fn)

# NOTE: grid needs to be the same for all wafers
meta_dill_fn = os.path.join(meta_folder, meta_dill_fn_str)
#with open(meta_dill_fn, 'rb') as f: meta_dict = dill.load(f)
meta_dict = dill_lock_and_load(meta_dill_fn) # for multiprocess aggregator, meta dill used to coordinate procs
rough_bounding_box_pixels = [x*meta_dict['scale_um_to_pix'] for x in rough_bounding_box]
grid_locations_pixels = grid_locations_um*meta_dict['scale_um_to_pix']
griddist_pixels = griddist_um*meta_dict['scale_um_to_pix']
rough_alignment_grid_pixels = rough_alignment_grid_um*meta_dict['scale_um_to_pix']
rough_distance_cutoff_pixels = rough_distance_cutoff_um*meta_dict['scale_um_to_pix']
residual_threshold_pixels = fine_residual_threshold_um*meta_dict['scale_um_to_pix']
inlier_min_component_size_edge_pixels = inlier_min_component_size_edge_um*meta_dict['scale_um_to_pix']
rough_smoothing_radius_pixels = rough_smoothing_radius_um*meta_dict['scale_um_to_pix']
rough_smoothing_std_pixels = rough_smoothing_std_um*meta_dict['scale_um_to_pix']
fine_smoothing_radius_pixels = fine_smoothing_radius_um*meta_dict['scale_um_to_pix']
fine_smoothing_std_pixels = fine_smoothing_std_um*meta_dict['scale_um_to_pix']
if fine_nearby_points_um is not None:
    fine_nearby_points_pixels=fine_nearby_points_um*meta_dict['scale_um_to_pix']
else:
    fine_nearby_points_pixels = None
fine_filtering_shape_pixels = np.array(fine_filtering_shape_um)*meta_dict['scale_um_to_pix']
blk_ovlp_pix = np.array(blk_ovlp_um)*meta_dict['scale_um_to_pix']
if ok_outlier_zscore < 0:
    # interpret this as a residual value in microns, not a zscore
    ok_outlier_zscore = ok_outlier_zscore*meta_dict['scale_um_to_pix']
if not_ok_inlier_zscore < 0:
    # interpret this as a residual value in microns, not a zscore
    not_ok_inlier_zscore = not_ok_inlier_zscore*meta_dict['scale_um_to_pix']

# other inits across all wafers to be accumulated.
# array containing solved slice order for all wafers being accumulated.
solved_orders = [None]*nwafer_ids
# array containing total number of regions present in wafer.
# this is the total even if they are included in the solved ordering or not.
wafers_nregions = np.zeros((nwafer_ids,), dtype=np.int32)
for i in range(nwafer_ids):
    if any_rough:
        solved_orders[i] = rough_affine_skip_dicts[i][irough_skip]['solved_order']
        wafers_nregions[i] = rough_affine_skip_dicts[i][irough_skip]['nregions']
    elif valid_experiment_folders:
        solved_orders[i] = rough_dicts[i]['solved_order']
        wafers_nregions[i] = rough_dicts[i]['nregions']
        ## xxx - stupid hack for when old rough alignment dills were accidentally overwritten
        #from def_common_params import order_txt_fn_str
        #tmp_fn = os.path.join(alignment_folders[i], order_txt_fn_str.format(wafer_ids[i]))
        #solved_orders[i] = np.fromfile(tmp_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based
    else:
        # xxx - instwafer, allows us to bypass the rough steps for aligning an image stack
        wafers_nregions[i] = len(region_strs[i])
        solved_orders[i] = np.arange(0,wafers_nregions[i])

if fine_affine:
    agg_dill_in_fn = os.path.join(meta_folder, agg_dill_in_fn_str)
    with open(agg_dill_in_fn, 'rb') as f: agg_dict = dill.load(f)



# xxx - overall it would make more sense to have aggregator as a helper class to wafer,
#   similar to wafer_solver. BUT, I was impatient and did not want to wait for region inits
#   every time I ran the aggregator which can be slow for big wafers. so, there are a few
#   hacky things here to get some wafer properties without having to instantiate a wafer.
#   see places marked with xxx - instwafer
aggregator = wafer_aggregator(wafer_ids=wafer_ids, region_strs=region_strs, solved_orders=solved_orders,
        wafers_nregions=wafers_nregions, rough_bounding_box_pixels=rough_bounding_box_pixels,
        outlier_affine_degree=outlier_affine_degree, residual_threshold_pixels=residual_threshold_pixels,
        inlier_min_neighbors=inlier_min_neighbors, fine_nearby_points_pixels=fine_nearby_points_pixels,
        inlier_min_component_size_edge_pixels=inlier_min_component_size_edge_pixels, C_hard_cutoff=C_hard_cutoff,
        ninlier_neighhbors_cmp=ninlier_neighhbors_cmp, ok_outlier_zscore=ok_outlier_zscore,
        not_ok_inlier_zscore=not_ok_inlier_zscore, min_percent_inliers_C_cutoff=min_percent_inliers_C_cutoff,
        order_range=order_range, region_interp_type_deltas=region_interp_type_deltas, meta_dill_fn=meta_dill_fn,
        ransac_repeats=ransac_repeats, ransac_max=ransac_max, rigid_type=affine_rigid_type,
        verbose_iterations=verbose_iterations, verbose=True)



if rough_reconcile:
    assert(single_block) # xxx - blocking not currently supported for rough reconcile
    affine_percent_matches = None

    forward_affines = [[x['forward_affines'] for x in y] for y in rough_affine_skip_dicts]
    reverse_affines = [[x['reverse_affines'] for x in y] for y in rough_affine_skip_dicts]
    forward_pts_src = [[x['forward_pts_src'] for x in y] for y in rough_affine_skip_dicts]
    forward_pts_dst = [[x['forward_pts_dst'] for x in y] for y in rough_affine_skip_dicts]
    reverse_pts_src = [[x['reverse_pts_src'] for x in y] for y in rough_affine_skip_dicts]
    reverse_pts_dst = [[x['reverse_pts_dst'] for x in y] for y in rough_affine_skip_dicts]
    aggregator.init_rough(rough_alignment_grid_pixels,
                          forward_affines, forward_pts_src, forward_pts_dst,
                          reverse_affines, reverse_pts_src, reverse_pts_dst,
                          affine_percent_matches=affine_percent_matches)

if print_rough_status:
    ind = irough_skip

    tmp = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=sys.maxsize)
    # print any bad matches and cluster slices by groups without bad matches.
    # this is used as input for the manual re-ordering of the clusters needed after slice z-order is solved.
    for i in range(nwafer_ids):
        # these prints are useful for fixing the "bad match" neighbors manually.
        # should only really need this info on the first iteration when some spots in the solved ordering
        #   need to be manually corrected.
        print()
        print('For wafer %d' % (wafer_ids[i],))
        print('Solved region order')
        print(solved_orders[i])
        print('%d bad matches' % (rough_affine_skip_dicts[i][ind]['solved_order_bad_matches'].shape[0],))
        #print('%d bad matches neighboring regions' % \
        #      (rough_affine_skip_dicts[i][ind]['solved_order_bad_matches'].shape[0],))
        #print(rough_affine_skip_dicts[i][ind]['solved_order_bad_matches'])
        print('bad matches solved order inds:')
        print(rough_affine_skip_dicts[i][ind]['solved_order_bad_matches_inds'])
        print('bad matches fail types, parallel to inds:')
        print(rough_affine_skip_dicts[i][ind]['bad_matches_fail_types'])
        print('Solved region order split into good matches sections')
        sections = np.split(solved_orders[i], rough_affine_skip_dicts[i][ind]['solved_order_bad_matches_inds'])
        sections_lens = np.array([x.size for x in sections])
        for j in range(len(sections)):
            with np.printoptions(formatter={'all':lambda x: '{:>3}'.format(x)}):
                print(sections[j])
        if save_rough_sequences:
            bfnseq, extseq = os.path.splitext(save_rough_sequences)
            save_rough_sequences_fn = '{}_w{}{}'.format(bfnseq, wafer_ids[i], extseq)
            if os.path.isfile(save_rough_sequences_fn): os.remove(save_rough_sequences_fn) # computers suck
            with open(save_rough_sequences_fn, "w") as text_file:
                print('{} bad matches'.format(rough_affine_skip_dicts[i][ind]['solved_order_bad_matches'].shape[0],),
                    file=text_file)
                print(file=text_file)
                print('sections_lens', file=text_file)
                print(sections_lens, file=text_file)
                print(file=text_file)
                print('{} sections with length == 1'.format((sections_lens == 1).sum()), file=text_file)
                print('{} sections with length <= 5'.format((sections_lens <= 5).sum()), file=text_file)
                print('all section lengths with lengths > 5', file=text_file)
                print(sections_lens[sections_lens > 5], file=text_file)
                print(file=text_file)
                with np.printoptions(formatter={'all':lambda x: '{:>3}'.format(x)}):
                    for j in range(len(sections)):
                        print(sections[j], file=text_file)

            # also save the sections for different approaches to the meta dill file.
            # this allows for easy "automatic" assimilation of differently generated section sequences.
            d, f1, f2 = dill_lock_and_load(meta_dill_fn, keep_locks=True)
            if wafer_ids[i] not in d['order_solving']: d['order_solving'][wafer_ids[i]] = {}
            bn = os.path.basename(bfnseq)
            d['order_solving'][wafer_ids[i]][bn] = {'sections':sections, 'excludes':exclude_regions[wafer_ids[i]]}
            dill_lock_and_dump(meta_dill_fn, d, f1, f2)
        print('Solved region order split into good matches sections, mapped into region strings')
        for j in range(len(sections)):
            with np.printoptions(formatter={'all':lambda x: '{:>3}'.format(x)}):
                print([region_strs[i][x] for x in sections[j]])
        print('Solved region order split into good matches sections region strings, first and last slices only')
        for j in range(len(sections)):
            with np.printoptions(formatter={'all':lambda x: '{:>3}'.format(x)}):
                print([region_strs[i][sections[j][0]], region_strs[i][sections[j][-1]]])
        #print('Utilized ROI polygon scale indices:')
        #print(rough_affine_skip_dicts[i][ind]['roi_polygon_scale_index'])
    #for i in range(nwafer_ids):
    np.set_printoptions(threshold=tmp)

elif rough_reconcile:
    # this is so that the solved affine can be applied on top of an existing rough affine.
    if rerough:
        aggregator.wafers_imaged_order_rough_affines = [None]*nwafer_ids
        for i in range(nwafer_ids):
            aggregator.wafers_imaged_order_rough_affines[i] = rough_dicts[i]['imaged_order_affines']

    if rough_reconcile_merge:
        # these are not needed after the original merge
        aggregator.cum_deltas = aggregator.cum_deltas_inds = aggregator.all_pts_src = None
        aggregator.grid_nvertices = 0
        # init the other member variables that will be stored
        aggregator.init_refit_affines_rough_alignments()

        print('Merging merged (second merge) rough dills'); t = time.time()
        merge_nproc = nprocesses_all[1] if len(nprocesses_all) > 1 else nprocesses
        for j in range(merge_nproc):
            proc_str = '.merge' + str(j)
            for i in range(nwafer_ids):
                with open(rough_dill_fns[i]+proc_str, 'rb') as f: d = dill.load(f)
                for k, caffine in enumerate(d['imaged_order_affines']):
                    if caffine is not None:
                        aggregator.wafers_imaged_order_rough_affines[i][k] = caffine
                with open(rough_rigid_dill_fns[i]+proc_str, 'rb') as f: d = dill.load(f)
                for k, caffine in enumerate(d['imaged_order_affines']):
                    if caffine is not None:
                        aggregator.wafers_imaged_order_rough_rigid_affines[i][k] = caffine

            with open(agg_dill_fn+proc_str, 'rb') as f: d = dill.load(f)
            for i, caffine in enumerate(d['cum_affines']):
                if caffine is not None:
                    aggregator.cum_affines[i] = caffine
            with open(agg_rigid_dill_fn+proc_str, 'rb') as f: d = dill.load(f)
            for i, caffine in enumerate(d['cum_affines']):
                if caffine is not None:
                    aggregator.cum_rigid_affines[i] = caffine
    else: # if rough_reconcile_merge:
        if run_merge:
            print('Merging rough dills'); t = time.time()
            niter = nprocesses if single_block else aggregator.ntblocks
            for j in range(niter):
                proc_str = '.' + str(j)
                print(agg_dill_fn+proc_str)
                with open(agg_dill_fn+proc_str, 'rb') as f: d = dill.load(f)
                if j==0:
                    cum_deltas = np.zeros((aggregator.total_nimgs,d['nvoronoi_vertices'],2), dtype=np.double)
                    #sanity_check = np.zeros((d['nvoronoi_vertices'],), dtype=np.int64)
                cum_deltas[:,d['cum_deltas_inds'],:] = d['cum_deltas']
                #sanity_check[d['cum_deltas_inds']] += 1
            #assert( (sanity_check > 0).all() )
            aggregator.cum_deltas = cum_deltas
            aggregator.grid_nvertices = d['nvoronoi_vertices']
            aggregator.cum_deltas_inds = None # this was only needed for the merge
            # xxx - this is saved in every process dill, slightly wasteful
            #   is is utilized by refit_affines_rough_alignments.
            #   another option could be to write a separate init_rough (like for fine reconciler).
            aggregator.all_pts_src = d['all_pts_src']

            # can only convert back to the points from voronoi points after they are all merged because the points
            #   are recovered by interpolating corresponding voronoi vectors for each grid point using idw.
            aggregator.cum_deltas = aggregator.voronoi_vectors_to_point_vectors()
            print('\tdone in %.4f s' % (time.time() - t, ))

            # also support parallelization of this affine refits (basically parallelize the merge process)
            merge_nproc = nprocesses_all[1] if len(nprocesses_all) > 1 else 1
            inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), merge_nproc)
            rngs = [[x[0],x[-1]+1] for x in inds]

            aggregator.refit_affines_rough_alignments(rough_distance_cutoff_pixels, img_range=rngs[iprocess])
            # save memory, these are no longer needed
            aggregator.cum_deltas = None
            aggregator.all_pts_src = None
        else: # if run_merge:
            aggregator.reconcile_rough_alignments(rough_distance_cutoff_pixels,
                L1_norm=reconcile_L1_norm, L2_norm=reconcile_L2_norm, regr_bias=rough_regression_remove_bias,
                neighbor_dist_scale=rough_neighbor_dist_scale, neighbors2D_radius_pixels=rough_smoothing_radius_pixels,
                neighbors2D_W=rough_smoothing_weight, neighbors2D_std_pixels=rough_smoothing_std_pixels,
                neighbors2D_expected=rough_smoothing_neighbors, z_neighbors_radius=z_neighbors_radius,
                nworkers=arg_nworkers, iprocess=iprocess, nprocesses=nprocesses)
            if not single_proc:
                # init the other member variables that will be stored
                aggregator.init_refit_affines_rough_alignments()
    #else: if rough_reconcile_merge:

    if single_proc or run_merge or rough_reconcile_merge:
        if not run_merge or merge_nproc==1 or rough_reconcile_merge:
            proc_str = ''
        else:
            proc_str = '.merge' + str(iprocess)
        for i in range(nwafer_ids):
            d = {
                 'solved_order':solved_orders[i],
                 'nregions':wafers_nregions[i],
                 'imaged_order_affines':aggregator.wafers_imaged_order_rough_affines[i],
                }
            with open(rough_dill_fns[i]+proc_str, 'wb') as f: dill.dump(d, f)

            # also save a version where the reconciled deltas were fit with a rigid transform.
            d['imaged_order_affines'] = aggregator.wafers_imaged_order_rough_rigid_affines[i]
            with open(rough_rigid_dill_fns[i]+proc_str, 'wb') as f: dill.dump(d, f)

    # these are used for plotting / debug and also for re-running accumulation subset.
    # also for supporting multiple processes.
    if single_proc or run_merge or rough_reconcile_merge:
        if not run_merge or merge_nproc==1 or rough_reconcile_merge:
            proc_str = ''
        else:
            proc_str = '.merge' + str(iprocess)
    else:
        proc_str = '.' + (str(iprocess) if single_block else str(aggregator.itblock))
    d = {
         'wafer_ids':wafer_ids,
         'wafers_template_order':np.nan, # throwback to accumulator, remove
         'wafers_nimgs':aggregator.wafers_nimgs,
         'cum_wafers_nimgs':aggregator.cum_wafers_nimgs,
         'order_rng':aggregator.order_rng,
         'cum_affines':aggregator.cum_affines,
         'total_nimgs':aggregator.total_nimgs,
         # to support multiple processes
         'cum_deltas':aggregator.cum_deltas,
         'cum_deltas_inds':aggregator.cum_deltas_inds,
         'nvoronoi_vertices':aggregator.grid_nvertices,
         'all_pts_src':aggregator.all_pts_src,
        }
    with open(agg_dill_fn+proc_str, 'wb') as f: dill.dump(d, f)

    # also save a version where the reconciled deltas were fit with a rigid transform.
    d['cum_affines'] = aggregator.cum_rigid_affines
    with open(agg_rigid_dill_fn+proc_str, 'wb') as f: dill.dump(d, f)

elif fine_outliers:
    inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
    rngs = [[x[0],x[-1]+1] for x in inds]
    print('Initializing for fine outliers from delta dills, range {}-{}'.format(rngs[iprocess][0],
        rngs[iprocess][1])); t = time.time()
    # special mode to initialize dill files for block processing
    init_blocks = any([x < 0 for x in nblks])
    assert( not init_blocks or nprocesses==1 ) # do not init block dills with nprocesses > 1
    if init_blocks: nblks = [abs(nblks[0]), 1] # block dills are stored one per first dimension
    load_type = 'solved' if init_blocks else ('outliers-merge' if run_merge else 'outliers')
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, img_range=rngs[iprocess], run_str=run_str_in, nblocks=nblks, iblock=iblk,
        block_overlap_pix=blk_ovlp_pix, load_type=load_type)
    print('\tdone in %.4f s' % (time.time() - t, ))

    if run_merge:
        # NOTE: for this merge it is still ok for the it to parallelized by images, img_range
        print('Merging fine outlier blocks'); t = time.time()
        aggregator.update_fine(update_type='outliers-merge', merge_inliers_blk_cutoff=merge_inliers_blk_cutoff,
            merge_inliers_min_comp=merge_inliers_min_comp, merge_inliers_min_hole_comp=merge_inliers_min_hole_comp)
        print('\tdone in %.4f s' % (time.time() - t, ))
    else:
        if not init_blocks:
            aggregator.fine_deltas_outlier_detection(min_inliers=interp_inliers, nworkers=arg_nworkers,
                doplots=plot_deltas)
        if init_blocks:
            print('Initializing fine outliers temporary block dills'); t = time.time()
            if single_block:
                print('Single block, do nothing')
            else:
                aggregator.update_fine(update_type='block-init')
        else:
            print('Dumping fine aggregation outliers back to delta dills'); t = time.time()
            aggregator.update_fine(update_type='outliers')
        print('\tdone in %.4f s' % (time.time() - t, ))

elif fine_accumulate:
    # paralliztion for fine reconcile is more complicated than per image because all the images (z-direction)
    #   are all used for each grid point. the grid points are mostly independent, except to and from the
    #   voronoi points, so the parallization is done over the grid points instead of over the images.
    # this means DO NOT use img_range for parallelization. ok for running a subset of the stack tho.
    if fine_reslice:
        init_fine_reslice = any([x < 0 for x in nblks])
        inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
        rngs = [[x[0],x[-1]+1] for x in inds]
        if init_fine_reslice:
            print('Initializing for fine reslice init')
            # to init the h5 file for the deltas reslice, do not need to load all the dills
            img_range = [0,1]
            _nblocks=[abs(x) for x in nblks]; _iblock=iblk; _block_overlap_pix=blk_ovlp_pix
        else:
            # the fine delta reslice can be parallelized over the images (using img_range).
            print('Initializing for fine reslice, range {}-{}'.format(rngs[iprocess][0], rngs[iprocess][1]))
            img_range = rngs[iprocess]
            _nblocks=[1,1]; _iblock=[0,0]; _block_overlap_pix=[0.,0.]
    else:
        assert(nprocesses==1 or single_block) # did not see use case for multiple processes within blocks
        if (order_range < 0).any() or order_range[0] >= order_range[1]:
            img_range = None # accumulation can NOT be parallized by images in normal usage
            print('Initializing for fine reconcile from delta dills')
        else:
            # BUT, can use img_range to specify a subset of the whole stack to accumulate
            img_range = order_range
            print('Initializing for fine outliers from delta dills, range {}-{}'.format(img_range[0], img_range[1]))
        _nblocks=nblks; _iblock=iblk; _block_overlap_pix=blk_ovlp_pix
    t = time.time()
    if filtered_fine_deltas: print('\tusing filtered deltas')
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, load_type='solved' if (run_merge or use_fine_reslice) else 'reconcile', run_str=run_str_in,
        load_filtered_deltas=filtered_fine_deltas, load_interpolated_deltas=load_interpolated_deltas,
        img_range=img_range, nblocks=_nblocks, iblock=_iblock, block_overlap_pix=_block_overlap_pix,
        fine_interp_weight=fine_interp_weight, keep_xcorrs=keep_xcorrs, zero_deltas_indices=use_slice_blur_z_indices,
        fine_interp_neighbor_dist_scale=fine_interp_neighbor_dist_scale)
    print('\tdone in %.4f s' % (time.time() - t, ))

    if use_fine_reslice:
        assert(nprocesses==1) # this was implemented with the intention of using blocks
        print('Loading fine reslice for iblock {} {}'.format(_iblock[0], _iblock[1]))
        inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses_all[1])
        rngs = [[x[0],x[-1]+1] for x in inds]
        aggregator.fine_deltas_reslice_load(fine_reslice_fn, rngs)
        print('\tdone in %.4f s' % (time.time() - t, ))

    if fine_reslice:
        bfn, ext = os.path.splitext(fine_reslice_fn)
        fine_reslice_fn = bfn + '.{}'.format(iprocess) + ext
        t = time.time()
        if init_fine_reslice:
            print('Initializing fine deltas h5 files that can be block loaded, process {}'.format(iprocess))
            aggregator.fine_deltas_reslice_init(fine_reslice_fn, nworkers=arg_nworkers, keep_xcorrs=keep_xcorrs,
                    nprocesses=nprocesses)
        else:
            print('Creating fine deltas h5 files that can be block loaded, process {}'.format(iprocess))
            aggregator.fine_deltas_reslice(fine_reslice_fn, nblocks=nblks, block_overlap_pix=blk_ovlp_pix,
                keep_xcorrs=keep_xcorrs)

        print('\tdone in %.4f s' % (time.time() - t, ))
        print('JOB FINISHED: run_wafer_aggregator.py')
        print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
        sys.exit(0)

    if run_merge:
        print('Merging fine dills'); t = time.time()
        niter = nprocesses if single_block else aggregator.ntblocks
        #dt = time.time()
        for j in range(niter):
            proc_str = '.' + str(j)
            with open(agg_dill_fn+proc_str, 'rb') as f: d = dill.load(f)
            if j==0:
                cum_deltas = np.zeros((aggregator.total_nimgs,d['nvoronoi_vertices'],2), dtype=np.double)
                cum_comps_sel = np.zeros((aggregator.total_nimgs,d['nvoronoi_vertices']), dtype=bool)
                #sanity_check = np.zeros((d['nvoronoi_vertices'],), dtype=np.int64)
            cum_deltas[:,d['cum_deltas_inds'],:] = d['cum_deltas']
            cum_comps_sel[:,d['cum_deltas_inds']] = d['cum_comps_sel']
            #sanity_check[d['cum_deltas_inds']] += 1
            #print('\tdone in %.4f s' % (time.time() - dt, )); dt=time.time()
        #assert( (sanity_check > 0).all() )
        aggregator.cum_deltas = cum_deltas
        aggregator.cum_comps_sel = cum_comps_sel
        aggregator.grid_nvertices = d['nvoronoi_vertices']
        aggregator.cum_deltas_inds = None # this was only needed for the merge
        aggregator.all_fine_outliers = d['cum_outliers']
        # aggregator.all_fine_valid_comparisons = d['cum_valid_comparisons']
        aggregator.count_included = d['cum_included']

        # can only convert back to the points from voronoi points after they are all merged because the points
        #   are recovered by interpolating corresponding voronoi vectors for each grid point using idw.
        aggregator.cum_deltas = aggregator.voronoi_vectors_to_point_vectors()
        aggregator.cum_comps_sel = (aggregator.voronoi_vectors_to_point_vectors(\
                vrdeltas=aggregator.cum_comps_sel[:,:,None]) > 0.).any(2)
        print('\tdone in %.4f s' % (time.time() - t, ))
    else: # if run_merge:
        aggregator.reconcile_fine_alignments(L1_norm=reconcile_L1_norm, L2_norm=reconcile_L2_norm,
            min_valid_slice_comparisons=fine_min_valid_slice_comparisons, solve_reverse=fine_solve_reverse,
            neighbor_dist_scale=fine_neighbor_dist_scale, neighbors2D_radius_pixels=fine_smoothing_radius_pixels,
            neighbors2D_W=fine_smoothing_weight, neighbors2D_std_pixels=fine_smoothing_std_pixels,
            neighbors2D_expected=fine_smoothing_neighbors, regr_bias=fine_regression_remove_bias,
            z_neighbors_radius=z_neighbors_radius, nworkers=arg_nworkers, iprocess=iprocess, nprocesses=nprocesses)
    if filtered_fine_deltas and (single_proc or run_merge):
        aggregator.solved_deltas_affine_filter(fine_filtering_shape_pixels, nworkers=arg_nworkers)
    aggregator.waferize_reconcile_fine_alignments()

    if single_proc or run_merge:
        key_str = 'imaged_order_reverse_deformations' if fine_solve_reverse else 'imaged_order_forward_deformations'
        for i in range(nwafer_ids):
            d = {
                 'deformation_points':aggregator.grid_locations_pixels,
                 key_str : None,
                }
            with open(fine_dill_fns[i], 'wb') as f: dill.dump(d, f)

            # save the actual deltas to an h5 file for easy partial loading
            h5fn = fine_dill_fns[i] + '.h5'
            big_img_save(h5fn, aggregator.imaged_order_deltas[i], aggregator.imaged_order_deltas[i].shape,
                dataset='imaged_order_deltas', compression=True, recreate=True, truncate=True)

    # save results in solved order and as 'single' dill files.
    # used as part of the merge process and for aggregation plotting.
    cum_deltas_save = None if aggregator.cum_deltas.size >= 2**32 else aggregator.cum_deltas
    cum_comps_sel_save = None if aggregator.cum_comps_sel.size >= 2**32 else aggregator.cum_comps_sel
    d = {
         'wafer_ids':wafer_ids,
         'wafers_template_order':np.nan,
         'wafers_nimgs':aggregator.wafers_nimgs,
         'cum_wafers_nimgs':aggregator.cum_wafers_nimgs,
         'order_rng':aggregator.order_rng,
         'total_nimgs':aggregator.total_nimgs,
         'deformation_points':aggregator.grid_locations_pixels,
         'nvoronoi_vertices':aggregator.grid_nvertices,
         'cum_deltas':cum_deltas_save,
         'cum_comps_sel':cum_comps_sel_save,
         'cum_deltas_inds':aggregator.cum_deltas_inds,
         'cum_outliers':aggregator.all_fine_outliers,
         # 'cum_valid_comparisons':aggregator.all_fine_valid_comparisons,
         'cum_included':aggregator.count_included,
        }
    if single_proc or run_merge:
        proc_str = ''
    else:
        proc_str = '.' + (str(iprocess) if single_block else str(aggregator.itblock))
    with open(agg_dill_fn+proc_str, 'wb') as f: dill.dump(d, f)
    if cum_deltas_save is None:
        np.save(agg_dill_fn+proc_str+'_cum_deltas', aggregator.cum_deltas)
    # xxx - omitted > 4 GB cum_comps_sel save as this case is not necessary for current workflow

elif fine_interp:
    inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
    rngs = [[x[0],x[-1]+1] for x in inds]

    print('Initializing for fine interpolate from delta dills, range {}-{}'.format(rngs[iprocess][0],
        rngs[iprocess][1])); t = time.time()
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, img_range=rngs[iprocess], load_type='process_fine', run_str=run_str_in,
        nblocks=nblks, iblock=iblk, block_overlap_pix=blk_ovlp_pix)
    print('\tdone in %.4f s' % (time.time() - t, ))

    # xxx - add option for this, get the number of regular grid points from center point,
    #   with different distance estimates.
    # xxx - maybe just delete this because the grids are not spatially heterogeneous in xy
    #   instead estimate based on number of points within a particular hex grid diameter (like in mfovs).
    if False:
        interp_inlier_radius_um = 32.1
        inlier_radius = interp_inlier_radius_um*meta_dict['scale_um_to_pix']
        blk_center = aggregator.grid_locations_pixels[0,:]
        sel_rad = l2_and_delaunay_distance_select(aggregator.grid_locations_pixels, inlier_radius,
                aggregator.grid_locations_pixels[0,:], griddist_pixels, use_delaunay=False)
        inlier_radius_cutoff = sel_rad.sum()
        print('Computed inlier_radius_cutoff {} based on inlier_radius {} pixels'.\
            format(inlier_radius_cutoff, inlier_radius))
        sys.exit(0)

    print('Interpolating with method {}, n_neighbors {}'.format(aggregator.delta_interp_method,
        interp_inlier_nneighbors))
    aggregator.interpolate_fine_outliers(interp_inliers=interp_inliers, inlier_nneighbors=interp_inlier_nneighbors,
        nworkers=arg_nworkers, doplots=plot_deltas)
    # xxx - this did not work very well, keep for reference for now, maybe delete?
    #aggregator.fine_deltas_affine_filter(fine_filtering_shape_pixels, affine_degree=1, use_interp_points=False,
    #    affine_interpolation=True, output_features_scale=6., doplots=plot_deltas)
    print('Dumping fine interpolated deltas back to delta dills'); t = time.time()
    aggregator.update_fine(update_type='interpolated_deltas')
    print('\tdone in %.4f s' % (time.time() - t, ))

elif fine_filter:
    # xxx - expose this as a param?
    use_interp_points = True

    inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
    rngs = [[x[0],x[-1]+1] for x in inds]
    print('Initializing for fine filter affines from delta dills, range {}-{}'.format(rngs[iprocess][0],
        rngs[iprocess][1])); t = time.time()
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, img_range=rngs[iprocess], load_type='process_fine', run_str=run_str_in,
        load_interpolated_deltas=use_interp_points)
    print('\tdone in %.4f s' % (time.time() - t, ))

    aggregator.fine_deltas_affine_filter(fine_filtering_shape_pixels, affine_degree=1,
        use_interp_points=use_interp_points, doplots=plot_deltas)
    print('Dumping fine filtered deltas back to delta dills'); t = time.time()
    aggregator.update_fine(update_type='filtered_deltas')
    print('\tdone in %.4f s' % (time.time() - t, ))

elif fine_affine:
    assert(nprocesses==1) # did not implement the merge, see below
    inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
    rngs = [[x[0],x[-1]+1] for x in inds]
    print('Initializing for fine to rough alignment, range {}-{}'.format(rngs[iprocess][0],
        rngs[iprocess][1])); t = time.time()
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, img_range=rngs[iprocess], load_type='solved', run_str=run_str_in)
    print('\tdone in %.4f s' % (time.time() - t, ))

    aggregator.cum_deltas = agg_dict['cum_deltas']
    aggregator.cum_comps_sel = agg_dict['cum_comps_sel']
    # this is so that the solved affine can be applied on top of an existing rough affine.
    if rerough:
        aggregator.wafers_imaged_order_rough_affines = [None]*nwafer_ids
        for i in range(nwafer_ids):
            aggregator.wafers_imaged_order_rough_affines[i] = rough_dicts[i]['imaged_order_affines']

    if run_merge:
        print('Merging fine dills'); t = time.time()
        assert(False) # xxx - implement me
        # need a "None-based" merge, replaces None's in current merge dict with not Nones in current proc load
    else:
        aggregator.fine_deltas_to_rough_affines()

    for i in range(nwafer_ids):
        d = {
             'solved_order':solved_orders[i],
             'nregions':wafers_nregions[i],
             'imaged_order_affines':aggregator.wafers_imaged_order_rough_affines[i],
            }
        with open(rough_dill_out_fns[i], 'wb') as f: dill.dump(d, f)
    # these are used for plotting / debug and also for re-running accumulation subset.
    d = {
         'wafer_ids':wafer_ids,
         'wafers_template_order':np.nan,
         'wafers_nimgs':aggregator.wafers_nimgs,
         'cum_wafers_nimgs':aggregator.cum_wafers_nimgs,
         'order_rng':aggregator.order_rng,
         'cum_affines':aggregator.cum_affines,
         'total_nimgs':aggregator.total_nimgs,
        }
    with open(agg_dill_fn, 'wb') as f: dill.dump(d, f)

elif fine_affine_raw:
    assert(nprocesses==1) # did not implement the merge, see below
    inds = np.array_split(np.arange(aggregator.order_rng[0],aggregator.order_rng[1]), nprocesses)
    rngs = [[x[0],x[-1]+1] for x in inds]
    print('Initializing for fine to rough raw affines from delta dills, range {}-{}'.format(rngs[iprocess][0],
        rngs[iprocess][1])); t = time.time()
    aggregator.init_fine(grid_locations_pixels, alignment_folders, delta_dill_fn_str, range_skips, use_crops_um,
        griddist_pixels, img_range=rngs[iprocess], load_type='process_fine', run_str=run_str_in)
    print('\tdone in %.4f s' % (time.time() - t, ))

    if run_merge:
        print('Merging fine dills'); t = time.time()
        assert(False) # xxx - implement me
        # need a "None-based" merge, replaces None's in current merge dict with not Nones in current proc load
    else:
        aggregator.fine_deltas_to_rough_deltas()

    for i in range(nwafer_ids):
        for k in range(max_range_skips):
            d = {'solved_order':solved_orders[i], 'nregions':wafers_nregions[i],
                 'forward_affines':aggregator.forward_affines[i][k],
                 'reverse_affines':aggregator.reverse_affines[i][k],
                 'forward_pts_src':aggregator.forward_pts_src[i][k],
                 'forward_pts_dst':aggregator.forward_pts_dst[i][k],
                 'reverse_pts_src':aggregator.reverse_pts_src[i][k],
                 'reverse_pts_dst':aggregator.reverse_pts_dst[i][k],
                 }
            proc_str = ('.' + str(iprocess)) if nprocesses > 1 and not run_merge else ''
            with open(rough_out_affine_skip_dill_fns[i][k]+proc_str, 'wb') as f: dill.dump(d, f)

            d = {'solved_order':solved_orders[i], 'nregions':wafers_nregions[i],
                 'rforward_affines':aggregator.forward_affines[i][k],
                 'rreverse_affines':aggregator.reverse_affines[i][k],
                 'forward_pts_src':aggregator.forward_pts_src[i][k],
                 'forward_pts_dst':aggregator.forward_pts_dst[i][k],
                 'reverse_pts_src':aggregator.reverse_pts_src[i][k],
                 'reverse_pts_dst':aggregator.reverse_pts_dst[i][k],
                 }
            proc_str = ('.' + str(iprocess)) if nprocesses > 1 and not run_merge else ''
            with open(rough_out_rigid_affine_skip_dill_fns[i][k]+proc_str, 'wb') as f: dill.dump(d, f)
        #for k in range(max_range_skips):
    #for i in range(nwafer_ids):

# run_type switch

print('JOB FINISHED: run_wafer_aggregator.py')
print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
