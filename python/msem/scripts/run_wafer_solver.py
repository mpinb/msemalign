#!/usr/bin/env python3
"""run_wafer_solver.py

Top level command-line interface for the wafer ordering solving and for
  computing the affine transformations of the rough alignment.

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
import os
import sys
import argparse
import time

from msem import wafer, wafer_solver
from msem.utils import make_hex_points, dill_lock_and_load, dill_lock_and_dump

# all parameters loaded from an experiment-specific import
from def_common_params import get_paths, dsstep, use_thumbnails_ds, dsthumbnail, exclude_regions, region_manifest_cnts
from def_common_params import scale_nm, nimages_per_mfov, legacy_zen_format
from def_common_params import keypoints_dill_fn_str, matches_dill_fn_str, rough_affine_dill_fn_str
from def_common_params import order_txt_fn_str, exclude_txt_fn_str, limi_dill_fn_str
from def_common_params import thumbnail_subfolders, thumbnail_subfolders_order, debug_plots_subfolder
from def_common_params import region_suffix, thumbnail_suffix
from def_common_params import lowe_ratio, nfeatures, max_npts_feature_correspondence
from def_common_params import affine_rigid_type, min_feature_matches, roi_polygon_scales, matches_iroi_polygon_scales
from def_common_params import rough_residual_threshold_um, min_fit_pts_radial_std_um, max_fit_translation_um
from def_common_params import rough_bounding_box_xy_spc, rough_grid_xy_spc
from def_common_params import wafer_solver_bbox_xy_spc, wafer_solver_bbox_trans
from def_common_params import keypoints_nworkers_per_process, keypoints_nprocesses, matches_full, matches_gpus
from def_common_params import keypoints_filter_size, keypoints_rescale
from def_common_params import tissue_mask_path, tissue_mask_fn_str, tissue_mask_ds
from def_common_params import tissue_mask_min_edge_um, tissue_mask_min_hole_edge_um, tissue_mask_bwdist_um


## argparse

parser = argparse.ArgumentParser(description='run_wafer_solver.py')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
    help='wafer to run solver for OR two neighboring wafers to run cross-wafer rough alignment')
parser.add_argument('--affine-rerun', dest='affine_rerun', action='store_true',
    help='re-run the affine fits in the previously solved order')
parser.add_argument('--run-str', nargs=1, type=str, default=['rigid'],
    help='string to differentiate rough alignments with different parameters')
parser.add_argument('--no-exclude-regions', dest='no_exclude_regions', action='store_true',
    help='ignore the exclude regions stored in def_common_params')
parser.add_argument('--matches-exclude-regions', dest='matches_exclude_regions', action='store_true',
    help='for --matches-only, only run matches for the exclude regions')
parser.add_argument('--keypoints-run-str', nargs=1, type=str, default=['none'],
    help='string to differentiate keypoints dill files with different parameters')
parser.add_argument('--matches-run-str', nargs=1, type=str, default=['none'],
    help='string to differentiate matches dill files with different parameters')
parser.add_argument('--thumbs-run-str', nargs=1, type=str, default=[''],
    help='string to differentiate different thumbnails for order solving')
parser.add_argument('--skip-slices', dest='skip_slices', nargs=1, type=int, default=[0],
    help='how many slices beyond the next slice to template match to')
parser.add_argument('--order-beg', nargs=1, type=int, default=[1],
    help='region order index to start with (base 1)')
parser.add_argument('--order-end', nargs=1, type=int, default=[0],
    help='region order index to stop on (base 1)')
parser.add_argument('--keypoints-only', dest='keypoints_only', action='store_true',
    help='only calculate keypoints and then exit after dill saved')
parser.add_argument('--matches-only', dest='matches_only', action='store_true',
    help='only compute percent matches and then exit after dill saved')
parser.add_argument('--disable-roi-polygons', dest='disable_roi_polygons', action='store_true',
    help='disable roi polygon scaling, i.e., for order solving manually set regions')
parser.add_argument('--manual-solve-order', dest='manual_solve_order', action='store_true',
    help='special mode to use "manually" set regions for order solving only')
parser.add_argument('--show-keypoints-matches', dest='show_keypoints_matches', action='store_true',
    help='diplay plots of the keypoints matches sequentially')
parser.add_argument('--save-keypoints-matches', dest='save_keypoints_matches', action='store_true',
    help='save plots of the keypoints matches sequentially')
parser.add_argument('--save-keypoints-overlays', dest='save_keypoints_overlays', action='store_true',
    help='save images with keypoints overlaid on them')
parser.add_argument('--iroi_polygon_scale', nargs=1, type=int, default=[-1],
    help='use only a single roi polygon scale for computing wafer matches')
parser.add_argument('--iprocess', nargs=1, type=int, default=[0],
    help='process index for jobs that are divided into multiple processes')
parser.add_argument('--nprocesses', nargs=1, type=int, default=[1],
    help='split some operations into multiple processes')
parser.add_argument('--nworkers', nargs=1, type=int, default=[1],
    help='split some operations into multiple threads (same-node processes)')
parser.add_argument('--load-all-keypoints', dest='load_all_keypoints', action='store_true',
    help='always load all the keypoints instead of loading incrementally (to save memory)')
parser.add_argument('--merge', dest='merge', action='store_true',
    help='merge results from multiple process runs into single dill (not keypoints)')
parser.add_argument('--write-order', dest='write_order', action='store_true',
    help='when solving order only, write out the solved order')
parser.add_argument('--tissue-masks', dest='tissue_masks', action='store_true',
    help='use the tissue masks')
parser.add_argument('--matches-ransac', dest='matches_ransac', action='store_true',
    help='compute percent matches by including ransac on matching points')
parser.add_argument('--percent-matches-topn', nargs=1, type=int, default=[0],
    help='preprocess percent matches for solver by only keeping top n')
parser.add_argument('--percent-matches-normalize', dest='percent_matches_normalize', action='store_true',
    help='normalize percent matches for solver with row means/stds')
parser.add_argument('--percent-matches-normalize-minmax', dest='percent_matches_normalize_minmax',
    action='store_true',
    help='normalize percent matches for solver with row means/stds')
parser.add_argument('--random-exclude-perc', nargs=1, type=float, default=[0.],
    help='a sensitivity test for the order solving method')
# make ransac iterations easily controlable from command line
# because of heuristics in wafer_solver ransac, makes sense for default to be a few repeats
parser.add_argument('--ransac-repeats', nargs=1, type=int, default=[5],
                    help='if ransac is run (fine outliers), parallized by workers')
parser.add_argument('--ransac-max', nargs=1, type=int, default=[1000],
                    help='if ransac is run (fine outliers)')

args = parser.parse_args()
args = vars(args)


## params that are set by command line arguments

wafer_ids = args['wafer_ids']
nwafer_ids = len(wafer_ids)
assert(nwafer_ids < 3) # more than one wafer only meant for cross-wafer rough alignment
skip_slices = args['skip_slices'][0]
# this is the actual value of n for the i+n slice comparisons (skip_slices is n-1)
skip_nslices = skip_slices+1
# # xxx - this is really unideal, but the alternative was really messy
# # for the cross-wafer skips, specify which ending index from the first wafer this is for, run separately
# #   value of -1 means the last index, -2 second to last, etc...
# skip_slices_cross_wafer_ind = args['skip_slices_cross_wafer_ind'][0]
# assert(skip_slices_cross_wafer_ind < 0) # negative values, where -1 is last index, -2 second to last, etc
# assert(-skip_slices_cross_wafer_ind <= skip_nslices) # skip ind bigger than number of skips
# for cross-wafer alignment or skip slices, force the affine_rerun (uses previously saved keypoints)
affine_rerun = args['affine_rerun'] or nwafer_ids > 1 or skip_slices > 0
compute_matches = args['matches_only']

# for jobs with nprocesses > 1, run this mode to merge the dills.
# use the same value for nprocesses on command line.
run_merge = args['merge']

# flags to determine whether previously processed items should be loaded
load_solved_order = affine_rerun
load_keypoints_dill = not args['keypoints_only']
load_matches_dill = not affine_rerun and not run_merge

# this is an identifier so that multiple rough/fine alignemnts can be exported / loaded easily.
run_str = args['run_str'][0]

# string to differentiate different keypoint and matches runs also so they can be exported / loaded easily.
keypoints_run_str = args['keypoints_run_str'][0]
matches_run_str = args['matches_run_str'][0]

# this is an identifier so that multiple thumbnails can be exported / loaded easily.
# this can be useful during the order solving for regions generated with different
#   contrast / brightness balancing, for example.
thumbs_run_str = args['thumbs_run_str'][0]

# optionally specify starting and stopping indices
order_beg = args['order_beg'][0]-1
order_end = args['order_end'][0]-1

# this disables using the scaled roi polygongs for keypoint rejection.
disable_roi_polygons = args['disable_roi_polygons']

# this sets all flags associated with using manual rough bounding box exports
#   and no roi polygon in order to basically manually avoid bad areas while solving order.
manual_solve_order = args['manual_solve_order']

# use a specific roi polygon scale for compute wafer alignments.
# specify zero to iterate the defined scales (default).
iroi_polygon_scale = args['iroi_polygon_scale'][0]

# show debug plots of the keypoints matches
show_keypoints_matches = args['show_keypoints_matches']

# save debug plots of the keypoints matches
save_keypoints_matches = args['save_keypoints_matches']

# save plots of thumbnails with keypoints overlaid on them
save_keypoints_overlays = args['save_keypoints_overlays']

# number of total processes for parallizing some of the run types.
nprocesses = args['nprocesses'][0]

# for specifying the index of this matches process
iprocess = args['iprocess'][0]

# for same-node process parallizations
arg_nworkers = args['nworkers'][0]

# the default is the rather unideal implementation of only loading keypoints as they are needed.
# this is required if all the keypoints for a wafer can not be loaded into memory at once.
# specify true for this to use the legacy mode that loads all the keypoints for the wafer for every process.
load_all_keypoints = args['load_all_keypoints']

# for solving the order only with the TSP solver, optionally write the order out.
write_order = args['write_order']

tissue_masks = args['tissue_masks']

# number of (parallized) ransac repeats and iterations per repeat
# NOTE: for robin rough rigid used 50 and 200000 respectively (used to be set above init in wafer_solver)
#   ransac_fail_repeats was still utilized at that point also, with a value of 5
ransac_repeats = args['ransac_repeats'][0]
ransac_max = args['ransac_max'][0]

# purposefully ignore the exclude regions
no_exclude_regions = args['no_exclude_regions']

# only run matches for the current defined exclude regions
matches_exclude_regions = args['matches_exclude_regions']

# whether to compute percent matches that includes ransac
matches_ransac = args['matches_ransac']

# preprocess percent matches before order solved to only use top n
percent_matches_topn = args['percent_matches_topn'][0]

# other options for preprocessing percent matches before order solving.
percent_matches_normalize = args['percent_matches_normalize']
percent_matches_normalize_minmax = args['percent_matches_normalize_minmax']

# a sensitivity test for the order solving method, introduce a random percentage of "removed slices".
# 0.0 deafult normal setting is normal functionality (no test).
random_exclude_perc = args['random_exclude_perc'][0]

## fixed parameters not exposed in def_common_params

# subset of regions to try to run rough alignment with.
# NOTE: pretty much always use None to specify all regions, compiling a subset might not work very well.
#   this was mostly intended as a debug flag.
# also this is set below for cross-wafer for rough alignments.
region_inds = None


## parameters that are determined based on above parameters

experiment_folders_all = [None]*nwafer_ids
thumbnail_folders = [None]*nwafer_ids
alignment_folders = [None]*nwafer_ids
region_strs_all = [None]*nwafer_ids
protocol_folders_all = [None]*nwafer_ids
for i,j in zip(wafer_ids, range(nwafer_ids)):
    experiment_folders_all[j], thumbnail_folders[j], protocol_folders_all[j], alignment_folders[j], meta_folder, \
        region_strs_all[j] = get_paths(i)

# these are the output dills, are only ever for the first wafer (cross wafer alignment saved in the first wafer dill)
rough_affine_dill_fn = \
    os.path.join(alignment_folders[0], rough_affine_dill_fn_str.format(wafer_ids[0], skip_slices, run_str))

keypoints_dill_fns = [[None]*keypoints_nprocesses for x in range(nwafer_ids)]
matches_dill_fns = [None]*nwafer_ids
order_txt_fns = [None]*nwafer_ids; exclude_txt_fns = [None]*nwafer_ids
limi_dill_fns = [None]*nwafer_ids; limi_dicts = [None]*nwafer_ids
for i in range(nwafer_ids):
    for j in range(keypoints_nprocesses):
        keypoints_dill_fns[i][j] = os.path.join(alignment_folders[i], keypoints_dill_fn_str.format(wafer_ids[i],
                j, keypoints_run_str))
    matches_dill_fns[i] = os.path.join(alignment_folders[i], matches_dill_fn_str.format(wafer_ids[i],
            matches_run_str))
    limi_dill_fns[i] = os.path.join(alignment_folders[i], limi_dill_fn_str.format(wafer_ids[i]))
    order_txt_fns[i] = os.path.join(alignment_folders[i], order_txt_fn_str.format(wafer_ids[i]))
    exclude_txt_fns[i] = os.path.join(alignment_folders[i], exclude_txt_fn_str.format(wafer_ids[i]))

    # open any previously saved per wafer dicts
    with open(limi_dill_fns[i], 'rb') as f: limi_dicts[i] = dill.load(f)

# where to save optional solver correspondence plots
plots_folder = os.path.join(meta_folder, debug_plots_subfolder)

matches_include_regions = None
if exclude_regions is not None:
    if matches_exclude_regions:
        matches_include_regions = exclude_regions[wafer_ids[0]]

    #if nwafer_ids > 1 or not load_solved_order:
    if nwafer_ids > 1 or not load_keypoints_dill or no_exclude_regions or compute_matches:
        # some run types have problems when exclude regions is set
        exclude_regions = None
    else:
        exclude_regions = exclude_regions[wafer_ids[0]]

# set flags associated with special mode for "manually" set rough bounding box per wafer.
# this can help with order solving for problem wafers.
if manual_solve_order:
    bbox = wafer_solver_bbox_xy_spc[wafer_ids[0]]
    trans = wafer_solver_bbox_trans[wafer_ids[0]]
    if bbox is not None:
        rough_bounding_box = make_hex_points(*bbox, trans=trans, bbox=True)
        grid_locations = make_hex_points(*bbox, trans=trans)
    else:
        rough_bounding_box = make_hex_points(*rough_bounding_box_xy_spc, bbox=True)
        grid_locations = make_hex_points(*rough_grid_xy_spc)
    use_thumbnail_subfolders = thumbnail_subfolders_order
else:
    rough_bounding_box = make_hex_points(*rough_bounding_box_xy_spc, bbox=True)
    grid_locations = make_hex_points(*rough_grid_xy_spc)
    use_thumbnail_subfolders = thumbnail_subfolders
    if thumbs_run_str: use_thumbnail_subfolders = [x + '-' + thumbs_run_str for x in use_thumbnail_subfolders]

# make the disabling of roi polygons work independently of manual_solve_order mode
if disable_roi_polygons:
    use_roi_polygon_scales = [0.]
    use_matches_iroi_polygon_scale = 0
else:
    use_roi_polygon_scales = roi_polygon_scales
    use_matches_iroi_polygon_scale = matches_iroi_polygon_scales[wafer_ids[0]]


# the loop is only active for the cross wafer alignments, loop over all the possible skip combinations.
if nwafer_ids > 1:
    skip_slices_cross_wafer_inds = range(-skip_nslices, 0)
else:
    skip_slices_cross_wafer_inds = [-1] # the default for within wafer skip_slices

# load_all_keypoints is legacy mode that always loads all the keypoints for each wafer.
use_keypoints_dill_fns = None if load_all_keypoints else keypoints_dill_fns

# # support for reading the tissue masks (to filter the keypoints)
# if tissue_masks:
#     assert(tissue_mask_path is not None)
# else:
#     tissue_mask_path = None


for skip_slices_cross_wafer_ind in skip_slices_cross_wafer_inds:
    solved_order = None
    backload_roi_polys = None
    if nwafer_ids > 1:
        # this is for cross-wafer rough alignment
        solved_order1 = np.fromfile(order_txt_fns[0], dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        solved_order2 = np.fromfile(order_txt_fns[1], dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        region_inds = np.array([solved_order1[skip_slices_cross_wafer_ind],
                               solved_order2[skip_slices_cross_wafer_ind+skip_nslices]], dtype=np.int32) + 1
        solved_order = [0,1]
        backload_roi_polys = [limi_dicts[0]['imaged_order_region_recon_roi_poly_raw'][region_inds[0]-1],
                limi_dicts[1]['imaged_order_region_recon_roi_poly_raw'][region_inds[1]-1]]
    else:
        # region order is not solved yet or loaded with the mat order below for affine rerun
        solved_order = None
        backload_roi_polys = limi_dicts[0]['imaged_order_region_recon_roi_poly_raw']



    cwafer = wafer(experiment_folders_all, protocol_folders_all, alignment_folders, region_strs_all,
            scale_nm=scale_nm, wafer_ids=wafer_ids, solved_order=solved_order, region_suffix=region_suffix,
            use_thumbnails_ds=use_thumbnails_ds, grid_locations=grid_locations, init_region_coords=False,
            rough_bounding_box=rough_bounding_box, dsstep=dsstep, exclude_regions=exclude_regions,
            thumbnail_folders=thumbnail_folders, region_inds=region_inds, backload_roi_polys=backload_roi_polys,
            legacy_zen_format=legacy_zen_format, nimages_per_mfov=nimages_per_mfov,
            use_tissue_masks=tissue_masks, tissue_mask_path=tissue_mask_path, tissue_mask_ds=tissue_mask_ds,
            tissue_mask_fn_str=tissue_mask_fn_str, tissue_mask_min_edge_um=tissue_mask_min_edge_um,
            tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um, tissue_mask_bwdist_um=tissue_mask_bwdist_um,
            region_manifest_cnts=region_manifest_cnts, verbose=True)

    solver = wafer_solver(cwafer, lowe_ratio=lowe_ratio, min_feature_matches=min_feature_matches,
            min_fit_pts_radial_std_um=min_fit_pts_radial_std_um, thumbnail_subfolders=use_thumbnail_subfolders,
            thumbnail_suffix=thumbnail_suffix, solved_order=solved_order, rigid_type=affine_rigid_type,
            max_npts_feature_correspondence=max_npts_feature_correspondence,
            max_fit_translation_um=max_fit_translation_um, roi_polygon_scales=use_roi_polygon_scales,
            dsthumbnail=dsthumbnail, residual_threshold_um=rough_residual_threshold_um,
            keypoints_nworkers_per_process=keypoints_nworkers_per_process,
            keypoints_nprocesses=keypoints_nprocesses, keypoints_dill_fns=use_keypoints_dill_fns,
            ransac_repeats=ransac_repeats, ransac_max=ransac_max, verbose=True)



    if load_keypoints_dill:
        if load_all_keypoints:
            use_inds = None if nwafer_ids == 1 else region_inds-1
            solver.load_keypoints_process_dills(keypoints_dill_fns=keypoints_dill_fns, inds=use_inds)
    else:
        # changed this code path so that all that happens is calculating keypoints.
        # needed to separate out workflow because over about 500 slices / wafer
        #   the entire workflow is no longer runable at once on machine with 64G memory.
        solver.compute_wafer_keypoints(nfeatures, nthreads_per_job=arg_nworkers, iprocess=iprocess,
            filter_size=keypoints_filter_size, rescale=keypoints_rescale)
        solver.wafer_images = [None]*solver.wafer_nimages
        d = {'wafer_descriptors':solver.wafer_descriptors,
             'wafer_pickleable_keypoints':solver.wafer_pickleable_keypoints,
             'wafer_processed_keypoints':solver.wafer_processed_keypoints,
             }
        print('Saving keypoints dill'); t = time.time()
        with open(keypoints_dill_fns[0][iprocess], 'wb') as f: dill.dump(d, f)
        print('\tdone in %.4f s' % (time.time() - t, ))
        print('JOB FINISHED: computed / saved keypoints only')
        print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
        sys.exit()

    if compute_matches:
        if run_merge:
            print('Merging matches dills'); t = time.time()
            for i in range(nprocesses):
                with open(matches_dill_fns[0] + '.' + str(i), 'rb') as f: d = dill.load(f)
                if i==0:
                    solver.percent_matches = d['percent_matches']
                else:
                    solver.percent_matches = solver.percent_matches + d['percent_matches']
                del d

            d = {'percent_matches':solver.percent_matches,
                 }
            with open(matches_dill_fns[0], 'wb') as f: dill.dump(d, f)
        else:
            # changed this code path so that all that happens is calculating percent matches.
            solver.compute_wafer_matches(iroi_polygon_scale=use_matches_iroi_polygon_scale, full=matches_full,
                gpus=matches_gpus, njobs_per_gpu=arg_nworkers, nprocesses=nprocesses, iprocess=iprocess,
                run_ransac=matches_ransac, include_regions=matches_include_regions)
            d = {'percent_matches':solver.percent_matches,
                 }
            with open(matches_dill_fns[0] + '.' + str(iprocess), 'wb') as f: dill.dump(d, f)

        print('JOB FINISHED: computed / saved or merged matches only')
        print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
        sys.exit()

    if load_matches_dill and nwafer_ids == 1:
        with open(matches_dill_fns[0], 'rb') as f: d = dill.load(f)
        solver.percent_matches = d['percent_matches']

    # this is here so that affines can be re-run after the ordering is manually corrected at the bad_matches spots.
    if load_solved_order:
        loaded_solved_order = np.fromfile(order_txt_fns[0], dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        if nwafer_ids == 1:
            solved_order = loaded_solved_order
        else:
            # nothing is done here purposely, solved_order is set above for cross-wafer alignment
            pass

    solved_order_mask = None
    if skip_slices > 0:
        solved_order_reorder = [None]*skip_nslices
        skip_lens = np.zeros((skip_nslices,), dtype=np.int32)
        reorder_skip_slices_cross_wafer_ind = None
        # this could be done with some modulo arithmetic also, did not feel like calculating it / debugging it
        reverse_nregions_arange = np.arange(loaded_solved_order.size) - loaded_solved_order.size
        for i in range(skip_nslices):
            solved_order_reorder[i] = loaded_solved_order[i::skip_nslices]
            skip_lens[i] = solved_order_reorder[i].size
            if reverse_nregions_arange[i::skip_nslices][-1] == skip_slices_cross_wafer_ind:
                # for cross wafer skips, need to set the appropriate index depending on whether
                #   the last index in each segment matches the desired cross-wafer skip amount for this run.
                reorder_skip_slices_cross_wafer_ind = i
        cum_skip_lens = np.cumsum(skip_lens)
        if nwafer_ids == 1:
            solved_order = np.concatenate(tuple(solved_order_reorder))
            # this prevents computing alignments at the modulo boundaries for skip alignment with one wafer.
            # they are filled in instead with cross-wafer alignments for middle wafers.
            solved_order_mask = np.ones((solved_order.size,), dtype=bool)
            solved_order_mask[cum_skip_lens-1] = 0
        else:
            # nothing is done here purposely, solved_order is set above for cross-wafer alignment
            pass

    doplots = (show_keypoints_matches or save_keypoints_matches or save_keypoints_overlays)
    dosave_path = plots_folder if save_keypoints_matches or save_keypoints_overlays else ''
    is_nprocs = (order_beg==0 and order_end==-1 and nprocesses > 1 and nwafer_ids==1)
    if is_nprocs:
        assert( nwafer_ids == 1 ) # no multiple process support for cross wafer
        # the solved order returned from the solver is not entirely deterministic in the sense of:
        #   (1) the ordering can be forwards or backwards
        #   (2) if there are multiple possible optimal routes, the solver can fluctuate between them
        # so even if this is the initial order solving, run it once, save it to the solved order file
        #   temporarily so that all the processes use the same solved order.
        assert( run_merge or solved_order is not None ) # for initial order solving, save temp order first
        rngs = [[x[0],x[-1]+1] for x in np.array_split(np.arange(cwafer.nvalid_regions), nprocesses)]
        order_beg, order_end = rngs[iprocess][0], rngs[iprocess][1]
    if run_merge:
        assert( is_nprocs )
        print('Merging rough affine dills'); t = time.time()
        for i in range(nprocesses):
            fn = rough_affine_dill_fn + '.' + str(i)
            with open(fn, 'rb') as f: d = dill.load(f)
            if i==0:
                dmrg = d
            else:
                dmrg['forward_affines'][rngs[i][0]:rngs[i][1]] = d['forward_affines'][rngs[i][0]:rngs[i][1]]
                dmrg['reverse_affines'][rngs[i][0]:rngs[i][1]] = d['reverse_affines'][rngs[i][0]:rngs[i][1]]
                dmrg['forward_pts_src'][rngs[i][0]:rngs[i][1]] = d['forward_pts_src'][rngs[i][0]:rngs[i][1]]
                dmrg['forward_pts_dst'][rngs[i][0]:rngs[i][1]] = d['forward_pts_dst'][rngs[i][0]:rngs[i][1]]
                dmrg['reverse_pts_src'][rngs[i][0]:rngs[i][1]] = d['reverse_pts_src'][rngs[i][0]:rngs[i][1]]
                dmrg['reverse_pts_dst'][rngs[i][0]:rngs[i][1]] = d['reverse_pts_dst'][rngs[i][0]:rngs[i][1]]
                dmrg['roi_polygon_scale_index'][rngs[i][0]:rngs[i][1]] = \
                    d['roi_polygon_scale_index'][rngs[i][0]:rngs[i][1]]
                dmrg['solved_order_bad_matches'] = np.concatenate((dmrg['solved_order_bad_matches'],
                    d['solved_order_bad_matches']), axis=0)
                dmrg['solved_order_bad_matches_inds'] = np.concatenate((dmrg['solved_order_bad_matches_inds'],
                    d['solved_order_bad_matches_inds']), axis=0)
                dmrg['bad_matches_fail_types'] = np.concatenate((dmrg['bad_matches_fail_types'],
                    d['bad_matches_fail_types']), axis=0)
                dmrg['affine_percent_matches'][rngs[i][0]:rngs[i][1],:] = \
                    d['affine_percent_matches'][rngs[i][0]:rngs[i][1],:]
        with open(rough_affine_dill_fn, 'wb') as f: dill.dump(dmrg, f)
        print('\tdone in %.4f s' % (time.time() - t, ))
        print('JOB FINISHED: merged affine dills only')
        print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
        sys.exit()

    random_excludes = solver.compute_wafer_alignments(solved_order=solved_order, solved_order_mask=solved_order_mask,
            iroi_polygon_scale=iroi_polygon_scale, beg_ind=order_beg, end_ind=order_end, gpus=matches_gpus,
            iprocess=iprocess, nworkers=arg_nworkers, doplots=doplots, dosave_path=dosave_path,
            keypoints_overlays=save_keypoints_overlays, percent_matches_topn=percent_matches_topn,
            percent_matches_normalize=percent_matches_normalize, random_exclude_perc=random_exclude_perc,
            percent_matches_normalize_minmax=percent_matches_normalize_minmax)
    if solved_order is None:
        if write_order:
            # add one because solved_order is stored 1-based
            solved_order = solver.solved_order+1
            print('Writing out solved order')
            # remove exclude regions if they are loaded
            if exclude_regions is not None and len(exclude_regions) > 0:
                solved_order = solved_order[np.logical_not(np.in1d(solved_order, exclude_regions))]
            tmp = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=sys.maxsize)
            with open(order_txt_fns[0], "w") as text_file:
                text_file.write(' ' + np.array2string(solved_order, separator=' ',
                    formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
            with open(exclude_txt_fns[0], "w") as text_file:
                text_file.write(' ' + np.array2string(random_excludes, separator=' ',
                    formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
            np.set_printoptions(threshold=tmp)
            #solver.solved_order.tofile(order_txt_fns[0], sep=' ')

        print('JOB FINISHED: order solver only')
        print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
        sys.exit()

    if nwafer_ids == 1:
        d = {'solved_order':solver.solved_order, 'nregions':cwafer.nregions,
             'forward_affines':solver.forward_affines, 'reverse_affines':solver.reverse_affines,
             'forward_pts_src':solver.forward_pts_src, 'forward_pts_dst':solver.forward_pts_dst,
             'reverse_pts_src':solver.reverse_pts_src, 'reverse_pts_dst':solver.reverse_pts_dst,
             'solved_order_bad_matches':solver.solved_order_bad_matches,
             'solved_order_bad_matches_inds':solver.solved_order_bad_matches_inds,
             'bad_matches_fail_types':solver.bad_matches_fail_types,
             'roi_polygon_scale_index':solver.roi_polygon_scale_index,
             'affine_percent_matches':solver.affine_percent_matches,
             }
        fn = (rough_affine_dill_fn + '.' + str(iprocess)) if is_nprocs else rough_affine_dill_fn
        with open(fn, 'wb') as f: dill.dump(d, f)
    else:
        #with open(rough_affine_dill_fn, 'rb') as f: d = dill.load(f)
        d, f1, f2 = dill_lock_and_load(rough_affine_dill_fn, keep_locks=True)

        if solver.solved_order_bad_matches_inds.size > 0:
            print('Cross-wafer alignment failed, setting cross-wafer affines to None')
            print('Fail type %d' % (solver.bad_matches_fail_types[0][0],))
        else:
            print('Cross-wafer alignment success')
            print('Success at scale index %d' % (solver.roi_polygon_scale_index[0],))
        ind = cum_skip_lens[reorder_skip_slices_cross_wafer_ind]-1 if skip_slices > 0 else -1
        d['forward_affines'][ind] = solver.forward_affines[0]
        d['reverse_affines'][ind] = solver.reverse_affines[0]
        d['forward_pts_src'][ind] = solver.forward_pts_src[0]; d['forward_pts_dst'][ind] = solver.forward_pts_dst[0]
        d['reverse_pts_src'][ind] = solver.reverse_pts_src[0]; d['reverse_pts_dst'][ind] = solver.reverse_pts_dst[0]
        d['affine_percent_matches'][ind,:] = solver.affine_percent_matches[0,:]
        if len(d['solved_order']) != len(d['reverse_affines']):
            print('region order len, affines len:')
            print(len(d['solved_order']), len(d['reverse_affines']))
            assert(False) # this should not happen

        #with open(rough_affine_dill_fn, 'wb') as f: dill.dump(d, f)
        dill_lock_and_dump(rough_affine_dill_fn, d, f1, f2)

print('JOB FINISHED: run_wafer_solver.py end of line')
print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
