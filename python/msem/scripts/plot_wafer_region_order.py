#!/usr/bin/env python3
"""plot_wafer_region_order.py

Top level command-line interface for generating plots showing the ordering
  of the sections on top of their original locations on a wafer.

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

from matplotlib import pylab as plt
#import matplotlib.patches as patches

import numpy as np

import dill
import os
import argparse

from msem import wafer
from msem.utils import PolyCentroid, argsort
from msem.wafer_solver import wafer_solver
from aicspylibczimsem import CziMsem

from def_common_params import get_paths #, exclude_regions
from def_common_params import czifiles, czfiles, czipath, czifile_ribbons, czifile_scenes, czifile_use_roi_polygons
from def_common_params import order_txt_fn_str, rough_affine_dill_fn_str
from def_common_params import legacy_zen_format, czifile_rotations, region_rotations_all, nimages_per_mfov, scale_nm

## argparse

parser = argparse.ArgumentParser(description='plot_region_order_limi.py')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
                    help='single wafer id to plot region order limi view for')
parser.add_argument('--overlay-limi', dest='overlay_limi', action='store_true',
                    help='overlay the orders on the limi overview image')
parser.add_argument('--polygonal-center', dest='polygonal_center', action='store_true',
                    help='plot at region polygonal center instead of bounding box center')
parser.add_argument('--show-plots', dest='save_plots', action='store_false',
                    help='display plots instead of saving as png')
parser.add_argument('--show-order-lines', dest='show_order_lines', action='store_true',
                    help='show the lines connecting sequence in the order plots')
parser.add_argument('--show-region_strs', dest='show_region_strs', action='store_true',
                    help='show the actual region strings instead of the solved order')
parser.add_argument('--run-str-in', nargs=1, type=str, default=[''],
                    help='instead of loading solved order, specify rough alignemnt')
parser.add_argument('--use-rough-skip', nargs=1, type=int, default=[0],
                    help='if using rough alignment, which rough alignment skip to use')
parser.add_argument('--sort-type', nargs=1, type=str, default=['none'], choices=['none', 'zen', 'shortest'],
                    help='some other sorting to applied to the imaged order')
parser.add_argument('--zen-reimages', nargs='+', type=str, default=[],
                    help='for zen sort-type, parallel to wafer_ids, filenames that order reimages')
#parser.add_argument('--shortest-topn', nargs=1, type=int, default=[0],
#                    help='for shortest sort-type sparsify the distance matrix with topn columns')
parser.add_argument('--no-exclude-regions', dest='no_exclude_regions', action='store_true',
                    help='ignore the exclude regions stored in def_common_params')
args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

# wafer starting at 1, used only for specifying directories and filenames, does not map to any zeiss info.
#   in zeiss terminology, a single wafer is an experiment, however multiple wafers make up an actual tissue block.
wafer_ids = args['wafer_ids']
nwafer_ids = len(wafer_ids)

overlay_limi = args['overlay_limi']
polygonal_center = args['polygonal_center']

# whether to save or display plots
save_plots = args['save_plots']

# whether to show the lines connecting sequence in the order plots or not
show_order_lines = args['show_order_lines']

# whether to show the region strings instead of the order value
show_region_strs = args['show_region_strs']

# this is an identifier to specify rough alignment to use instead of saved solved order.
run_str_in = args['run_str_in'][0]

# if using rough aligment (run_str_in), which skip index to look at (must be less than max).
use_rough_skip = args['use_rough_skip'][0]

# sort the imaged order in some other way than the slice order from the manifest.
sort_type = args['sort_type'][0]

# this is for the legacy_zen_format support, optionally re-order the manifest by the initial integer
#   but only within each raw subfolder ("experiment folder" in Zen, when software/imaging is restarted).
zen_order_sort = sort_type == 'zen'
# re-order the manifest by a shortest path on the wafer through the slices
shortest_order_sort = sort_type == 'shortest'

# use along with zen_order_sort.
# reimages always comes in separate "experiment folders".
# this specifies files that will specify the original imaging order of a reimage.
zen_reimages = args['zen_reimages']

## use along with shortest_order_sort
## sparsify the distance matrix by preprocessing and taking only the top n columns
#shortest_topn = args['shortest_topn'][0]

# purposefully ignore the exclude regions
no_exclude_regions = args['no_exclude_regions']

## fixed parameters not exposed in def_common_params

# use the polygons for the slices instead of ROIs. will do automatically if there are no ROIs defined.
use_polygons=czifile_use_roi_polygons

# font size for the numbers indicating the order
#fontsize=5
#fontsize=8
fontsize=18

# how much to scale the xy lims for showing regions (without limi overlay)
#limit_scale = 1.0
limit_scale = 1.05

# alpha for the overlay scatter of the imaged / solved order
scatter_alpha = 1.0

# how big in inches to make the output figures
fig_size = [8,8]
#fig_size = [16,16]

# whether to use the colormap for the order also or not
use_cmap = True

# show first/last slice numbers only
first_last = True


## parameters that are determined based on above parameters

assert( len(zen_reimages) == 0 or len(wafer_ids) == len(zen_reimages) ) # must be parallel to wafer ids

for wafer_id,wafer_ind in zip(wafer_ids, range(nwafer_ids)):
#for wafer_id in wafer_ids:

    experiment_folders, thumbnail_folders, protocol_folders, alignment_folder, meta_folder, \
            region_strs = get_paths(wafer_id)

    czifile = os.path.join(czipath, czifiles[wafer_id]) if czifiles[wafer_id] is not None else None
    czifile_ribbon = czifile_ribbons[wafer_id]
    czifile_scene = czifile_scenes[wafer_id]
    czfile = os.path.join(czipath, czfiles[wafer_id]) if czfiles[wafer_id] is not None else None

    order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))
    rough_affine_skip_dill_fn = os.path.join(alignment_folder,
            rough_affine_dill_fn_str.format(wafer_id, use_rough_skip, run_str_in))

    czifile_rotation = czifile_rotations[wafer_ids[0]]
    region_rotation = region_rotations_all[wafer_ids[0]]

    cwafer = wafer([experiment_folders], [protocol_folders], [alignment_folder], [region_strs],
                 wafer_ids=[wafer_id], thumbnail_folders=thumbnail_folders, nimages_per_mfov=nimages_per_mfov,
                 scale_nm=scale_nm, legacy_zen_format=legacy_zen_format, verbose=True)
    if legacy_zen_format:
        cwafer.set_region_rotations_czi(czifile, scene=czifile_scene, ribbon=czifile_ribbon,
                 czfile=czfile, use_roi_polygons=czifile_use_roi_polygons)
        #cwafer.set_region_rotations_czi(czifile, scene=1, ribbon=czifile_ribbon, czfile=czfile, doplots=False)
    else:
        if region_rotation is not None:
            rotations = [x + czifile_rotation for x in region_rotation]
        else:
            rotations = [czifile_rotation]
        cwafer.set_region_rotations_manual(rotations=rotations)

    if run_str_in:
        with open(rough_affine_skip_dill_fn, 'rb') as f: d = dill.load(f)
        solved_order = d['solved_order']
        solved_order_bad_matches = d['solved_order_bad_matches']
        sections = np.split(solved_order, d['solved_order_bad_matches_inds'])
        sections_lens = np.array([x.size for x in sections])
    else:
        if os.path.isfile(order_txt_fn):
            solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        else:
            # just so this script can run without a saved solved ordering
            solved_order = np.arange(cwafer.region_to_limi_roi.size, dtype=np.uint64)
        solved_order_bad_matches = np.zeros((0,2), dtype=np.int64)
    print('Solved order size is {}'.format(solved_order.size))

    # if exclude_regions is not None and not no_exclude_regions:
    #     cexclude_regions = np.array(exclude_regions[wafer_id], dtype=np.int32) - 1
    # else:
    #     cexclude_regions = np.zeros((0,), dtype=np.int32)
    if not no_exclude_regions:
        cexclude_regions = np.ones(cwafer.region_to_limi_roi.size, dtype=bool)
        cexclude_regions[solved_order] = 0
        cexclude_regions = np.nonzero(cexclude_regions)[0]
    else:
        cexclude_regions = np.zeros((0,), dtype=np.int64)

    if legacy_zen_format:
        scene = CziMsem(czifile, scene=czifile_scene, ribbon=czifile_ribbon, verbose=True)
        if czfile:
            scene.load_cz_file_to_polys_or_rois(czfile, load_rois=True)
        else:
            scene.read_scene_meta()
        if overlay_limi:
            img = scene.read_scene_image()
        use_polygons = use_polygons or (scene.nROIs == 0)
        rois_points = (scene.polygons_points if use_polygons else scene.rois_points)
        scale = scene.scale / scene.scale_units * 1e6 # um
    else:
        rois_points = cwafer.region_roi_poly_raw
        scale = cwafer.scale_nm / 1e3 # um
    nROIs = len(rois_points)
    roictrs = np.zeros((nROIs,2), dtype=np.double)
    for i in range(nROIs):
        # xxx - this is  probably better added to CziScene whenever it stabilizes...
        if (rois_points[i][0,:] == rois_points[i][-1,:]).all(): rois_points[i] = rois_points[i][:-1,:]
        if polygonal_center:
            roictrs[i,:] = PolyCentroid(rois_points[i][:,0], rois_points[i][:,1])
        else:
            roictrs[i,:] = (rois_points[i].max(0)+rois_points[i].min(0))/2 # bounding box center
    rois_size = np.concatenate(np.array([x.max(0) - x.min(0) for x in rois_points])[None,:], axis=0)
    rois_size = rois_size.max(0) # meh, min or max for slice size?

    fig = plt.figure(1); fig.clf()
    fig.set_size_inches(fig_size[0],fig_size[1])
    ax = plt.gca()
    if solved_order_bad_matches.shape[0] > 0:
        cmap=[plt.cm.get_cmap('Blues'), plt.cm.get_cmap('Greens'), plt.cm.get_cmap('Reds')]
    else:
        cmap=plt.cm.get_cmap('viridis')
    if overlay_limi:
        plt.imshow(img, cmap='gray')
    else:
        #plt.xlim(roictrs[:,0].min(), roictrs[:,0].max())
        #plt.ylim(roictrs[:,1].min(), roictrs[:,1].max())
        rng = (roictrs[:,0].max() - roictrs[:,0].min())*limit_scale
        mid = (roictrs[:,0].max() + roictrs[:,0].min())/2
        plt.xlim(mid-rng/2, mid+rng/2)
        rng = (roictrs[:,1].max() - roictrs[:,1].min())*limit_scale
        mid = (roictrs[:,1].max() + roictrs[:,1].min())/2
        #plt.ylim(mid-rng/2, mid+rng/2)
        plt.ylim(mid+rng/2, mid-rng/2) # invert y to match image
    d = roictrs[cwafer.region_to_limi_roi[solved_order],:]

    # print the path length for total path length comparisons (mostly relevant for imaging order)
    path_length = np.sqrt(((d[:-1,:] - np.roll(d, -1, axis=0)[:-1,:])**2).sum())
    print('Solved slice order path length {:.3f}'.format(path_length))

    strs = [cwafer.wafer_region_strs[x].split('_')[1] for x in solved_order]
    ibad_match = jbad_match = ibad_seq_match = 0
    for i in range(solved_order.size):
        if solved_order_bad_matches.shape[0] > 0:
            color = cmap[ibad_seq_match](jbad_match/sections_lens[ibad_match]) if use_cmap else 'k'
        else:
            color = cmap(i/nROIs) if use_cmap else 'k'
        tcolor = color if not first_last else 'r'
        if not first_last or i==0 or i==solved_order.size-1:
            use_str = strs[i] if show_region_strs else str(i+1)
            ax.annotate(use_str, xy=(d[i,0], d[i,1]), ha='center', va='center', color=tcolor, fontsize=fontsize)
        # if bad matches is populated, do not draw the line if it is a bad match.
        # bad matches are defined by the adjacent region indices (not the solved order indices).
        # the order they are stored however is always in the solved order (from wafer_solver).
        cur_match = np.array([solved_order[i], solved_order[i+1]]) if i < solved_order.size-1 else -np.ones(2,)
        cur_bad_match = np.nonzero((cur_match == solved_order_bad_matches).all(1))[0]
        if cur_bad_match.size == 0:
            if show_order_lines:
                if i < solved_order.size-1:
                    ax.plot([d[i,0],d[i+1,0]],[d[i,1],d[i+1,1]], color=color, alpha=0.5)
            else:
                plt.plot(d[i,0], d[i,1], marker="o", markersize=10, alpha=scatter_alpha, linewidth=0,
                    fillstyle='full', markeredgecolor='red', markeredgewidth=0.0, markerfacecolor=color)
            jbad_match += 1
        else:
            jbad_match = 0
            ibad_match += 1
            ibad_seq_match += 1
    d = roictrs[cwafer.region_to_limi_roi[cexclude_regions],:]
    for i in range(cexclude_regions.size):
        plt.plot(d[i,0], d[i,1], marker="x", markersize=9, alpha=scatter_alpha, markeredgewidth=3,
            markeredgecolor='red')
    # do not do these
    #plt.gca().invert_yaxis()
    #plt.gca().set_aspect('equal', 'datalim')
    plt.axis('off')
    #fig.canvas.set_window_title('Solved region order') # deprecated
    plt.gcf().canvas.manager.set_window_title('Solved slice order')

    fig = plt.figure(2); fig.clf()
    fig.set_size_inches(fig_size[0],fig_size[1])
    ax = plt.gca()
    cmap=plt.cm.get_cmap('viridis')
    if overlay_limi:
        plt.imshow(img, cmap='gray')
    else:
        #plt.xlim(roictrs[:,0].min(), roictrs[:,0].max())
        #plt.ylim(roictrs[:,1].min(), roictrs[:,1].max())
        rng = (roictrs[:,0].max() - roictrs[:,0].min())*limit_scale
        mid = (roictrs[:,0].max() + roictrs[:,0].min())/2
        plt.xlim(mid-rng/2, mid+rng/2)
        rng = (roictrs[:,1].max() - roictrs[:,1].min())*limit_scale
        mid = (roictrs[:,1].max() + roictrs[:,1].min())/2
        #plt.ylim(mid-rng/2, mid+rng/2)
        plt.ylim(mid+rng/2, mid-rng/2) # invert y to match image

    if legacy_zen_format and zen_order_sort:
        nregion_strs = len(region_strs)

        new_region_strs = region_strs
        ireimage = [np.arange(len(x)) for x in region_strs]
        if len(zen_reimages) > 0:
            # this is to put the "original" slices back for slices that were reimaged,
            #   so the imaging order is preserved.
            zen_reimage = [[] for x in range(nregion_strs)]
            with open(zen_reimages[wafer_ind], 'r') as f:
                # mapaping file format is:
                # old_slice_str new_slice_str old_region_str_index new_slice_str_index
                for line in f:
                    sline = line.strip()
                    # comments and blank lines allowed
                    if not sline or sline[0]=='#': continue
                    fields = sline.split()
                    zen_reimage[int(fields[2])].append([fields[0], fields[1], int(fields[3])])

            new_region_strs = [[] for x in range(nregion_strs)]
            # first get the indices in the manifest of the old reimages
            iremove = [np.zeros(len(x), dtype=np.int64) for x in zen_reimage]
            cumsum = 0
            for i in range(nregion_strs):
                if len(zen_reimage[i]) > 0:
                    for region_str,j in zip(region_strs[i], range(len(region_strs[i]))):
                        inds = [region_str.endswith(x[0]) for x in zen_reimage[i]]
                        if any(inds):
                            ireimage[i][j] = -1
                            iremove[i][inds.index(True)] = j + cumsum + cwafer.nvalid_regions
                        else:
                            new_region_strs[i].append(region_str)
                else:
                    new_region_strs[i] = list(region_strs[i])
                cumsum += len(region_strs[i])
            # insert the new reimage names and the old reimage indices in the destination region_str indices
            for i in range(nregion_strs):
                inserts = []; iinserts = np.array([], dtype=np.int64)
                for j in range(nregion_strs):
                     inserts += [x[1] for x in zen_reimage[j] if x[2]==i]
                     iinserts = np.concatenate((iinserts, np.array([iremove[j][y] for x,y in \
                             zip(zen_reimage[j],range(len(zen_reimage[j]))) if x[2]==i], dtype=np.int64)))
                new_region_strs[i] += inserts
                ireimage[i] = ireimage[i][ireimage[i] > -1]
                ireimage[i] = np.concatenate((ireimage[i], iinserts))
            new_region_strs = [x for x in new_region_strs if len(x) > 0]
            ireimage = [x for x in ireimage if x.size > 0]
        # if len(zen_reimages) > 0:

        isort = np.array([], dtype=np.int64)
        cumsum = 0
        for i in range(len(new_region_strs)):
            inds = ireimage[i][np.array(argsort(new_region_strs[i]), np.int64)]
            inds[inds < cwafer.nvalid_regions] += cumsum
            inds[inds >= cwafer.nvalid_regions] -= cwafer.nvalid_regions
            isort = np.concatenate((isort, inds))
            cumsum += len(new_region_strs[i])
    else:
        isort = np.nonzero(cwafer.sel_valid_regions)[0]
    # if legacy_zen_format and zen_order_sort:

    #print([cwafer.wafer_region_strs[x] for x in isort])
    d = roictrs[cwafer.region_to_limi_roi[isort],:]

    if shortest_order_sort:
        isort2, _ = wafer_solver.distance_tsp_solver(d, corner_endpoints=False) #topn_cols=shortest_topn)
        isort = isort[isort2]; d = d[isort2,:]

    # print the path length for total path length comparisons (mostly relevant for imaging order)
    path_length = np.sqrt(((d[:-1,:] - np.roll(d, -1, axis=0)[:-1,:])**2).sum())
    print('Imaged slice order path length {:.3f}'.format(path_length))

    strs = [cwafer.wafer_region_strs[x].split('_')[1] for x in isort]
    for i in range(isort.size):
        color = cmap(i/nROIs) if use_cmap else 'k'
        tcolor = color if not first_last else 'r'
        if not first_last or i==0 or i==isort.size-1:
            use_str = strs[i] if show_region_strs else str(i+1)
            ax.annotate(use_str, xy=(d[i,0], d[i,1]), ha='center', va='center', color=tcolor, fontsize=fontsize)
        if show_order_lines:
            if i < isort.size-1:
                ax.plot([d[i,0],d[i+1,0]],[d[i,1],d[i+1,1]], color=color) #, alpha=0.3)
        else:
            plt.plot(d[i,0], d[i,1], marker="o", markersize=10, alpha=scatter_alpha, linewidth=0,
                fillstyle='full', markeredgecolor='red', markeredgewidth=0.0, markerfacecolor=color)
    # do not do these
    #plt.gca().invert_yaxis()
    #plt.gca().set_aspect('equal', 'datalim')
    plt.axis('off')
    #fig.canvas.set_window_title('Zeiss region order') # deprecated
    plt.gcf().canvas.manager.set_window_title('Imaged slice order')

    plt.figure(3); plt.gcf().clf()
    inds = isort[solved_order] if legacy_zen_format and zen_order_sort else solved_order
    plt.plot(inds)
    plt.xlabel('imaged order')
    plt.ylabel('solved slice order')
    plt.gca().set_aspect('equal')
    plt.xlim([-10, solved_order.size + 10])
    plt.ylim([-10, solved_order.size + 10])

    # histograms of distance and direction between neighboring slices
    plt.figure(4); plt.gcf().clf()
    d = roictrs[cwafer.region_to_limi_roi[solved_order],:]
    rads = np.zeros((nROIs-1,), dtype=np.double)
    angs = np.zeros((nROIs-1,), dtype=np.double)
    for i in range(solved_order.size-1):
        v = (d[i+1,:]-d[i,:])*scale
        rads[i] = np.sqrt((v*v).sum())
        angs[i] = np.arctan2(v[1], v[0])
    plt.subplot(1,2,1)
    #step = 0.1; bins = np.arange(5,10,step)
    #hist,bins = np.histogram(rads, bins)
    hist,bins = np.histogram(rads, 50)
    cbins = bins[:-1] + (bins[1]-bins[0])/2
    plt.plot(cbins, hist)
    if legacy_zen_format:
        if scene.nribbons > 0 and czifile_ribbon > 0:
            sz = scene.ribbon_sizes[czifile_ribbon-1,:].flat[:] * scale
        elif czifile_ribbon == 0:
            sz = scene.scene_size_pix.flat[:] * scale
        else:
            sz = [0,0]
    else:
        #sz = [0,0]
        sz = (roictrs.max(0) - roictrs.min(0)) * scale
    sz2 = rois_size * scale
    plt.title('box %.2fx%.2f (um)\nslice %.0fx%.0f (um)' % (sz[0], sz[1], sz2[0], sz2[1]))
    plt.xlabel('next dist (um)')
    plt.ylabel('count')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplot(1,2,2)
    hist,bins = np.histogram(angs/np.pi*180, 25)
    cbins = bins[:-1] + (bins[1]-bins[0])/2
    plt.plot(cbins, hist)
    plt.xlabel('next angle (deg)')


    if save_plots:
        # to save instead of show figures
        for f in plt.get_fignums():
            plt.figure(f)
            plt.savefig(os.path.join(meta_folder, 'order_plots',
                'wafer%d_order_limi_figure%d.png' % (wafer_id,f,)), dpi=300)
            plt.savefig(os.path.join(meta_folder, 'order_plots',
                'wafer%d_order_limi_figure%d.pdf' % (wafer_id,f,)))
    else:
        plt.show()
