#!/usr/bin/env python3
"""plot_regions.py

Top level command-line interface for generating plots related to the
  2D alignment and montaging of sections.

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
import time
import glob
import dill
import argparse

import numpy as np
#import scipy.linalg as lin
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing
#import scipy.spatial.distance as scidist
# import scipy.ndimage as nd
# import tifffile

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from msem import region, zimages
from msem.utils import big_img_load
from def_common_params import get_paths, meta_folder, all_wafer_ids, total_nwafers, order_txt_fn_str
from def_common_params import scale_nm, nimages_per_mfov, legacy_zen_format, native_subfolder
from def_common_params import dsstep, use_thumbnails_ds, twopass_default_tol_nm
from def_common_params import wafer_region_prefix_str, slice_balance_fn_str, align_subfolder
from def_common_params import brightness_slice_histo_nsat #, region_manifest_cnts

from def_common_params import tissue_mask_ds, tissue_mask_min_edge_um, tissue_mask_min_hole_edge_um


## argparse

parser = argparse.ArgumentParser(description='plot_regions.py')
parser.add_argument('--run-type', nargs=1, type=str, default=['residuals'],
    choices=['residuals', 'deltas', 'brightness', 'median-diff', 'histo-width'],
    help='the type of plots to generate')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[],
    help='specify to only plot a subset of wafer ids')
parser.add_argument('--region_inds', nargs='+', type=int, default=[-1],
    help='list of region indices to run (< 0 for all regions in wafer)')
parser.add_argument('--region-inds-rng', nargs=2, type=int, default=[-1,-1],
    help='if region_inds is not defined, create region_inds from this range (default to region_inds)')
parser.add_argument('--mfov-ids', nargs='*', type=int, default=[],
    help='which mfov ids (base 1) to view plots for (optional)')
parser.add_argument('--show-plots', dest='save_plots', action='store_false',
    help='display plots instead of saving as png')
parser.add_argument('--mean-wafer-id', nargs=1, type=int, default=[0],
    help='for the mean wafer values, subtract the mean for this wafer, 0 for off')
parser.add_argument('--no-solved-order', dest='solved_order', action='store_false',
    help='just use the indexed order instead of solved order')
parser.add_argument('--tissue-masks', dest='tissue_masks', action='store_true',
    help='use the tissue masks')
parser.add_argument('--native', dest='native', action='store_true',
    help='process native resolution images, not thumbnails')
parser.add_argument('--quiver-scale-diff', nargs=1, type=float, default=[1.],
    help='scale for the diff quivers, 1. is at xy scale')
parser.add_argument('--quiver-scale-res', nargs=1, type=float, default=[0.1],
    help='scale for the residual quivers, 1. is at xy scale')
parser.add_argument('--dsstep', nargs=1, type=int, default=[1],
                    help='downsampling to use for histo-width')

args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

# only plot for a subset of wafer ids
wafer_ids = args['wafer_ids']

# starting at 1 (Zeiss numbering)
region_inds = args['region_inds']
if region_inds[0] < 0 and all([x > -1 for x in args['region_inds_rng']]):
    region_inds = range(args['region_inds_rng'][0],args['region_inds_rng'][1])

# optionally specify particular mfovs to view (for deltas/residuals)
mfov_ids = args['mfov_ids']

# specify to subtract the mean for this wafer from the other param wafer means, zero to disable.
mean_wafer_id = args['mean_wafer_id'][0]

# whether to use the solved order or not
use_solved_order = args['solved_order']

# whether to save or display plots
save_plots = args['save_plots']

# scale for diff quivers, > 1. scales arrows down, < 1. scales arrows up
quiver_scl_diff = args['quiver_scale_diff'][0]

# scale for res quivers, > 1. scales arrows down, < 1. scales arrows up
quiver_scl_res = args['quiver_scale_res'][0]

# downsampling to use for histo width mode
dsstep_histo = args['dsstep'][0]

# read in tissues masks for each slice, more refined/precise that roi polygon
tissue_masks = args['tissue_masks']

# option to process / export native resolution regions
native = args['native']

# run type string is used for some filenames / paths
run_type = args['run_type'][0]

# specify to plot diff between solved and zeiss coordinates
residuals_plot = run_type == 'residuals'

# specify to plot deltas for individual mfovs
deltas_plot = run_type == 'deltas'

# specify to plot the rough accumulation
brightness_plot = run_type == 'brightness'

# specify to calculate widths of the mfov median-delta diffs
median_diff_plot = run_type == 'median-diff'

# specify to calculate widths of the mfov median-delta diffs
histo_width_plot = run_type == 'histo-width'


## fixed parameters not exposed in def_common_params

# for heads on quiver plots
#hw = 0; hl = 0; hal = 0 # no quiver arrows
hw = 5; hl = 5; hal = 5

# scatter of deltas vs position in x/y separately.
# if ther was a systematic accumulation of positions as a
#   function of tile position in the final image then this plot would show it.
# usually just annoying (too many plots come up).
plot_xy_scatters = False


## parameters that are determined based on above parameters

outdir = os.path.join(meta_folder, 'brightness')
os.makedirs(outdir, exist_ok=True)
use_solved_order = use_solved_order and brightness_plot
nmfov_ids = len(mfov_ids)

# # for annoying parts of this script that have to be aware of wafers
# # xxx - how to shield the slices from knowledge of wafers / the whole dataset stack???
# cum_manifest_cnts = np.cumsum(region_manifest_cnts[1:])

# # support for reading the tissue masks (to filter the keypoints)
# if tissue_masks:
#     assert(tissue_mask_path is not None)
# else:
#     tissue_mask_path = None


## plot routines

def make_param_plot(y, wafers_nimgs, ylabel='brightness', default_val=0., dimstrs=['x','y'], figno=3):
    plt.figure(figno); plt.clf()
    cum_wafers_nimgs = np.concatenate(([0], np.cumsum(wafers_nimgs)))
    total_nimgs = wafers_nimgs.sum()
    wafers_slices = np.concatenate([np.arange(x) for x in wafers_nimgs])
    x = np.arange(total_nimgs, dtype=np.double)
    plt.plot(x,y, alpha=0.8); ax1 = plt.gca()

    if not np.logical_or(y==0, np.logical_not(np.isfinite(y))).all():
        selx = np.isfinite(x)
        if y.ndim==1: y = y[:,None]
        for d in range(y.shape[1]):
            dimstr = dimstrs[d] if y.ndim > 1 else ''
            # calcs / plots linear regressions, useful for estimating linear decay term incase of any "bias"
            sel = np.logical_and(selx, np.isfinite(y[:,d]))
            X = preprocessing.PolynomialFeatures(degree=1).fit_transform(x[sel].reshape(-1, 1))
            reg = linear_model.LinearRegression(fit_intercept=False).fit(X, y[sel,d])
            yfit = reg.predict(X)
            print('Fitted %s %s slope is %.8f pixels / slice' % (ylabel, dimstr, reg.coef_[1],))
            plt.plot(x[sel],yfit, color='k', linestyle='dashed', linewidth=2)
            if mean_wafer_id > 0:
                imean = all_wafer_ids.index(mean_wafer_id)
                wmean = y[cum_wafers_nimgs[imean]:cum_wafers_nimgs[imean+1],d].mean()
            else:
                wmean = 0
            for iw in range(wafers_nimgs.size):
                wafermean = y[cum_wafers_nimgs[iw]:cum_wafers_nimgs[iw+1],d].mean()-wmean
                print('Mean %s %s for w%d = %g' % (ylabel, dimstr, all_wafer_ids[iw], wafermean,))

    amin = np.nanmin(y); amax = np.nanmax(y)
    for w in (cum_wafers_nimgs - 0.5):
        plt.plot([w,w], [amin,amax], 'r')
    for wid,w,cnt in zip(all_wafer_ids, cum_wafers_nimgs[:-1], wafers_nimgs):
        plt.text(w+cnt/2, amax, 'w%d' % (wid,), color='r')
    ax2 = ax1.twiny()
    y2 = np.empty((total_nimgs,), dtype=np.double); y2.fill(default_val)
    ax2.plot(x, y2, 'r') # Create a dummy plot
    ax2.set_xlim(ax1.get_xlim())
    def format_wafer_slice(x, pos=None):
        thisind = np.clip(int(x+0.5), 0, total_nimgs - 1)
        return str(wafers_slices[thisind])
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_wafer_slice))
    ax1.set_xlabel('slice index in order')
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel('wafer index in order')

def make_quiver_plot(figno, xy, dxy, scl, scatter=False, sel_r=None, keep=False):
    plt.figure(figno)
    if not keep: plt.gcf().clf()
    plt.quiver(xy[:,0], xy[:,1], dxy[:,0], dxy[:,1],
               angles='xy', scale_units='xy', scale=scl, color='k',
               linewidth=1, headaxislength=hal, headwidth=hw, headlength=hl)
    if scatter:
        plt.scatter(xy[:,0], xy[:,1], c='g', s=14, marker='.')
    plt.plot(0, 0, 'rx')
    if sel_r is not None:
        plt.scatter(xy[sel_r,0], xy[sel_r,1], c='r', s=14, marker='.')
    plt.gca().set_aspect('equal')
    if not keep: plt.gca().invert_yaxis()
    plt.axis('off')

## loop over wafers and get adjusts

if len(wafer_ids) > 0:
    use_nwafers = len(wafer_ids)
    use_wafer_ids = wafer_ids
else:
    use_nwafers = total_nwafers
    use_wafer_ids = all_wafer_ids

adjusts = np.zeros((0,1), dtype=np.double)
wafers_nimgs = np.zeros((use_nwafers,), dtype=np.int64)
for wafer_id, wafer_ind in zip(use_wafer_ids, range(use_nwafers)):
    experiment_folders, thumbnail_folders, protocol_folders, alignment_folder, _, region_strs = get_paths(wafer_id)
    nregions = sum([len(x) for x in region_strs])
    # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
    region_strs_flat = [item for sublist in region_strs for item in sublist]

    native_alignment_folder = os.path.join(alignment_folder, native_subfolder)
    order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))
    if native:
        slice_balance_fn = os.path.join(native_alignment_folder, slice_balance_fn_str.format(wafer_id))
    else:
        slice_balance_fn = os.path.join(alignment_folder, slice_balance_fn_str.format(wafer_id))

    if use_solved_order:
        solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based
    else:
        solved_order = np.arange(nregions)
    wafers_nimgs[wafer_ind] = solved_order.size

    if brightness_plot:
        assert( os.path.isfile(slice_balance_fn) )
        cadjusts = zimages.load_slice_balance_file(slice_balance_fn)
        adjusts = np.concatenate((adjusts, cadjusts[solved_order][:,None]), axis=0)

    if deltas_plot or residuals_plot or (histo_width_plot and region_inds[0] > -1) or median_diff_plot:
        cregion_inds = region_inds if region_inds[0] > -1 else range(1,nregions+1)
        for region_ind in cregion_inds:
            print('instantiating region {}'.format(region_ind,)); t = time.time()
            cregion = region(experiment_folders, protocol_folders, region_strs, region_ind,
                    thumbnail_folders=thumbnail_folders, dsstep=dsstep, use_thumbnails_ds=use_thumbnails_ds,
                    mfov_ids=(None if nmfov_ids==0 else mfov_ids), legacy_zen_format=legacy_zen_format,
                    scale_nm=scale_nm, nimages_per_mfov=nimages_per_mfov,
                    tissue_mask_ds=tissue_mask_ds, tissue_mask_min_edge_um=tissue_mask_min_edge_um,
                    tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um)
            prefix = wafer_region_prefix_str.format(wafer_id, cregion.region_str)
            mfov_str = '' if nmfov_ids != 1 else ('_mfov{}'.format(mfov_ids[0]))
            coords_fn = os.path.join(alignment_folder, prefix + mfov_str + '_coords.txt')
            slice_alignment_dill_fn = os.path.join(alignment_folder, align_subfolder,
                    prefix + mfov_str + '_alignment.dill')
            cregion.load_stitched_region_image_coords(coords_fn)
            cregion.create_region_coords_from_inner_rect_coords(cregion.region_coords, load_neighbors=(nmfov_ids > 0))
            if twopass_default_tol_nm is not None:
                twopass_default_tol = [[x / cregion.scale_nm / cregion.dsstep for x in twopass_default_tol_nm[y]] \
                    for y in range(2)]
            print(twopass_default_tol)
            print('\tdone in %.4f s' % (time.time() - t, ))

            if median_diff_plot and region_ind == cregion_inds[0]:
                # init
                md_width = np.empty((nregions,2), dtype=np.double); md_width.fill(np.nan)
                md_step = 4; md_bins = np.arange(-768,768,md_step); md_cbins = md_bins[:-1] + md_step/2
                md_cbins = md_bins[:-1] + (md_bins[1]-md_bins[0])/2
                md_ctr = md_cbins.size//2
                md_hist_selx = np.logical_or(md_cbins < -twopass_default_tol[1][0],
                        md_cbins > twopass_default_tol[1][0])
                md_hist_sely = np.logical_or(md_cbins < -twopass_default_tol[1][1],
                        md_cbins > twopass_default_tol[1][1])
                md_cdf = np.empty((nregions,2), dtype=np.double); md_cdf.fill(np.nan)
                md_strs = [None]*nregions

                if wafer_ind == 0:
                    rh_step = 1/16; rh_bins = np.arange(0,320,rh_step); rh_cbins = rh_bins[:-1] + rh_step/2
                    rh_cbins = rh_bins[:-1] + (rh_bins[1]-rh_bins[0])/2
                    rh_counts = np.zeros((rh_cbins.size,2), dtype=np.int64)
                    rho_counts = np.zeros((rh_cbins.size,2), dtype=np.int64)
                    rh_mag_counts = np.zeros((rh_cbins.size,), dtype=np.int64)
                    rho_mag_counts = np.zeros((rh_cbins.size,), dtype=np.int64)
                    residual_region_cnt = residual_mfov_cnt = residual_inlier_cnt = residual_outlier_cnt = 0

            if residuals_plot or median_diff_plot:
                with open(slice_alignment_dill_fn, 'rb') as f: d = dill.load(f)

            cmfov = cregion
            if nmfov_ids==0 and not median_diff_plot:
                # mfov relative positions plot
                plt.figure(40); plt.gcf().clf()
                ax = plt.subplot(1, 1, 1)
                ax.set_aspect('equal', 'datalim')
                cmfov = cregion
                rng = (cmfov.region_coords[:,0,0].max() - cmfov.region_coords[:,0,0].min())*2
                mid = (cmfov.region_coords[:,0,0].max() + cmfov.region_coords[:,0,0].min())/2
                plt.xlim(mid-rng/2, mid+rng/2)
                rng = (cmfov.region_coords[:,0,1].max() - cmfov.region_coords[:,0,1].min())*2
                mid = (cmfov.region_coords[:,0,1].max() + cmfov.region_coords[:,0,1].min())/2
                plt.ylim(-mid+rng/2, -mid-rng/2)
                # y-direction is flipped in the image space
                plt.gca().invert_yaxis()
                for i in range(cmfov.nmfovs):
                   ax.text(cmfov.region_coords[i,0,0],-cmfov.region_coords[i,0,1],i+1,color='red')
                plt.axis('off')
                if save_plots:
                    plt.savefig(os.path.join(meta_folder, 'region_plots',
                        'region_{}_mfov_positions.png'.format(cregion.region_str)), dpi=300)

            if deltas_plot:
                mfov_ids = [x+1 for x in cregion.mfov_ids] if nmfov_ids > 0 else [cregion.mfov_ids[0]+1]
            elif residuals_plot:
                mfov_ids = [x+1 for x in cregion.mfov_ids] if nmfov_ids > 0 else []
            elif median_diff_plot:
                mfov_ids = []
            elif histo_width_plot:
                break
            else:
                assert(False)
            for mfov_id in mfov_ids:
                mfov_ind = mfov_id-1

                # select out the neighboring tiles based on non-zero entries in the adj matrix for current mfov
                subs = np.nonzero(cmfov.adj_matrix)
                inds = np.ravel_multi_index(subs, (cmfov.nTilesRect,cmfov.nTilesRect)); subs = np.transpose(subs)

                # re-create deltas from solved mfov coordinates
                xy_fixed = cregion.mfov_coords[mfov_ind]
                nimgs = cmfov.adj_matrix.shape[0]
                # Dx = np.zeros((nimgs,nimgs), dtype=np.double)
                # Dy = np.zeros((nimgs,nimgs), dtype=np.double)
                # Dx[subs[:,0], subs[:,1]] = xy_fixed[subs[:,0],0] - xy_fixed[subs[:,1],0]
                # Dy[subs[:,0], subs[:,1]] = xy_fixed[subs[:,0],1] - xy_fixed[subs[:,1],1]
                mfov_deltas = xy_fixed[subs[:,0],:] - xy_fixed[subs[:,1],:]

                # create tiled coords
                coords_imgs_rect = np.zeros((cmfov.nTilesRect,2), dtype=np.double)
                coords = cmfov.ring['coords']*np.array(cmfov.images[0].shape)[::-1]
                coords_imgs_rect[cmfov.ring['hex_to_rect'],0] = coords[:,0]
                coords_imgs_rect[cmfov.ring['hex_to_rect'],1] = coords[:,1]
                coords_imgs_rect_xy = (coords_imgs_rect[subs[:,0],:] + coords_imgs_rect[subs[:,1],:])/2
                triu = (subs[:,0] < subs[:,1])

                # show the mfov / tile ids for a single mfov with neighbors.
                # useful for debug / finding individual tiles.
                plt.figure(1234); plt.gcf().clf()
                ax = plt.subplot(1, 1, 1)
                ax.set_aspect('equal', 'datalim')
                itmp = np.unique(subs)
                crds = coords_imgs_rect
                plt.xlim(np.nanmin(crds[itmp,0]), np.nanmax(crds[itmp,0]))
                # y-direction is flipped in the image space
                plt.ylim(-np.nanmax(crds[itmp,1]), -np.nanmin(crds[itmp,1]))
                for i in range(cmfov.nTilesRect):
                    if i not in itmp or not np.isfinite(crds[i]).all(): continue
                    ax.text(crds[i,0],-crds[i,1],cmfov.mfov_ids_rect[i]+1,color='green',fontsize=10)
                    ax.text(crds[i,0],-crds[i,1]-350,cmfov.mfov_tile_ids_rect[i]+1,color='blue',fontsize=10)
                plt.axis('off')
                if save_plots:
                    plt.savefig(os.path.join(meta_folder, 'region_plots',
                        'deltas_{}_mfov{}.png'.format(cregion.region_str,mfov_id)), dpi=300)

                if residuals_plot:
                    # individual mfov residuals
                    residuals = d['mfov_residuals'][mfov_ind]; residuals_xy = d['mfov_residuals_xy'][mfov_ind]
                    residuals_triu = d['mfov_residuals_triu'][mfov_ind]
                    # select comparisons only in one direction
                    residuals = residuals[residuals_triu,:]; residuals_xy = residuals_xy[residuals_triu,:]
                    residuals_xy -= residuals_xy.mean(0)
                    make_quiver_plot(1244, residuals_xy, residuals, quiver_scl_res, scatter=True)
                    plt.gca().set_title('residuals after outliers removed')
                    if save_plots:
                        plt.savefig(os.path.join(meta_folder, 'region_plots',
                            'residuals_region_{}_mfov{}.png'.format(cregion.region_str,mfov_ind)), dpi=300)
                    mag = np.sqrt((residuals*residuals).sum(1))
                    print('residual mean mag {}'.format(np.mean(mag)))
                    print('residual mean std {}'.format(np.std(mag)))

                    residuals_orig = d['mfov_residuals_orig'][mfov_ind]
                    residuals_orig = residuals_orig[residuals_triu,:]
                    make_quiver_plot(1254, residuals_xy, residuals_orig, quiver_scl_diff,
                        sel_r=(residuals != residuals_orig).any(1), scatter=True)
                    plt.gca().set_title('residuals before outliers removed')
                    if save_plots:
                        plt.savefig(os.path.join(meta_folder, 'region_plots',
                            'residuals_woutliers_region_{}_mfov{}.png'.format(cregion.region_str,mfov_id)), dpi=300)
                #if residuals_plot:

                # individual mfov deltas relative to tiled deltas
                plt.figure(1248); plt.gcf().clf()
                deltas = (coords_imgs_rect[subs[:,0],:] - coords_imgs_rect[subs[:,1],:]) - mfov_deltas
                coords_imgs_rect_xy = (coords_imgs_rect[subs[:,0],:] + coords_imgs_rect[subs[:,1],:])/2
                # select comparisons only in one direction
                deltas = deltas[triu,:]; deltas_xy = coords_imgs_rect_xy[triu,:]
                make_quiver_plot(1248, deltas_xy, deltas, quiver_scl_diff, scatter=True)

                if residuals_plot:
                    plt.figure(1264); plt.gcf().clf()
                    # xcorr histogram
                    C = d['xcorrs'][mfov_ind]
                    step = 0.025; Cbins = np.arange(0,1,step); Ccbins = Cbins[:-1] + step/2
                    nzC = C[C > 0].flat[:]
                    #Chist,Cbins = np.histogram(nzC, 50)
                    Chist,Cbins = np.histogram(nzC, Cbins)
                    Ccbins = Cbins[:-1] + (Cbins[1]-Cbins[0])/2
                    ax = plt.subplot(1, 2, 1); plt.plot(Ccbins, Chist, 'b.-')
                    plt.xlabel('correlations'); plt.ylabel('count')
                    # image tile variance histogram
                    V = d['img_variances'][mfov_ind]; Vsel = np.isfinite(V)
                    Vhist,Vbins = np.histogram(V[Vsel], 20)
                    Vcbins = Vbins[:-1] + (Vbins[1]-Vbins[0])/2
                    ax = plt.subplot(1, 2, 2)
                    plt.plot(Vcbins, Vhist, 'b.-'); plt.xlabel('image variance'); plt.ylabel('count')
                    plt.title('total imgs %d' % (Vsel.sum(),))
                    #plt.plot([V_cutoff, V_cutoff], [0, Vhist.max()], 'r--')
                    #ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    if save_plots:
                        plt.savefig(os.path.join(meta_folder, 'region_plots',
                            'xcorrs_vars_region_{}_mfov{}.png'.format(cregion.region_str,mfov_id)), dpi=300)
                #if residuals_plot:

                if not save_plots:
                    plt.show()
            #for mfov_id in mfov_ids:

            if (residuals_plot or median_diff_plot) and nmfov_ids == 0:
                print(cregion.region_str)
                # for coords deltas plot
                # self.stitched_region_coords = np.zeros((self.nTotalTiles,2), dtype=np.double)
                # self.flat_zeiss_region_coords = np.zeros((self.nTotalTiles,2), dtype=np.double)
                sel = np.isfinite(cregion.stitched_region_coords).all(1)
                acrds = cregion.stitched_region_coords[sel,:]; zcrds = cregion.flat_zeiss_region_coords[sel,:]
                acrds -= acrds.mean(0); zcrds -= zcrds.mean(0)
                dcrds = acrds - zcrds

                if residuals_plot:
                    make_quiver_plot(10, zcrds, dcrds, quiver_scl_diff)
                    plt.gca().set_title('stage coords diff')
                    if save_plots:
                        plt.savefig(os.path.join(meta_folder, 'region_plots',
                            'residuals_coords_diff_{}.png'.format(cregion.region_str,)), dpi=300)

                # for region mfov_stitch residuals plot
                residuals = d['residuals']; residuals_xy = d['residuals_xy']; residuals_triu = d['residuals_triu']
                # select comparisons only in one direction
                residuals = residuals[residuals_triu,:]; residuals_xy = residuals_xy[residuals_triu,:]

                # create mask that only selects for residuals within the tissue mask area
                # xxx - this is broken, need to read the masks out of the slice
                # if cregion.tissue_mask_path is not None:
                #     # create the full filename for the masks
                #     # have to do this if you did not sort the region exports by manifest index before cubing.
                #     #use_region_ind = argsort(region_strs_flat).index(region_ind-1)
                #     # or if they were
                #     use_region_ind = region_ind-1
                #     tind = use_region_ind if wafer_id < 2 else use_region_ind + cum_manifest_cnts[wafer_id-2]
                #     fn = os.path.join(cregion.tissue_mask_path, cregion.tissue_mask_fn_str.format(tind))
                #
                #     bw = tifffile.imread(fn).astype(bool)
                #     #cregion.tissue_mask_min_size, cregion.tissue_mask_min_hole_size, cregion.tissue_mask_ds,
                #
                #     if cregion.tissue_mask_min_size > 0:
                #         # remove small components
                #         labels, nlbls = nd.label(bw, structure=nd.generate_binary_structure(2,2))
                #         if nlbls > 0:
                #             sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                #             rmv = np.nonzero(sizes < cregion.tissue_mask_min_size)[0] + 1
                #             if rmv.size > 0:
                #                 bw[np.isin(labels, rmv)] = 0
                #
                #     if cregion.tissue_mask_min_hole_size > 0:
                #         # remove small holes
                #         labels, nlbls = nd.label(np.logical_not(bw),
                #             structure=nd.generate_binary_structure(2,1))
                #         if nlbls > 0:
                #             sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                #             add = np.nonzero(sizes < cregion.tissue_mask_min_hole_size)[0] + 1
                #             if add.size > 0:
                #                 bw[np.isin(labels, add)] = 1
                #
                #     residuals_xy -= residuals_xy.min(0)
                #     rel_ds = tissue_mask_ds // dsstep
                #     ipts = np.round(residuals_xy / rel_ds).astype(np.int64)
                #     tissue_mask_sel = np.array([bw[x[1],x[0]] for x in ipts])
                #     tissue_mask_nsel = np.logical_not(tissue_mask_sel)
                #     tissue_mask_bw = bw
                #     tm_sz = np.array(tissue_mask_bw.shape)[::-1]
                # else: #if cregion.tissue_mask_path is not None:
                #     tissue_mask_sel = tissue_mask_nsel = tissue_mask_bw = None
                print('WARNING: you need to fix the tissue masks for residual plotting')
                tissue_mask_sel = tissue_mask_nsel = tissue_mask_bw = None

                residuals_xy -= residuals_xy.mean(0)
                if tissue_mask_sel is not None: residuals_xy[tissue_mask_nsel,:] = np.nan

                if residuals_plot:
                    make_quiver_plot(20, residuals_xy, residuals, quiver_scl_res)
                    plt.gca().set_title('region stitch residuals')
                    if save_plots:
                        plt.savefig(os.path.join(meta_folder, 'region_plots',
                            'residuals_region_stitch_{}.png'.format(cregion.region_str,)), dpi=300)

                if median_diff_plot: assert(d['mfov_deltas'] is not None)
                if d['mfov_deltas'] is not None:
                    # median deltas over all mfovs, compute for the twopass method in region
                    residuals = d['mfov_deltas']; residuals_xy = d['mfov_deltas_xy']
                    residuals_triu = d['mfov_deltas_triu']
                    residuals_xy = residuals_xy[residuals_triu,:]
                    residuals_xy -= residuals_xy.mean(0)
                    mfov_ctrs = cregion.region_coords[:,0,:]; mfov_ctrs = mfov_ctrs - mfov_ctrs.mean(0)
                    mfov_ctrs *= 1.5 # so the overlapping mfovs are visible in each one

                    # xxx - this is broken, need to read the masks out of the slice
                    # if cregion.tissue_mask_path is not None:
                    #     #cresiduals_xy = d['mfov_deltas_xy'][residuals_triu,:] # NO, need all for histos
                    #     cresiduals_xy = d['mfov_deltas_xy']
                    #     cresiduals_xy = cresiduals_xy - cresiduals_xy.min(0)
                    #     cmfov_ctrs = cregion.region_coords[:,0,:]
                    #     cmfov_ctrs = cmfov_ctrs - cmfov_ctrs.min(0)
                    #
                    #     mfov_tissue_mask_sel = np.zeros(d['mfovDx'].shape, dtype=bool)
                    #     for mfov_id,cnt in zip(cregion.mfov_ids, range(cregion.nmfov_ids)):
                    #         xy = cresiduals_xy + cmfov_ctrs[cnt,:]
                    #         rel_ds = tissue_mask_ds // dsstep
                    #         ipts = np.round(xy / rel_ds).astype(np.int64)
                    #         sel = (ipts >= tm_sz).any(1); ipts[sel, :] = 0
                    #         n = sel.sum()
                    #         if n > 0:
                    #             print('WARNING: {} delta points are outside of mask image shape'.format(n))
                    #         mfov_tissue_mask_sel[:,cnt] = np.array([tissue_mask_bw[x[1],x[0]] for x in ipts])
                    #         mfov_tissue_mask_sel[sel,cnt] = 0
                    #     mfov_tissue_mask_nsel = np.logical_not(mfov_tissue_mask_sel)
                    # else:
                    #     mfov_tissue_mask_sel = mfov_tissue_mask_nsel = None
                    mfov_tissue_mask_sel = mfov_tissue_mask_nsel = None
                    print('WARNING: you need to fix the tissue masks for residual plotting')

                    sel_bmfov = d['sel_btw_mfovs']; sel_imfov = np.logical_not(sel_bmfov)
                    if residuals_plot:
                        # select comparisons only in one direction
                        residuals = residuals[residuals_triu,:]
                        make_quiver_plot(30, residuals_xy, residuals, quiver_scl_diff, scatter=True)
                        plt.gca().set_title('mfov median deltas vs tiled')

                        if save_plots:
                            plt.savefig(os.path.join(meta_folder, 'region_plots',
                                'median_mfov_deltas_{}.png'.format(cregion.region_str,)), dpi=300)

                        # make histograms of all the deltas vs the tiled deltas for within and between mfovs
                        plt.figure(35); plt.gcf().clf()
                        plt.subplot(2,2,1)
                        step = 1; bins = np.arange(-768,768,step); cbins = bins[:-1] + step/2
                        sel = np.logical_and(np.isfinite(d['mfovDx']), sel_imfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDx'][sel], bins)
                        cbins = bins[:-1] + (bins[1]-bins[0])/2
                        plt.plot(cbins, hist)
                        plt.gca().set_xlabel('inner x delta vs median')
                        plt.gca().set_ylabel('count')
                        plt.subplot(2,2,2)
                        step = 4; bins = np.arange(-768,768,step); cbins = bins[:-1] + step/2
                        sel = np.logical_and(np.isfinite(d['mfovDx']), sel_bmfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDx'][sel], bins)
                        cbins = bins[:-1] + (bins[1]-bins[0])/2
                        plt.plot(cbins, hist)
                        plt.gca().set_xlabel('outer x delta vs median')
                        plt.gca().set_ylabel('count')
                        plt.subplot(2,2,3)
                        step = 1; bins = np.arange(-768,768,step); cbins = bins[:-1] + step/2
                        sel = np.logical_and(np.isfinite(d['mfovDy']), sel_imfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDy'][sel], bins)
                        cbins = bins[:-1] + (bins[1]-bins[0])/2
                        plt.plot(cbins, hist)
                        plt.gca().set_xlabel('inner y delta vs median')
                        plt.gca().set_ylabel('count')
                        plt.subplot(2,2,4)
                        step = 4; bins = np.arange(-768,768,step); cbins = bins[:-1] + step/2
                        sel = np.logical_and(np.isfinite(d['mfovDy']), sel_bmfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDy'][sel], bins)
                        cbins = bins[:-1] + (bins[1]-bins[0])/2
                        plt.plot(cbins, hist)
                        plt.gca().set_xlabel('outer y delta vs median')
                        plt.gca().set_ylabel('count')

                        # plot the deltas versus position (including mfov)
                        plt.figure(37); plt.gcf().clf()
                        for mfov_id,cnt in zip(cregion.mfov_ids, range(cregion.nmfov_ids)):
                            # select comparisons only in one direction
                            residuals2 = np.concatenate((d['mfovDx'][:,cnt][:,None],d['mfovDy'][:,cnt][:,None]),axis=1)
                            residuals2 = residuals2[residuals_triu,:]
                            xy = residuals_xy + mfov_ctrs[cnt,:]
                            if mfov_tissue_mask_nsel is not None:
                                xy[mfov_tissue_mask_nsel[residuals_triu,cnt],:] = np.nan
                            make_quiver_plot(37, xy, residuals2, quiver_scl_res, scatter=False, keep=True)
                        plt.gca().invert_yaxis()
                        plt.gca().set_title('mfov deltas vs median')

                        # plot the deltas versus position (including mfov) with outliers removed
                        # also make scattter plots collapsed across x and y
                        plt.figure(38); plt.gcf().clf()
                        if plot_xy_scatters:
                            plt.figure(101); plt.gcf().clf()
                            plt.figure(102); plt.gcf().clf()
                            plt.figure(103); plt.gcf().clf()
                            plt.figure(104); plt.gcf().clf()
                        residuals = d['mfov_deltas']
                        sel_bmfov = d['sel_btw_mfovs']; sel_imfov = np.logical_not(sel_bmfov)
                        for mfov_id,cnt in zip(cregion.mfov_ids, range(cregion.nmfov_ids)):
                            # select comparisons only in one direction
                            residuals2 = np.concatenate((d['mfovDx'][:,cnt][:,None],d['mfovDy'][:,cnt][:,None]),axis=1)
                            fsel = np.logical_not(np.isfinite(residuals2).all(1))
                            #residuals2[fsel,:] = 0

                            # remove the outliers based on the medians
                            sel = (np.abs(residuals2) > twopass_default_tol[0]).any(axis=1)
                            sel[sel_bmfov] = 0
                            sel_r = sel
                            sel = (np.abs(residuals2) > twopass_default_tol[1]).any(axis=1)
                            sel[sel_imfov] = 0
                            sel_r = np.logical_or(sel_r,sel)
                            residuals2[sel_r,:] = np.nan
                            residuals2[fsel,:] = np.nan

                            residuals2 = residuals2[residuals_triu,:] #; sel_r = sel_r[residuals_triu]
                            xy = residuals_xy + mfov_ctrs[cnt,:]
                            if mfov_tissue_mask_nsel is not None:
                                xy[mfov_tissue_mask_nsel[residuals_triu,cnt],:] = np.nan
                            make_quiver_plot(38, xy, residuals2, quiver_scl_res, scatter=False, keep=True, sel_r=None)

                            if plot_xy_scatters:
                                residuals_bmfov = residuals2.copy()
                                residuals_imfov = residuals2.copy()
                                residuals_bmfov[sel_imfov,:] = np.nan
                                residuals_imfov[sel_bmfov,:] = np.nan
                                residuals_bmfov = residuals_bmfov[residuals_triu,:]
                                residuals_imfov = residuals_imfov[residuals_triu,:]
                                plt.figure(101)
                                plt.scatter(xy[:,0], residuals_bmfov[:,0], c='b', s=14, marker='.')
                                plt.scatter(xy[:,0], residuals_bmfov[:,1], c='g', s=14, marker='.')
                                plt.figure(102)
                                plt.scatter(xy[:,0], residuals_imfov[:,0], c='b', s=14, marker='.')
                                plt.scatter(xy[:,0], residuals_imfov[:,1], c='g', s=14, marker='.')
                                plt.figure(103)
                                plt.scatter(xy[:,1], residuals_bmfov[:,0], c='b', s=14, marker='.')
                                plt.scatter(xy[:,1], residuals_bmfov[:,1], c='g', s=14, marker='.')
                                plt.figure(104)
                                plt.scatter(xy[:,1], residuals_imfov[:,0], c='b', s=14, marker='.')
                                plt.scatter(xy[:,1], residuals_imfov[:,1], c='g', s=14, marker='.')
                            #if plot_xy_scatters:
                        #for mfov_id,cnt in zip(cregion.mfov_ids, range(cregion.nmfov_ids)):

                        plt.figure(38)
                        plt.gca().invert_yaxis()
                        plt.gca().set_title('mfov deltas vs median no outliers')
                        if plot_xy_scatters:
                            plt.figure(101)
                            plt.gca().set_title('mfov deltas vs median btw mfovs no outliers')
                            plt.gca().set_xlabel('x (tiled) mfov position')
                            plt.legend(['x', 'y'])
                            plt.figure(102)
                            plt.gca().set_title('mfov deltas vs median in mfovs no outliers')
                            plt.gca().set_xlabel('x (tiled) mfov position')
                            plt.legend(['x', 'y'])
                            plt.figure(103)
                            plt.gca().set_title('mfov deltas vs median btw mfovs no outliers')
                            plt.gca().set_xlabel('y (tiled) mfov position')
                            plt.legend(['x', 'y'])
                            plt.figure(104)
                            plt.gca().set_title('mfov deltas vs median in mfovs no outliers')
                            plt.gca().set_xlabel('y (tiled) mfov position')
                            plt.legend(['x', 'y'])
                        #if plot_xy_scatters:
                    else: #if residuals_plot:
                        i = region_ind-1
                        sel = np.logical_and(np.isfinite(d['mfovDx']), sel_bmfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDx'][sel], md_bins)
                        cutoff = hist.max()/4 # base cutoff on max to measure width
                        md_width[i,0] = max([(md_ctr - np.nonzero(hist > cutoff)[0][0])*md_step,
                                             (md_ctr - np.nonzero(hist[::-1] > cutoff)[0][0])*md_step])
                        md_cdf[i,0] = hist[md_hist_selx].sum() / hist.sum()
                        sel = np.logical_and(np.isfinite(d['mfovDy']), sel_bmfov[:,None])
                        if mfov_tissue_mask_sel is not None: sel = np.logical_and(sel, mfov_tissue_mask_sel)
                        hist,bins = np.histogram(d['mfovDy'][sel], md_bins)
                        cutoff = hist.max()/4 # base cutoff on max to measure width
                        md_width[i,1] = max([(md_ctr - np.nonzero(hist > cutoff)[0][0])*md_step,
                                            (md_ctr - np.nonzero(hist[::-1] > cutoff)[0][0])*md_step])
                        md_cdf[i,1] = hist[md_hist_sely].sum() / hist.sum()
                        md_strs[i] = prefix

                        mfov_ids = [x+1 for x in cregion.mfov_ids]
                        for mfov_id in cregion.mfov_ids:
                            mfov_ind = mfov_id-1

                            # individual mfov residuals
                            residuals = d['mfov_residuals'][mfov_ind]
                            #residuals_xy = d['mfov_residuals_xy'][mfov_ind]
                            residuals_triu = d['mfov_residuals_triu'][mfov_ind]
                            # select comparisons only in one direction
                            residuals = residuals[residuals_triu,:]
                            #residuals_xy = residuals_xy[residuals_triu,:]
                            #residuals_xy -= residuals_xy.mean(0)

                            residuals_orig = d['mfov_residuals_orig'][mfov_ind]
                            residuals_orig = residuals_orig[residuals_triu,:]

                            isel=(residuals == residuals_orig).all(1)
                            osel=(residuals != residuals_orig).any(1)
                            hist,bins = np.histogram(residuals[isel,0], rh_bins); rh_counts[:,0] += hist
                            hist,bins = np.histogram(residuals[isel,1], rh_bins); rh_counts[:,1] += hist
                            #hist,bins = np.histogram(residuals_orig[osel,0], rh_bins); rho_counts[:,0] += hist
                            #hist,bins = np.histogram(residuals_orig[osel,1], rh_bins); rho_counts[:,1] += hist
                            hist,bins = np.histogram(residuals[osel,0], rh_bins); rho_counts[:,0] += hist
                            hist,bins = np.histogram(residuals[osel,1], rh_bins); rho_counts[:,1] += hist
                            tmp = np.sqrt((residuals[isel,:]**2).sum(1))
                            hist,bins = np.histogram(tmp, rh_bins); rh_mag_counts += hist
                            tmp = np.sqrt((residuals[osel,:]**2).sum(1))
                            hist,bins = np.histogram(tmp, rh_bins); rho_mag_counts += hist

                            residual_inlier_cnt += isel.sum()
                            residual_outlier_cnt += osel.sum()
                            residual_mfov_cnt += 1
                        #for mfov_id in cregion.mfov_ids:
                        residual_region_cnt += 1

                #if d['mfov_deltas'] is not None:

                if not save_plots and residuals_plot: plt.show()
            #if residuals_plot and nmfov_ids == 0:
        #for region_ind in cregion_inds:
    #if deltas_plot or residuals_plot:

    # this is for trying to figure out a good tolerance vs the median deltas for twopass 2d alignment.
    if median_diff_plot:
        nprint = nregions
        #nprint = 20
        print('wafer {}'.format(wafer_id))
        pvals = [md_width, md_cdf]
        pvals_strs = ['histo width', '% out of tol']
        for j in range(len(pvals)):
            print('metric {}'.format(pvals_strs[j]))
            if nprint < nregions:
                print('smallest {}'.format(nprint))
            pval = pvals[j].copy()
            pval[np.logical_not(np.isfinite(pval))] = np.inf
            inds = np.argsort(pval, 0); vals = np.sort(pval, 0)
            for i in range(nprint):
                print('x {} {}, y {} {}'.format(md_strs[inds[i,0]], vals[i,0], md_strs[inds[i,1]], vals[i,1]))
            if nprint < nregions:
                print('biggest {}'.format(nprint))
                pval = pvals[j].copy()
                pval[np.logical_not(np.isfinite(pval))] = 0
                inds = np.argsort(pval, 0)[::-1,:]; vals = np.sort(pval, 0)[::-1,:]
                for i in range(nprint):
                    print('x {} {}, y {} {}'.format(md_strs[inds[i,0]], vals[i,0], md_strs[inds[i,1]], vals[i,1]))
        # for j in range(len(pvals)):
    #if median_diff_plot:

    # this is for trying to figure out a good tolerance vs the median deltas for twopass 2d alignment.
    if histo_width_plot:
        # just glob for all the available stitched regions instead of having to instantiate each region
        fns = glob.glob(os.path.join(alignment_folder, native_subfolder if native else '', '*_stitched.h5'))
        # sort newest to oldest makes looking at reimages easier
        fns.sort(key=os.path.getmtime); fns = fns[::-1]
        if region_inds[0] > -1:
            fns = [x for x in fns if cregion.region_str in x]
        nfns = len(fns)
        #nfns = 20 # to test or use with reimages
        rng = [0.05, 0.98]
        width = np.empty((nfns,1), dtype=np.int64); width.fill(-1)
        mode = np.empty((nfns,1), dtype=np.int64); mode.fill(-1)

        for fn,i in zip(fns, range(nfns)):
            dsstr = ('_'+str(dsstep_histo)) if dsstep_histo > 1 else ''
            print(fn)
            histo, _ = big_img_load(fn, dataset='histogram'+dsstr)
            # if i==0:
            #     histos = np.empty((nfns, histo.size), dtype=np.int64); histos.fill(-1)
            # histos[i,:] = histo

            # remove saturated pixels at ends.
            histo_nsat = brightness_slice_histo_nsat
            # replace with the next bin
            #histo[:histo_nsat[0]] = histo[histo_nsat[0]]
            #histo[-histo_nsat[1]:] = histo[-histo_nsat[1]-1]
            # replace with zeros
            histo[:histo_nsat[0]] = 0; histo[-histo_nsat[1]:] = 0

            mode[i] = np.argmax(histo)
            dhisto = np.cumsum(histo)
            hsum = histo.sum()
            if hsum > 0:
                dhisto = dhisto / hsum
                width[i] = np.nonzero(dhisto > rng[1])[0][0] - np.nonzero(dhisto > rng[0])[0][0]

            if region_inds[0] > -1:
                print(cregion.region_str)
                print(width[i])
                print(mode[i])
                plt.figure(1234); plt.gcf().clf(); plt.plot(histo)
                plt.figure(1235); plt.gcf().clf(); plt.plot(dhisto)
                plt.show()

        pfns = [os.path.basename(x) for x in fns]
        nprint = nfns
        #nprint = 20 # to test
        print('wafer {}'.format(wafer_id))
        pvals = [width,mode]
        pvals_strs = ['histo width','histo_mode']
        for j in range(len(pvals)):
            print('metric {}'.format(pvals_strs[j]))
            if nprint < nfns:
                print('smallest {}'.format(nprint))
            pval = pvals[j].astype(np.double, copy=True)
            pval[pval==-1] = np.inf
            inds = np.argsort(pval, 0); vals = np.sort(pval, 0)
            for i in range(nprint):
                others = ' '.join([str(x[inds[i,0]]) for x in pvals])
                print('{} {} {}'.format(pfns[inds[i,0]], vals[i,0], others))
            if nprint < nfns:
                print('biggest {}'.format(nprint))
                pval = pvals[j].astype(np.double, copy=True)
                pval[pval==-1] = 0
                inds = np.argsort(pval, 0)[::-1,:]; vals = np.sort(pval, 0)[::-1,:]
                for i in range(nprint):
                    print('{} {}'.format(pfns[inds[i,0]], vals[i,0]))
        # for j in range(len(pvals)):
    #if histo_width_plot:

#for wafer_id, wafer_ind in zip(use_wafer_ids, range(use_nwafers)):

if median_diff_plot:
    # for generating histograms of all 2D stitching residuals.
    # the iteration over all regions is somewhat slow, so dump a dill file instead to be used to generate plots.
    #plt.figure(9876)
    #plt.plot(rh_cbins, rh_counts[:,0]/rh_counts[:,0].sum())
    #plt.plot(rh_cbins, rh_counts[:,1]/rh_counts[:,1].sum())
    #plt.plot(rh_cbins, rho_counts[:,0]/rho_counts[:,0].sum())
    #plt.plot(rh_cbins, rho_counts[:,1]/rho_counts[:,1].sum())
    #plt.plot(rh_cbins, rh_mag_counts/rh_mag_counts.sum())
    #plt.plot(rh_cbins, rho_mag_counts/rho_mag_counts.sum())
    #plt.gca().set_xlabel('residual')
    #plt.gca().set_ylabel('pdf')
    #plt.gca().legend(['inlier x', 'inlier y', 'outlier x', 'outlier y', 'inlier mag', 'outlier mag'])
    #ccbins = rh_cbins * 16
    #plt.figure(9877)
    #plt.plot(ccbins, rh_mag_counts/rh_mag_counts.sum())
    #plt.plot(ccbins, rho_mag_counts/rho_mag_counts.sum())
    #plt.figure(9878)
    #plt.plot(rh_cbins, np.log10(rh_counts[:,0]/rh_counts[:,0].sum()))
    #plt.plot(rh_cbins, np.log10(rh_counts[:,1]/rh_counts[:,1].sum()))
    #plt.plot(rh_cbins, np.log10(rho_counts[:,0]/rho_counts[:,0].sum()))
    #plt.plot(rh_cbins, np.log10(rho_counts[:,1]/rho_counts[:,1].sum()))
    #plt.show()
    dill_fn = 'residual_histos-2D_stitching.dill'
    d = {'rh_cbins':rh_cbins, 'rh_counts':rh_counts, 'rho_counts':rho_counts, 'rh_mag_counts':rh_mag_counts,
         'rho_mag_counts':rho_mag_counts, 'residual_region_cnt':residual_region_cnt,
         'residual_mfov_cnt':residual_mfov_cnt, 'residual_inlier_cnt':residual_inlier_cnt,
         'residual_outlier_cnt':residual_outlier_cnt,}
    with open(dill_fn, 'wb') as f: dill.dump(d, f)

if brightness_plot:
    make_param_plot(adjusts, wafers_nimgs)
    if save_plots:
        plt.savefig(os.path.join(meta_folder, 'region_plots',
            'brightness_slices_{}-order.png'.format('solved' if use_solved_order else 'manifest')), dpi=300)

    # also make histogram of the adjusts
    plt.figure(10)
    hist,bins = np.histogram(adjusts, 50)
    cbins = bins[:-1] + (bins[1]-bins[0])/2
    ax = plt.plot(cbins, hist)
    plt.gca().set_xlabel('brightness adjusts')
    plt.gca().set_ylabel('count')
    if save_plots:
        plt.savefig(os.path.join(meta_folder, 'region_plots', 'brightness_slices_histo.png'), dpi=300)

    if not save_plots: plt.show()

print('JOB FINISHED: run_wafer.py')
print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
