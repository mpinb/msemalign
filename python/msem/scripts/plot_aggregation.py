#!/usr/bin/env python3
"""plot_aggregation.py

Top level command-line interface for generating plots related to the
  aggregation, i.e., applying the LSS to rough / fine alignments.

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

# <<< so figures can be saved without X11, uncomment
#import matplotlib as mpl
#mpl.use('Agg')
# so figures can be saved without X11, uncomment >>>
import matplotlib.pyplot as plt

import os
import dill
import argparse
import time

import numpy as np
import scipy.linalg as lin
import scipy.spatial as spatial
import scipy.interpolate as interp
import scipy.ndimage as nd
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing
from sklearn.neighbors import NearestNeighbors

import matplotlib.tri as mtri
import matplotlib.ticker as ticker

from def_common_params import meta_folder, meta_dill_fn_str, fine_grid_xy_spc
from msem.utils import big_img_load, big_img_info
from msem import wafer_solver

# <<< turn on stack trace for warnings
#import traceback
#import warnings
#import sys
#
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    log = file if hasattr(file,'write') else sys.stderr
#    traceback.print_stack(file=log)
#    log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#warnings.showwarning = warn_with_traceback
#warnings.simplefilter('error', UserWarning) # have the warning throw an exception
# turn on stack trace for warnings >>>

## argparse

parser = argparse.ArgumentParser(description='plot_aggregation.py')
parser.add_argument('--run-type', nargs=1, type=str, default=['rough'],
                    choices=['deltas', 'rough', 'fine_outliers', 'fine_xcorrs'], help='the type of plots to generate')
parser.add_argument('--run-str', nargs=1, type=str, default=['none'],
                    help='string to differentiate alignments with different parameters')
parser.add_argument('--show-plots', dest='save_plots', action='store_false',
                    help='display plots instead of saving as png')
parser.add_argument('--mean-wafer-id', nargs=1, type=int, default=[0],
                    help='for the mean wafer values, subtract the mean for this wafer, 0 for off')
parser.add_argument('--grid-plots', dest='grid_plots', action='store_true',
                    help='display/save grid plots in addition to alignment plots')
parser.add_argument('--no-reslice', dest='use_reslice', action='store_false',
                    help='use the old fine agg dill format instead of reslice dill for outliers')
parser.add_argument('--deltas-grids-dump', dest='delta_overview_plots_only', action='store_false',
                    help='specify to dump all delta/warped grids on per-slice deltas (slow)')
parser.add_argument('--no-negate-deltas', dest='negate_deltas', action='store_false',
                    help='do not flip the sign on the deltas (the mapping direction)')
parser.add_argument('--quiver_scale', nargs=1, type=float, default=[1.],
                    help='scale for the delta quivers, 1. is at xy scale')
parser.add_argument('--fine_xcorrs-tsp', dest='fine_xcorrs_tsp', action='store_true',
                    help='special mode for fine xcorrs plots to run local reordering')

args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

# whether to save or display plots
save_plots = args['save_plots']

# specify to plot the grid plots as well as alignment plots
grid_plots = args['grid_plots']

# force usage of the deltas reslice h5 file for loading outlier info
use_reslice = args['use_reslice']

# set this true to only export the delta statistic as a function of slice plots.
# i.e., (not the individiual slice deltas/grids)
delta_overview_plots_only = args['delta_overview_plots_only']

# may want to invert deltas for the plots, depending on which direction they were solved
#   and what the desired plots are. typical solve is dst->src (so inverted mapping)
negate_deltas = args['negate_deltas']

# specify to subtract the mean for this wafer from the other param wafer means, zero to disable.
mean_wafer_id = args['mean_wafer_id'][0]

# scale for delta quivers, > 1. scales arrows down, < 1. scales arrows up
quiver_scl = args['quiver_scale'][0]

# special mode along with fine xcorrs plots that reruns tsp solver to do local re-orderings
fine_xcorrs_tsp = args['fine_xcorrs_tsp']

# this is an identifier so that multiple rough/fine alignemnts can be exported / loaded easily.
run_str = args['run_str'][0]

# run type string is used for some filenames / paths
run_type = args['run_type'][0]

# specify to plot the delta accumulation
deltas_plot = run_type == 'deltas'

# specify to plot the rough accumulation
rough_plot = run_type == 'rough'

# specify to plot the delta accumulation
fine_outliers_plot = run_type == 'fine_outliers'

# specify to plot the delta accumulation
fine_xcorrs_plot = run_type == 'fine_xcorrs'

## fixed parameters not exposed in def_common_params

# this is incase the "center" grid location is not moved to the first grid location
set_ictr = True

# whether to subtract the center delta from the quiver plots or not
center_deltas = False

# size in inches when saving the grid/delta plots
#save_size = [24,24] # allow for good zooming
save_size = [8,8] # to view entire stack

# cutoff to show the distorted grid plots
grid_plot_cutoff = 1e4


## parameters that are determined based on above parameters

any_fine = (deltas_plot or fine_outliers_plot or fine_xcorrs_plot)

base_path = meta_folder
rough_dill_fn = os.path.join(base_path, 'accum_meta_rough.' + run_str + '.dill')
fine_dill_fn = os.path.join(base_path, 'accum_meta_fine.' + run_str + '.dill')
outdir = os.path.join(base_path, 'export_' + run_type, run_str)
os.makedirs(outdir, exist_ok=True)

if any_fine:
    if deltas_plot:
        os.makedirs(os.path.join(outdir, 'warped_grid_stack'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'warped_delta_stack'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'delta_mag_mean_stack'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'delta_mag_std_stack'), exist_ok=True)
    with open(fine_dill_fn, 'rb') as f: load_dict = dill.load(f)
    if load_dict['cum_deltas'] is None:
        load_dict['cum_deltas'] = np.load(fine_dill_fn+'_cum_deltas.npy')
    # force using the reslice file if the fine delta agg file not saved in the old format
    use_reslice = use_reslice or 'cum_valid_comparisons' not in load_dict
else:
    with open(rough_dill_fn, 'rb') as f: load_dict = dill.load(f)

if grid_plots and not any_fine:
    grid_points = load_dict['grid_locations_pixels'].copy()
if deltas_plot:
    grid_points = load_dict['deformation_points'].copy()

if grid_plots or deltas_plot:
    ngrid_points = grid_points.shape[0]
    if ngrid_points < grid_plot_cutoff:
        delaunay_tris = spatial.Delaunay(grid_points)
        grid_simplices = delaunay_tris.simplices

    grid_points = grid_points - grid_points.min(0)
    if set_ictr:
        # this is incase it was not done when the grid locations were created, find point closest to center
        pts = grid_points - grid_points.mean(0)
        ictr = np.argmin((pts*pts).sum(1))
    else:
        ictr = 0
    cgrid_points = grid_points - grid_points[ictr,:]
    #min_cgrid = cgrid_points.min(0) - 5000
    #max_cgrid = cgrid_points.max(0) + 5000
    min_cgrid = cgrid_points.min(0) - 500
    max_cgrid = cgrid_points.max(0) + 500

    if ngrid_points < grid_plot_cutoff:
        vor = spatial.Voronoi(cgrid_points, furthest_site=False)
        grid_voronoi = vor.vertices

meta_dill_fn = os.path.join(meta_folder, meta_dill_fn_str)
with open(meta_dill_fn, 'rb') as f: meta_dict = dill.load(f)
griddist = fine_grid_xy_spc[2]*meta_dict['scale_um_to_pix']

# for the "gradient plots" of the vector fields
fine_filtering_shape_um = [57., 57.]
fine_filtering_depth = 5
fine_filtering_shape_pixels = np.array(fine_filtering_shape_um)*meta_dict['scale_um_to_pix']


# create range that restarts with each wafer
wafers_slices = np.concatenate([np.arange(x) for x in load_dict['wafers_nimgs']])



def make_grid_plot(simplices, deltas=None, plot_voronoi=False, figno=1):
    if deltas is None:
        deltas = np.zeros_like(grid_points)
    # Create the matplotlib Triangulation object
    x = cgrid_points[:,0] + deltas[:,0] - deltas[ictr,0]
    y = cgrid_points[:,1] + deltas[:,1] - deltas[ictr,1]
    triang = mtri.Triangulation(x=x, y=y, triangles=simplices)
    plt.figure(figno); plt.clf()
    plt.triplot(triang, 'k-', lw=0.5)
    if plot_voronoi:
        plt.scatter(grid_voronoi[:,0], grid_voronoi[:,1], c='r', s=12, marker='.')
    plt.xlim([min_cgrid[0], max_cgrid[0]])
    plt.ylim([min_cgrid[1], max_cgrid[1]])
    plt.gca().axis('off')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.axis('off')
    if save_plots:
        plt.gcf().set_size_inches(save_size[0],save_size[1])
    else:
        plt.gcf().set_size_inches(8,6)

def make_delta_plot(deltas=None, figno=2, center=True):
    if deltas is None:
        deltas = np.zeros_like(grid_points)
    d = (deltas - deltas[ictr,:]) if center else deltas
    msz = 14
    plt.figure(figno); plt.clf()
    if ngrid_points < grid_plot_cutoff:
        if save_plots:
            # for some reason the saved quivers behave differently than the displayed quivers.
            plt.quiver(cgrid_points[:,0], cgrid_points[:,1], d[:,0], d[:,1],
                       angles='xy', scale_units='xy', scale=quiver_scl, color='k',
                       linewidth=1, headaxislength=0, headwidth=0, headlength=0)
        else:
            plt.quiver(cgrid_points[:,0], cgrid_points[:,1], d[:,0], d[:,1],
                       angles='xy', scale_units='xy', scale=quiver_scl, color='k')
        plt.scatter(cgrid_points[:,0], cgrid_points[:,1], c='g', s=msz, marker='.')
        plt.plot(0, 0, 'r.')
        plt.gca().axis('off')
        plt.xlim([min_cgrid[0], max_cgrid[0]])
        plt.ylim([min_cgrid[1], max_cgrid[1]])
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.axis('off')
    else:
        p = cgrid_points
        pmin = np.floor(p.min(0)).astype(np.int64); pmax = np.ceil(p.max(0)).astype(np.int64)+1
        prng = pmax - pmin; gw, gh = prng
        #grid_y, grid_x = np.indices((gh, gw), dtype=np.double)
        grid_x, grid_y = np.mgrid[0:gw:griddist, 0:gh:griddist]
        p = p - pmin
        vx = interp.griddata(p, d[:,0], (grid_x, grid_y), fill_value=0., method='linear')
        vy = interp.griddata(p, d[:,1], (grid_x, grid_y), fill_value=0., method='linear')
        #vx = np.log10(np.abs(vx)); vy = np.log10(np.abs(vy))
        plt.subplot(1,2,1)
        plt.imshow(vx)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(vy)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.axis('off')
    if save_plots:
        plt.gcf().set_size_inches(save_size[0],save_size[1])
    else:
        plt.gcf().set_size_inches(8,6)

def make_param_plot(pdict, y, ylabel='angle', default_val=0., dimstrs=['x','y'], alpha=0.5, figno=3, hold=False):
    plt.figure(figno)
    if not hold: plt.clf()
    r = pdict['order_rng']
    x = np.arange(pdict['total_nimgs'], dtype=np.double)
    x[:r[0]] = np.nan; x[r[1]+2:] = np.nan
    y[:r[0]] = np.nan; y[r[1]+2:] = np.nan
    plt.plot(x,y, alpha=0.5); ax1 = plt.gca()
    if hold: return
    plt.legend(dimstrs)

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
            plt.plot(x[sel],yfit, color='k', linestyle='dashed', linewidth=2, alpha=0.5)
            if mean_wafer_id > 0:
                imean = pdict['wafer_ids'].index(mean_wafer_id)
                wmean = y[pdict['cum_wafers_nimgs'][imean]:pdict['cum_wafers_nimgs'][imean+1],d].mean()
            else:
                wmean = 0
            for iw in range(len(pdict['wafer_ids'])):
                wafermean = y[pdict['cum_wafers_nimgs'][iw]:pdict['cum_wafers_nimgs'][iw+1],d].mean()-wmean
                print('Mean %s %s for w%d = %g' % (ylabel, dimstr, pdict['wafer_ids'][iw], wafermean,))

    plt.plot(pdict['wafers_template_order'], default_val, 'r.', alpha=0.5)
    amin = np.nanmin(y); amax = np.nanmax(y)
    for w in (pdict['cum_wafers_nimgs'] - 0.5):
        plt.plot([w,w], [amin,amax], 'r', alpha=0.5)
    for wid,w,cnt in zip(pdict['wafer_ids'], pdict['cum_wafers_nimgs'][:-1], pdict['wafers_nimgs']):
        plt.text(w+cnt/2, amax, 'w%d' % (wid,), color='r')
    ax2 = ax1.twiny()
    y2 = np.empty((pdict['total_nimgs'],), dtype=np.double); y2.fill(default_val)
    y2[:r[0]] = np.nan; y2[r[1]+2:] = np.nan
    ax2.plot(x, y2, 'r', alpha=0.5) # Create a dummy plot
    ax2.set_xlim(ax1.get_xlim())
    def format_wafer_slice(x, pos=None):
        thisind = np.clip(int(x+0.5), 0, pdict['total_nimgs'] - 1)
        return str(wafers_slices[thisind])
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_wafer_slice))
    ax1.set_xlabel('slice index in order')
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel('wafer index in order')


if rough_plot:
    ntotal = len(load_dict['cum_affines'])
    cum_angles = np.zeros((ntotal,), dtype=np.double)
    cum_translations = np.zeros((ntotal,2), dtype=np.double)
    cum_scales = np.ones((ntotal,2), dtype=np.double)
    cum_shears = np.zeros((ntotal,2), dtype=np.double)
    for i in range(ntotal):
        aff = load_dict['cum_affines'][i]
        if aff is None: continue
        cum_translations[i,:] = aff[:2,2]
        # decompose affine in the order of rotation, scale, shear
        u,p = lin.polar(aff[:2,:2])
        # check if affine is a rigid body affine (rot and trans only)
        rigid_affine = np.allclose(p,np.eye(2))
        if rigid_affine:
            cum_angles[i] = np.arctan2(aff[1,0], aff[0,0])
        else:
            assert(np.allclose(u[0,0], u[1,1]) and np.allclose(u[0,1], -u[1,0]))
            cum_angles[i] = np.arctan2(u[1,0], u[0,0])
            cum_scales[i,:] = np.array([p[0,0], p[1,1]])
            cum_shears[i,:] = np.array([p[0,1]/p[0,0], p[1,0]/p[1,1]])
            assert((cum_scales[i,:] > 0).all())

    make_param_plot(load_dict, cum_angles/np.pi*180, figno=3)
    if save_plots:
        plt.gcf().savefig(os.path.join(outdir, 'angles_ordered.png'))
        plt.gcf().savefig(os.path.join(outdir, 'angles_ordered.eps'), format='eps')
    make_param_plot(load_dict, cum_translations, ylabel='translation', figno=4)
    if save_plots:
        plt.gcf().savefig(os.path.join(outdir, 'translations_ordered.png'))
        plt.gcf().savefig(os.path.join(outdir, 'translations_ordered.eps'), format='eps')
    if not rigid_affine:
        make_param_plot(load_dict, cum_scales, ylabel='scale', default_val=1., figno=6)
        if save_plots:
            plt.gcf().savefig(os.path.join(outdir, 'scales_ordered.png'))
            plt.gcf().savefig(os.path.join(outdir, 'scales_ordered.eps'), format='eps')
        make_param_plot(load_dict, cum_shears, ylabel='shear', figno=7)
        if save_plots:
            plt.gcf().savefig(os.path.join(outdir, 'shears_ordered.png'))
            plt.gcf().savefig(os.path.join(outdir, 'shears_ordered.eps'), format='eps')

if grid_plots:
    if ngrid_points < grid_plot_cutoff:
        make_grid_plot(grid_simplices, figno=2)
        plt.title('baseline triangulated grid')
        if save_plots: plt.gcf().savefig(os.path.join(outdir,'baseline_grid.png'))
        plt.show()

    plt.figure(5)
    for i in range(ngrid_points):
        plt.text(cgrid_points[i,0], cgrid_points[i,1], i)
    pmin = cgrid_points.min(0); pmax = cgrid_points.max(0)
    plt.xlim(pmin[0], pmax[0])
    plt.ylim(pmin[1], pmax[1])
    plt.gca().axis('off')
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    if save_plots:
        plt.gcf().set_size_inches(save_size[0],save_size[1])
    else:
        plt.gcf().set_size_inches(16,10)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'baseline_grid_numbers.png'))

if not save_plots: plt.show()

# xxx - move filenames to def_common_params?
fine_reslice_fn = 'fine_reslice.h5'
fine_reslice_fn = os.path.join(meta_folder, fine_reslice_fn)

if any_fine:
    order_rng = load_dict['order_rng']
    wafers_template_order = load_dict['wafers_template_order']
    if not use_reslice:
        # old mode - xxx delete at some point
        nimgs, nnbrs, ncrops = load_dict['cum_valid_comparisons'].shape
    else:
        bfn, ext = os.path.splitext(fine_reslice_fn)
        cfn = bfn + '.0' + ext
        tmp, _ = big_img_info(cfn, dataset='all_fine_outliers')
        nimgs, ncrops, nnbrs = tmp[:3]


if deltas_plot:
    delta_stat_ctr = np.zeros((load_dict['cum_deltas'].shape[0],), dtype=np.double)
    delta_stat_max = np.zeros((load_dict['cum_deltas'].shape[0],2), dtype=np.double)
    delta_stat_std = np.zeros((load_dict['cum_deltas'].shape[0],2), dtype=np.double)
    vmag = np.zeros(load_dict['cum_deltas'].shape[:2], dtype=np.double)
    vstd = np.zeros(load_dict['cum_deltas'].shape[:2], dtype=np.double)
    print('Processed wafers:')
    print(load_dict['wafer_ids'])
    print('Processed wafers nimgs:')
    print(load_dict['wafers_nimgs'])
    print('ngrid points {}'.format(ngrid_points))
    for i in range(order_rng[0], order_rng[1]):
        if not delta_overview_plots_only: print('Solved order ind {}'.format(i))
        edge = 0.025
        cdeltas = (-1. if negate_deltas else 1.)*load_dict['cum_deltas'][i,:,:]
        if not delta_overview_plots_only:
            do_stack_plots = True # for debug
            if do_stack_plots:
                if ngrid_points < grid_plot_cutoff:
                    make_grid_plot(grid_simplices, cdeltas)
                    if save_plots:
                        plt.gcf().savefig(os.path.join(outdir,'warped_grid_stack','%05d_warped_grid.png' % (i,)))
                make_delta_plot(cdeltas, center=center_deltas)
                if save_plots:
                    plt.gcf().savefig(os.path.join(outdir,'warped_delta_stack','%05d_warped_delta.png' % (i,)))
                if not save_plots: plt.show()

            for g in range(ngrid_points):
                sel_pts = np.logical_and(\
                        cgrid_points >= cgrid_points[g,:]-fine_filtering_shape_pixels/2,
                        cgrid_points <= cgrid_points[g,:]+fine_filtering_shape_pixels/2).all(1)
                zmin = i - fine_filtering_depth//2
                zmax = i + fine_filtering_depth//2 + fine_filtering_depth%2
                if zmin < order_rng[0]: zmin = order_rng[0]
                if zmax > order_rng[1]: zmax = order_rng[1]
                tmp = load_dict['cum_deltas'][zmin:zmax,sel_pts,:]
                tmp = np.sqrt((tmp*tmp).sum(1))/meta_dict['scale_um_to_pix']
                vmag[i,g] = tmp.mean(); vstd[i,g] = tmp.std()

            #plt.figure(4321); plt.gcf().clf()
            #plt.scatter(cgrid_points[:,0], cgrid_points[:,1], c=vmag[i,:], vmin=0, s=36, marker='.')
            #plt.gca().invert_yaxis()
            #plt.gca().set_aspect('equal')
            #plt.axis('off')
            #if i==order_rng[0]: plt.colorbar()
            #plt.show()
        #if not delta_overview_plots_only:
        delta_stat_std[i,:] = cdeltas.std(0)
        delta_stat_max[i,:] = cdeltas[np.argmax(np.abs(cdeltas), axis=0), [0,1]]
        delta_stat_ctr[i] = np.sqrt((cdeltas*cdeltas).sum(1)).mean()
    # for i in range(order_rng[0], order_rng[1]):

    if not delta_overview_plots_only:
        vmag_max = vmag.max(); vstd_max = vstd.max()
        for i in range(order_rng[0], order_rng[1]):
            plt.figure(4321); plt.gcf().clf()
            plt.scatter(cgrid_points[:,0], cgrid_points[:,1], c=vmag[i,:], vmin=0, vmax=vmag_max, s=36, marker='.')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.axis('off')
            if i==order_rng[0]: plt.colorbar()
            if save_plots:
                plt.gcf().savefig(os.path.join(outdir,'delta_mag_mean_stack','%05d_mag_mean.png' % (i,)))

            plt.figure(4322); plt.gcf().clf()
            plt.scatter(cgrid_points[:,0], cgrid_points[:,1], c=vstd[i,:], vmin=0, vmax=vstd_max, s=36, marker='.')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.axis('off')
            if i==order_rng[0]: plt.colorbar()
            if save_plots:
                plt.gcf().savefig(os.path.join(outdir,'delta_mag_std_stack','%05d_mag_std.png' % (i,)))
            if not save_plots: plt.show()

    make_param_plot(load_dict, delta_stat_std, ylabel='deltas std', figno=6)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'deltas_std_ordered.png'))
    make_param_plot(load_dict, delta_stat_max, ylabel='deltas max', figno=7)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'deltas_max_ordered.png'))
    make_param_plot(load_dict, delta_stat_ctr, ylabel='deltas mean mag', dimstrs=['mag'], figno=8)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'deltas_ctr_ordered.png'))

    step = 1/16; bins = np.arange(0,1000,step); cbins = bins[:-1] + step/2
    hist,bins = np.histogram(np.sqrt((load_dict['cum_deltas']**2).sum(2)), bins)
    plt.figure(9876)
    plt.plot(cbins/meta_dict['scale_um_to_pix'], hist/hist.sum())
    #plt.plot(cbins, np.log10(hist/hist.sum()))
    #plt.plot(cbins, np.log10(np.cumsum(hist)/hist.sum()))
    # for generating histogram of all deltas.
    # dump a dill file to be used to generate plots.
    dill_fn = 'cum_deltas_histos-ufine.dill'
    d = {'hist':hist, 'cbins':cbins, 'deltas_shape':load_dict['cum_deltas'].shape}
    with open(dill_fn, 'wb') as f: dill.dump(d, f)

    if not save_plots: plt.show()

if fine_outliers_plot:
    # make plot showing the number of outliers, for each direction, crop, skip
    nincluded = load_dict['cum_included']
    noutliers = np.zeros((nimgs,nnbrs,ncrops), dtype=np.int64)

    if not use_reslice:
        # old mode - xxx delete at some point
        for nbr in range(nnbrs):
            for i in range(order_rng[0], order_rng[1]):
                for c in range(ncrops):
                    if load_dict['cum_valid_comparisons'][i,nbr,c]:
                        noutliers[i,nbr,c] = load_dict['cum_outliers'][c][i][nbr].size
    else:
        bfn, ext = os.path.splitext(fine_reslice_fn)

        cfn = bfn + '.0' + ext
        attrs = {'nprocesses':None}
        shp, _ = big_img_info(cfn, dataset='all_fine_outliers', attrs=attrs)
        nblocks = shp[3:5]

        print('Loading fine reslice all_fine_outliers, blks {} {} nprocs {}'.\
            format(nblocks[0], nblocks[1], attrs['nprocesses']))
        t = time.time()
        inds = np.array_split(np.arange(nimgs), attrs['nprocesses'])
        img_ranges = [[x[0],x[-1]+1] for x in inds]
        nproc = len(img_ranges)
        for i in range(nproc):
            cfn = bfn + '.{}'.format(i) + ext
            print(cfn)
            ir = img_ranges[i]
            nir = ir[1] - ir[0]

            for x in range(nblocks[0]):
                for y in range(nblocks[1]):
                    shp, dtype = big_img_info(cfn, dataset='blk_grid_pts_blk_novlp_sel')
                    shp = np.array(shp); shp[0] = nir; shp[1:3] = 1
                    bsel = np.empty(shp, dtype=dtype)
                    custom_slc = np.s_[ir[0]:ir[1],x:x+1,y:y+1,:]
                    big_img_load(cfn, img_blk=bsel, dataset='blk_grid_pts_blk_novlp_sel', custom_slc=custom_slc)
                    bsel = bsel.reshape((nir, -1))

                    shp, dtype = big_img_info(cfn, dataset='all_fine_outliers')
                    shp = np.array(shp); shp[0] = nir; shp[3:5] = 1 #; shp[5] = self.blk_ngrid
                    all_fine_outliers = np.empty(shp, dtype=dtype)
                    custom_slc = np.s_[ir[0]:ir[1],:,:,x:x+1,y:y+1,:]
                    big_img_load(cfn, img_blk=all_fine_outliers, dataset='all_fine_outliers', custom_slc=custom_slc)
                    # xxx - gah, could not think of a way to avoid this loop
                    for j,k in zip(range(ir[0],ir[1]), range(nir)):
                        noutliers[j,:,:] += all_fine_outliers[k,:,:,:,:,:].reshape((ncrops,
                                nnbrs,-1))[:,:,bsel[k,:]].sum(2).transpose((1,0))
                        #noutliers[j,:,:] += bsel[k,:].sum()
        print('\tdone in %.4f s' % (time.time() - t, ))

    percent_inliers = (nincluded[:,:,None] - noutliers)/nincluded[:,:,None]
    if not use_reslice:
        percent_inliers[np.logical_not(load_dict['cum_valid_comparisons'])] = 0
    assert(nnbrs % 2 == 0)
    annbrs = nnbrs//2
    percent_inliers_nbr_dist = (percent_inliers[:,:annbrs,:][:,::-1,:] + percent_inliers[:,annbrs:,:])/2
    percent_inliers_mean_nbrs = percent_inliers.mean(1)

    # plots of the actual percent inliers
    strs = ['+/-'+str(x+1) for x in range(nnbrs//2)]
    for c in range(ncrops):
        make_param_plot(load_dict, percent_inliers_nbr_dist[:,:,c], figno=11+c, ylabel='%inliers crp {}'.format(c),
            dimstrs=strs)
        if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_crop{}.png').format(c))
    crpstrs = [str(x) for x in range(ncrops)]
    make_param_plot(load_dict, percent_inliers_mean_nbrs, figno=10, ylabel='%inliers mean over nbrs', dimstrs=crpstrs)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_mean_over_nbrs.png'))

    # plot without collapsing across direction, useful for finding "gaps", particularly using the direct neighbors
    strs = ['-'+str(x) for x in range(nnbrs//2,0,-1)]+['+'+str(x+1) for x in range(nnbrs//2)]
    slc = list(range(nnbrs//2-1, nnbrs//2+1))
    for c in range(ncrops):
        make_param_plot(load_dict, percent_inliers[:,:,c], figno=210+c, ylabel='%inliers crp {}'.format(c),
            dimstrs=strs)
        print('-1 neighbor, percent_inliers==0:')
        print( np.nonzero(percent_inliers[:,slc[0],c] == 0)[0] )
        print('+1 neighbor, percent_inliers==0:')
        print( np.nonzero(percent_inliers[:,slc[1],c] == 0)[0] )
        if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_crop{}_all.png').format(c))

    # plots of the percent inliers diffs
    for c in range(ncrops-1):
        make_param_plot(load_dict, percent_inliers_nbr_dist[:,:,c+1] - percent_inliers_nbr_dist[:,:,c],
            figno=110+c, ylabel='%inliers crp {} - crp {}'.format(c+1,c),
            dimstrs=['+/-'+str(x+1) for x in range(nnbrs//2)])
        if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_crop{}_diff.png').format(c))

    # plot of mean percent outliers with moving average removed
    cutoff = 0.15
    #sum_crops_percent_inliers_mean_nbrs = percent_inliers_mean_nbrs.sum(1) # why sum over crops?
    sum_crops_percent_inliers_mean_nbrs = percent_inliers_mean_nbrs[:,-1]
    ma = nd.uniform_filter1d(sum_crops_percent_inliers_mean_nbrs, 21, axis=0)
    madiff = ma - sum_crops_percent_inliers_mean_nbrs
    make_param_plot(load_dict, madiff, figno=100, ylabel='%inliers moving avg of mean over nbrs / last crop',
        dimstrs=crpstrs, default_val=cutoff)
    #make_param_plot(load_dict, sum_crops_percent_inliers_mean_nbrs, figno=101, hold=True)
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_mean_over_nbrs_moving_avg.png'))
    print('Top low percent inliers (based on moving diff > {:.5f}):'.format(cutoff))
    inds = np.argsort(madiff,axis=0); i = np.nonzero(madiff[inds] > cutoff)[0]
    offset = 0 # can set this for the crops so the original z index is printed, for example
    #print(i.size)
    if i.size > 0:
        for j in range(madiff.size-1, i[0]-1, -1):
        #for j in range(madiff.size-1, madiff.size-10, -1):
            print('{:d} {:.5f}'.format(inds[j].item() + offset, madiff[inds[j]].item()))

    ## plots of the diffs
    #for c in range(ncrops):
    #    tmp = np.diff(percent_inliers_nbr_dist[:,:,c], axis=1)
    #    make_param_plot(load_dict, tmp, figno=21+c, ylabel='%inliers crp {}'.format(c),
    #        dimstrs=['diff'+str(x+1) for x in range(nnbrs//2 - 1)])
    #    if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_crop{}.png').format(c))
    #tmp = np.diff(percent_inliers_mean_nbrs, axis=1)
    #make_param_plot(load_dict, tmp, figno=20, ylabel='%inliers mean over nbrs',
    #    dimstrs=['diff'+str(x) for x in range(ncrops-1)])
    #if save_plots: plt.gcf().savefig(os.path.join(outdir,'noutliers_mean_over_nbrs.png'))

    if not save_plots: plt.show()

if fine_xcorrs_plot:
    print_excel = False

    if use_reslice:
        print('WARNING: currently only a single process fine reslice supported')
        print('WARNING: reslice MUST be generated as a single block')

    # make plot showing the number of outliers, for each direction, crop, skip
    nincluded = load_dict['cum_included']

    assert( use_reslice ) # need to generate reslice for the xcorrs plot (use single block)
    bfn, ext = os.path.splitext(fine_reslice_fn)
    cfn = bfn + '.0' + ext
    # self.all_fine_outliers[i] = np.zeros((self.range_crops,self.nneighbors,self.ngrid), dtype=bool)
    all_fine_outliers, _ = big_img_load(cfn, dataset='all_fine_outliers')
    # have to remove the singleton "xy blocking dimensions"
    all_fine_outliers = all_fine_outliers.reshape((nimgs,ncrops,nnbrs,-1))

    # self.fine_weights = [None]*self.total_nimgs
    # self.fine_weights[i] = np.zeros((self.nneighbors,self.ngrid), dtype=np.double)
    xcorrs, _ = big_img_load(cfn, dataset='xcorrs')
    xcorrs = xcorrs.reshape((nimgs,nnbrs,-1))
    sel_excluded = np.logical_not(np.isfinite(xcorrs))
    sel_not_included = np.logical_or(sel_excluded, all_fine_outliers[:,-1,:,:])
    xcorrs = np.ma.masked_where(sel_not_included, xcorrs)

    assert(nnbrs % 2 == 0)
    annbrs = nnbrs//2
    xcorr_nbr_dist = (xcorrs[:,:annbrs,:][:,::-1,:] + xcorrs[:,annbrs:,:])/2
    xcorr_mean_grid = xcorr_nbr_dist.mean(axis=2)

    # plot of mean xcorrs at each neighbor dist
    make_param_plot(load_dict, xcorr_mean_grid, figno=11, ylabel='mean xcorr over grid points',
        dimstrs=['+/-'+str(x+1) for x in range(nnbrs//2)])
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'xcorrs_crop{}.png').format(ncrops-1))

    xcorr_nbr_comp = np.zeros((nimgs,annbrs-1), dtype=np.double)
    for ak in range(1,annbrs):
        xcorr_nbr_comp[:,ak-1] = np.maximum(xcorrs[:,4+ak,:] - xcorrs[:,3,:], xcorrs[:,3-ak,:] - xcorrs[:,4,:]).mean(1)
    make_param_plot(load_dict, xcorr_nbr_comp, figno=12, ylabel='xcorr neighbor diff',
        dimstrs=['swap {}'.format(x) for x in range(1,annbrs)])
    if save_plots: plt.gcf().savefig(os.path.join(outdir,'xcorrs_swaps.png'))

    # to identify possible swap locations
    bw = (xcorr_nbr_comp[:,0] > 0)
    #print(bw.sum())
    labels, nlbls = nd.label(bw)
    xcorr_nbr_positives = []
    if nlbls > 0:
        sizes = np.bincount(np.ravel(labels))[1:] # component sizes
        rmv = np.nonzero(sizes < 2)[0] + 1
        #rmv = np.nonzero(sizes < 1)[0] + 1 # does nothing, just so the rmv block does not need to be commented
        if rmv.size > 0:
            sel = np.isin(labels, rmv)
            bw[sel] = 0
            labels[sel] = 0
            labels, nlbls = nd.label(labels > 0)
        print('Number of xcorr neighbor positives {}'.format(nlbls))
        slcs = nd.find_objects(labels)
        xcorr_nbr_positives = [x[0].start for x in slcs]
        print(xcorr_nbr_positives)
        if print_excel:
            print('Indices (for copying to excel):')
            for slc in slcs: print(slc[0].start)
    xcorr_nbr_positives = np.array(xcorr_nbr_positives)

    if fine_xcorrs_tsp:
        print('running tsp solver in local neighborhoods using distance as one-minus-xcorr values')

        z_nnbrs_rad = [15, 5]
        min_inliers = 10
        assert(z_nnbrs_rad[0] > z_nnbrs_rad[1]  + 1 > 2)

        # code mostly copied from wafer_aggregator.py reconcile_deltas_job.
        # assign same variable names.
        n = nimgs
        assert(nnbrs % 2 == 0)
        neighbor_range = nnbrs//2
        neighbor_rng = [x for x in range(-neighbor_range,neighbor_range+1) if x != 0]
        nneighbors = nnbrs

        z_nnbrs = 2*z_nnbrs_rad[0] + 1
        assert(z_nnbrs < n) # z neighborhood can not be bigger than the number of slices
        z_nearest = NearestNeighbors(n_neighbors=z_nnbrs, algorithm='kd_tree').fit(np.arange(n).reshape(-1,1))
        nlocal = z_nnbrs #; nlocal_nbrs = z_nnbrs + nneighbors
        #min_nnbrs = -min(neighbor_rng)
        # second array value is the radius size of the chunk to store in each iteration (ziters).
        # the overlap that is not part of the chunk but part of the neighborhood is discarded.
        nchunks = max([np.round(n/(2*z_nnbrs_rad[1])).astype(np.int64), 1])
        ichunks = np.array_split(np.arange(n), nchunks)
        ziters = len(ichunks)

        print('Computing shortest path through slices z_nnbrs_rad = {} {}'.format(*z_nnbrs_rad)); t = time.time()
        # m = nlocal_nbrs
        m = z_nnbrs
        global_tour = np.zeros(n, dtype=np.int64)
        for iz in range(ziters):
            # distance matrix that the TSP solver will be applied to
            xcd = np.zeros((m,m), dtype=np.double)

            zinds = np.sort(z_nearest.kneighbors(np.array([ichunks[iz].mean(dtype=np.int64)]).reshape((1,1)),
                    return_distance=False).reshape(-1))
            # range to be solved in original slice indices
            zrng = [zinds.min(), zinds.max()+1]
            # range to be stored in original slice indices (not including overlap)
            crng = [ichunks[iz].min(), ichunks[iz].max()+1]
            # range to be stored in adjacency matrix indices (not including overlap)
            # lcrng = [np.nonzero(zinds == crng[0])[0][0] + min_nnbrs,
            #             np.nonzero(zinds == crng[1] - 1)[0][0] + min_nnbrs + 1]
            lcrng = [np.nonzero(zinds == crng[0])[0][0],
                        np.nonzero(zinds == crng[1] - 1)[0][0] + 1]

            # outer loop over slices to be solved
            for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):
                # inner loop over number of neighboring slices to use
                for k,ik in zip(neighbor_rng, range(nneighbors)):
                    j = i+k; jl = il+k
                    # unlike the loops in reconcile_deltas_job, do not include the "trailing" neighbors.
                    if il < 0 or il >= m or jl < 0 or jl >= m: continue

                    # xcorrs = xcorrs.reshape((nimgs,nnbrs,-1))
                    if xcorrs[i,ik,:].count() > min_inliers:
                        xcd[jl,il] = xcorrs[i,ik,:].mean() # worked best in quick tests, others too many FPs
                        #xcd[jl,il] = np.ma.median(xcorrs[i,ik,:])
                        #xcd[jl,il] = xcorrs[i,ik,:].max()
                        #xcd[jl,il] = xcorrs[i,ik,:].min()
                #for k,ik in zip(neighbor_rng, range(nneighbors)):
            #for i,il in zip(range(zrng[0],zrng[1]), range(nlocal)):

            xcd, _ = wafer_solver.preprocess_percent_matches(xcd)
            # xxx - this does not work if the solver tries to swap one of the single specified endpoints
            #iendpoints = [0 if iz > 0 else None, m-1 if iz < ziters-1 else None]
            iendpoints = [0, m-1]
            tour, endpoints = wafer_solver.normxcorr_tsp_solver(xcd, iendpoints=iendpoints)

            global_tour[crng[0]:crng[1]] = tour[lcrng[0]:lcrng[1]] + zrng[0]
            #print(crng, lcrng, zrng)
            #print(global_tour[crng[0]:crng[1]], tour[lcrng[0]:lcrng[1]])
            #print(tour)
            #print()
        # for iz in range(ziters):
        print('\tdone in %.4f s' % (time.time() - t, ))
        xcorr_tsp_positives = np.nonzero(global_tour != np.arange(n))[0]
        print('{} tour locations differing from original z-ordering:'.format(xcorr_tsp_positives.size))
        print(xcorr_tsp_positives)
        if print_excel:
            print('Indices (for copying to excel):')
            for x in xcorr_tsp_positives: print(x)
        print('Proposed order at tour locations differing from original z-ordering:')
        print(global_tour[xcorr_tsp_positives])
        if print_excel:
            print('Indices (for copying to excel):')
            for x in global_tour[xcorr_tsp_positives]: print(x)

        cutoff = 5
        print('Locations that agree within {} in z'.format(cutoff))
        xcorr_nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xcorr_nbr_positives.reshape(-1,1))
        xcdist, xcinds = xcorr_nn.kneighbors(xcorr_tsp_positives.reshape(-1,1), return_distance=True)
        xcinds = np.unique(xcinds[xcdist <= cutoff])
        print(xcorr_nbr_positives[xcinds])
        #print('Indices (for copying to excel):')
        #for x in xcorr_nbr_positives[xcinds]: print(x)

        plt.figure(13)
        plt.plot(global_tour)
        plt.ylabel('new tour')
        plt.xlabel('original z-ordering')
    #if fine_xcorrs_tsp:

    if not save_plots: plt.show()
