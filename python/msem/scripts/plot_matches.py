#!/usr/bin/env python3
"""plot_matches.py

Top level command-line interface for generating plots related to the
  wafer section order solving.

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
import dill
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as nd

from def_common_params import get_paths, exclude_regions, all_wafer_ids
from def_common_params import matches_dill_fn_str
from def_common_params import keypoints_nprocesses, keypoints_dill_fn_str
from def_common_params import order_txt_fn_str

from msem import wafer_solver

parser = argparse.ArgumentParser(description='run_wafer.py')
parser.add_argument('--wafer_id', nargs=1, type=int, default=[1], help='wafer id to plot matches for')
#parser.add_argument('--noise-cutoff', nargs=1, type=float, default=[0.05],
#    help='noise cutoff for percent matches for row max plot')
parser.add_argument('--matches-run-str', nargs='+', type=str, default=['none'],
    help='string to differentiate matches dill files with different parameters')
parser.add_argument('--keypoints-histo', dest='keypoints_histo', action='store_true',
    help='show histogram of the number of keypoints')
parser.add_argument('--keypoints-run-str', nargs=1, type=str, default=['none'],
    help='string to differentiate keypoints dill files with different parameters')
parser.add_argument('--region_inds', nargs='*', type=int, default=[],
    help='optionally show sorted matches rows as plots for these regions')
parser.add_argument('--show-plots', dest='save_plots', action='store_false',
    help='display plots instead of saving as png')
parser.add_argument('--percent-matches-topn', nargs=1, type=int, default=[0],
    help='preprocess percent matches for solver by only keeping top n')
parser.add_argument('--sort-by-solved-order', dest='sort_by_solved_order', action='store_true',
    help='sort percent matches by the solved order')
parser.add_argument('--solved-path', dest='solved_path', action='store_true',
    help='another mode along with solved order to show path length stats')
parser.add_argument('--no-exclude-regions', dest='no_exclude_regions', action='store_true',
    help='ignore the exclude regions stored in def_common_params')
parser.add_argument('--use-exclude-regions', dest='use_exclude_regions', action='store_true',
    help='automatically set region_inds to the excluded regions')
args = parser.parse_args()
args = vars(args)

## params that are set by command line arguments

wafer_id = args['wafer_id'][0]
#ncutoff = args['noise_cutoff'][0]
matches_run_str = args['matches_run_str'][0]
matches_run_strs = args['matches_run_str']
keypoints_histo = args['keypoints_histo']
keypoints_run_str = args['keypoints_run_str'][0]

# preprocess percent matches before order solved to only use top n
percent_matches_topn = args['percent_matches_topn'][0]

# sort the percent matches by an existing solved order
sort_by_solved_order = args['sort_by_solved_order']

# basically a separate mode for showing solved order path length stats
solved_path = args['solved_path']

# purposefully ignore the exclude regions
no_exclude_regions = args['no_exclude_regions']

# only run matches for the current defined exclude regions
use_exclude_regions = args['use_exclude_regions']

# starting at 1 (Zeiss numbering)
region_inds = args['region_inds']

# whether to save or display plots
save_plots = args['save_plots']

## parameters that are determined based on above parameters

_, _, _, alignment_folder, meta_folder, _ = get_paths(wafer_id)

# optionally plot histogram of descriptor counts
if keypoints_histo:

    for i in range(keypoints_nprocesses):
        keypoints_dill_fn = os.path.join(alignment_folder, keypoints_dill_fn_str.format(wafer_id,i,keypoints_run_str))
        print('Loading keypoints dill {}/{}'.format(i,keypoints_nprocesses)); t = time.time()
        with open(keypoints_dill_fn, 'rb') as f: d = dill.load(f)
        print('\tdone in %.4f s' % (time.time() - t, ))
        # xxx - need for this? since we are only getting counts here, do not save all the descriptors.
        #   for larger datasets with lots of keypoints, this runs out of memory.
        # if i==0:
        #     all_descriptors = d['wafer_descriptors']
        #     #all_keypoints = d['wafer_pickleable_keypoints']
        # else:
        #     for k in d['wafer_processed_keypoints']:
        #         all_descriptors[k] = d['wafer_descriptors'][k]
        #         #all_keypoints[k] = d['all_keypoints'][k]
        if i==0:
            descriptors_cnt = np.zeros(len(d['wafer_descriptors']), dtype=np.int64)
        for k in d['wafer_processed_keypoints']:
            descriptors_cnt[k] = d['wafer_descriptors'][k].shape[0]
        del d

    # for generating a prettier plot, dump dill
    dill_fn = 'sift_counts.dill'
    if os.path.isfile(dill_fn):
        with open(dill_fn, 'rb') as f: d = dill.load(f)
    else:
        d = {'descriptors_cnt':[[] for x in range(max(all_wafer_ids)+1)],}
    d['descriptors_cnt'][wafer_id] = descriptors_cnt
    with open(dill_fn, 'wb') as f: dill.dump(d, f)

    # descriptors_cnt = np.array([x.shape[0] for x in all_descriptors])
    histK,bins = np.histogram(descriptors_cnt, 50)
    cbinsK = bins[:-1] + (bins[1]-bins[0])/2

    plt.figure()
    plt.plot(cbinsK, histK)
    plt.title('min {}, max {}'.format(descriptors_cnt.min(), descriptors_cnt.max()))
    plt.show()


matches_dill_fn = os.path.join(alignment_folder, matches_dill_fn_str.format(wafer_id,matches_run_str))
with open(matches_dill_fn, 'rb') as f: d = dill.load(f)
percent_matches = d['percent_matches']

# this is a comparison mode, mostly for debug
if len(matches_run_strs) > 1:
    matches_dill_fn = os.path.join(alignment_folder, matches_dill_fn_str.format(wafer_id,matches_run_strs[1]))
    with open(matches_dill_fn, 'rb') as f: d = dill.load(f)
    percent_matches2 = d['percent_matches']

    # image of the percent keypoints matches matrix used to solve region order (TSP)
    d = percent_matches - percent_matches2
    print('first min {} max {}'.format(percent_matches.min(), percent_matches.max()))
    print('second min {} max {}'.format(percent_matches2.min(), percent_matches2.max()))
    print('diff min {} max {}'.format(d.min(), d.max()))
    plt.figure()
    plt.imshow(1-percent_matches, cmap='gray')
    plt.axis('off')
    plt.figure()
    plt.imshow(1-percent_matches2, cmap='gray')
    plt.axis('off')
    plt.figure()
    plt.imshow(percent_matches - percent_matches2, cmap='gray')
    plt.axis('off')
    plt.show()

order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))
if sort_by_solved_order:
    solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based

if exclude_regions is not None:
    exclude_regions = np.array(exclude_regions[wafer_id], dtype=np.int32) - 1
else:
    exclude_regions = np.zeros((0,), dtype=np.int32)

if use_exclude_regions: region_inds = exclude_regions + 1

nregions = len(region_inds)

if sort_by_solved_order:
    solved_order_plus_exclude = np.concatenate((solved_order, exclude_regions))
    percent_matches = percent_matches[solved_order_plus_exclude,:]
    percent_matches = percent_matches[:,solved_order_plus_exclude]

missing_regions = exclude_regions + 1 if not no_exclude_regions else np.zeros((0,), dtype=np.int32)
percent_matches, _ = wafer_solver.preprocess_percent_matches(percent_matches, missing_regions,
        topn_cols=percent_matches_topn)

if solved_path:
    d = (1 - percent_matches).diagonal(1)
    # get the path length
    print('Solved slice order path length {:.3f}'.format(d.sum()))

    # moving average of percent matches. xxx - param for window size?
    ma = nd.uniform_filter1d(d, 51, axis=0)

    # compare first diagonal with the next few
    ncompares = 5
    dc = np.zeros((d.size,ncompares), dtype=np.double)
    for i in range(ncompares):
        dc[:,i] = percent_matches.diagonal(1) - np.concatenate((percent_matches.diagonal(i+2), np.zeros(i+1)))

    plt.figure()
    plt.plot(ma,'b')
    plt.plot(np.argmin(ma), ma.min(), 'rx')
    plt.figure()
    plt.plot(dc)
    plt.show()

# image of the percent keypoints matches matrix used to solve region order (TSP)
plt.figure()
plt.imshow(1-percent_matches, cmap='gray')
plt.axis('off')

## for debug
#x,y = np.unravel_index(percent_matches.astype(np.int64),percent_matches.shape, order='F')
#plt.figure()
#plt.subplot(1,2,1); plt.imshow(x, cmap='gray')
#plt.subplot(1,2,2); plt.imshow(y, cmap='gray')


# the lowe ratio needs to be adjusted so that percent matches is sparse,
#   but not so sparse that most matches fall into the non-zero distribution of percent-matches for bad-matches
cutoff = 0.0 # should mostly just ignore the zero-ed tril part, higher values not really informative?
pc = percent_matches[percent_matches > cutoff]
#step = 0.025; bins = np.arange(0,1,step); cbins = bins[:-1] + step/2
histA,bins = np.histogram(pc.flat[:], 200)
cbinsA = bins[:-1] + (bins[1]-bins[0])/2

# this is a better indicator, it should not be skewed or bimodal at low percent matches values.
mpc = percent_matches.max(1)
#step = 0.025; bins = np.arange(0,1,step); cbins = bins[:-1] + step/2
histB,bins = np.histogram(mpc[mpc > cutoff], 33)
cbinsB = bins[:-1] + (bins[1]-bins[0])/2

# remove top n-matches for for each row to try to estimate a noise distribution
n=20
spc = np.sort(percent_matches, axis=1)
tmp = spc[:,:-n]
histC,bins = np.histogram(tmp[tmp > cutoff], 33)
cbinsC = bins[:-1] + (bins[1]-bins[0])/2
ncutoff = tmp.max()

histAp = np.log10(histA)
histBp = np.log10(histB)
histCp = np.log10(histC)
histAp = histAp / histAp.max()
histBp = histBp / histBp.max()
histCp = histCp / histCp.max()

try:
    sel = (cbinsA < ncutoff)
    nmode = cbinsA[np.argmax( histA[sel] )]
    nzero = cbinsA[np.nonzero(histA > 0)[0][0]]
except:
    nmode = nzero = 0

plt.figure()
plt.plot(cbinsA,histAp,'b')
plt.plot(cbinsB,histBp,'r.-')
plt.plot(cbinsC,histCp,'g.-')
plt.plot([ncutoff,ncutoff],[-0.05,1.05],'k')
plt.plot([nmode,nmode],[-0.05,1.05],'k--')
tstr = '%d row maxes below noise "cutoff" %.5f\n' % ((mpc < ncutoff).sum(),ncutoff)
tstr += ('%d row maxes below noise mode\n' % ((mpc < nmode).sum(),))
tstr += ('mode %g, left skew %g right skew %g' % (nmode, nmode - nzero, ncutoff - nmode,))
plt.title(tstr)
plt.legend(['full matrix', 'row max'])
plt.ylabel('normalized log count')
plt.xlabel('% matches')

# plot the percent matches matrix using scatter (so large matrices are more visible)
# image of the percent keypoints matches matrix used to solve region order (TSP)
plt.figure()
tmp = percent_matches.copy()
tmp[tmp < ncutoff] = 0
inds = np.transpose(np.nonzero(tmp))
c = tmp[inds[:,0], inds[:,1]]
#c = np.log10(c)
plt.scatter(inds[:,1], inds[:,0], c=c, s=12, marker='.')
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')


if nregions > 0:
    plt.figure()
    all_inds = []
    ntop=10
for i in range(nregions):
    ind = region_inds[i]-1
    if sort_by_solved_order:
        ind = np.nonzero(ind == solved_order_plus_exclude)[0][0]
    cur = np.maximum(percent_matches[ind,:], percent_matches[:,ind])
    print(region_inds[i], cur.max(), np.argmax(cur))
    scur = np.sort(cur)[::-1]
    acur = np.argsort(cur)[::-1]
    print('region_ind {}, ntop {}'.format(ind,ntop))
    print(acur[:ntop])
    #print(np.sort(acur[:ntop]))
    plt.plot(cur if sort_by_solved_order else scur)
    all_inds.append(acur[:ntop])
if nregions > 0:
    print('all unique top inds') # useful for finding exclude slices
    print(np.unique(all_inds))
    plt.legend(region_inds)

if save_plots:
    # to save instead of show figures
    for f in plt.get_fignums():
        plt.figure(f)
        plt.savefig(os.path.join(meta_folder, 'order_plots',
            'wafer%d_matches_figure%d.png' % (wafer_id,f,)), dpi=300)
        plt.savefig(os.path.join(meta_folder, 'order_plots',
            'wafer%d_matches_figure%d.pdf' % (wafer_id,f,)))
else:
    plt.show()
