#!/usr/bin/env python3

# take multiple solved orders created by run_wafer_solver.py and automatically "assimilate" them into
#   a single coherent solved order. essentially the method uses a majority method based on neighbor edges,
#   and runs this again through a TSP optimal solver.

import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt

import dill
import argparse
import os
import sys
import time

try:
    from def_common_params import get_paths, order_txt_fn_str, exclude_txt_fn_str, region_include_cnts
    from def_common_params import meta_folder, meta_dill_fn_str, all_wafer_ids, total_nwafers
    from msem.wafer_solver import wafer_solver
except ModuleNotFoundError:
    print('WARNING: no def_common_params (must specify meta dill / wafer ids)')

# <<< turn on stack trace for warnings
#import traceback
#import warnings
##import sys
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
parser = argparse.ArgumentParser(description='run_wafer_solver_seq')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
    help='specify wafer(s) for the regions to run')
parser.add_argument('--all-wafers', dest='all_wafers', action='store_true',
    help='instead of specifying wafer id(s), include all wafers for dataset')
parser.add_argument('--meta-dill-file', nargs=1, type=str, default=[''],
    help='override default location for the meta dill file containing the sequences')
parser.add_argument('--write-order', dest='write_order', action='store_true',
    help='write out the majority solved order')
parser.add_argument('--weight-soln', nargs=1, type=str, default=[''],
    help='apply more weight to the solution strings starting with this')
parser.add_argument('--meta-dills-merge', nargs='*', type=str, default=[],
    help='merge solved orders from multiple different meta dill files')
parser.add_argument('--plot-bad-matches', dest='plot_bad_matches', action='store_true',
    help='plot bad matches for order solver sensitivity test')
args = parser.parse_args()
args = vars(args)


### params that are set by command line arguments

# wafer starting at 1, used internally for numbering processed outputs, does not map to any zeiss info
wafer_ids = args['wafer_ids']

# to override the default meta dill file name / location
meta_dill_file = args['meta_dill_file'][0]

# optionally write the order out
write_order = args['write_order']

# apply more weight to solutions that start with this string
weight_soln = args['weight_soln'][0]

# for merging solved orders from different meta dills
meta_dills_merge = args['meta_dills_merge']

# hook to plot order solver sensitivity by artificially introducing bad or missed slices
plot_bad_matches = args['plot_bad_matches']

## fixed parameters not exposed in def_common_params

# how much more weight to apply to weight_soln optionally specified from command line
w_weight_soln = 5

# for plot_bad_matches mode
seq_keys = ["zero", "one", "two", "five", "ten", "fifteen", "twenty", "thirty", "fourty", "fifty", "eighty", "ninety"]
seq_keys_niters = 10

### inits based on params

meta_dill_fn = meta_dill_file if meta_dill_file else os.path.join(meta_folder, meta_dill_fn_str)

# set wafer_ids to contain all wafers, if specified
if args['all_wafers']:
    wafer_ids = list(all_wafer_ids)

nwafers = len(wafer_ids)

with open(meta_dill_fn, 'rb') as f: meta_dict = dill.load(f)

if len(meta_dills_merge) > 0:
    meta_dict['order_solving'] = {}
    for wafer_id, wafer_ind in zip(all_wafer_ids, range(total_nwafers)):
        meta_dict['order_solving'][wafer_id] = {}
    for cmeta_dill in meta_dills_merge:
        bfnseq, extseq = os.path.splitext(cmeta_dill)
        bn = os.path.basename(bfnseq)
        try:
            with open(cmeta_dill, 'rb') as f: cmeta_dict = dill.load(f)
        except:
            cmeta_dill = os.path.join(meta_folder, cmeta_dill)
            with open(cmeta_dill, 'rb') as f: cmeta_dict = dill.load(f)
        for wafer_id, wafer_ind in zip(all_wafer_ids, range(total_nwafers)):
            if wafer_id in cmeta_dict['order_solving']:
                for key in cmeta_dict['order_solving'][wafer_id].keys():
                    cbn = bn + '-' + key
                    meta_dict['order_solving'][wafer_id][cbn] = cmeta_dict['order_solving'][wafer_id][key]

    with open(meta_dill_fn, 'wb') as f: dill.dump(meta_dict, f)
    print('merged meta dills and saved to ' + meta_dill_fn)
    print('exiting')
    print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
    sys.exit(0)

if plot_bad_matches:
    keys = ['order_seqs-sensitivity_0p1', 'order_seqs-sensitivity_0p15', 'order_seqs-sensitivity_0p2', 'order_seqs-sensitivity_0p3', 'order_seqs-sensitivity_0p5', 'order_seqs-solved_rigid']
    nseq_keys = len(keys)
    #nbad_matches = np.zeros((nseq_keys, seq_keys_niters), dtype=np.int64)
    nbad_matches = np.zeros((nseq_keys,), dtype=np.int64)

# iterate the specified wafers
for wafer_id, wafer_ind in zip(wafer_ids, range(nwafers)):
    # create a neighbor adjacency count graph over all the sequences
    adj_count = None
    nsequences = len(meta_dict['order_solving'][wafer_id])
    for k, seq_dict in meta_dict['order_solving'][wafer_id].items():
        sequence = seq_dict['sections']
        nsequence = len(sequence)
        if plot_bad_matches:
            excludes = np.array(seq_dict['excludes'])-1
            new_sequence = []

        if adj_count is None:
            n = max([max(x) for x in sequence]) + 1
            adj_count = sp.csr_matrix((n,n), dtype=np.double)

        cadj_count = sp.dok_matrix((n,n), dtype=bool)
        for seq,i in zip(sequence, range(nsequence)):
            if seq.size > 1:
                cadj_count[seq[:-1], np.roll(seq, -1)[:-1]] = 1
            if plot_bad_matches:
                tmp = np.setdiff1d(seq, excludes)
                if tmp.size > 0:
                    new_sequence.append(tmp)

        cadj_count = cadj_count.tocsr()
        cadj_count = cadj_count + cadj_count.T
        if weight_soln and k.startswith(weight_soln): cadj_count = cadj_count * w_weight_soln
        adj_count += cadj_count
        # print(k)
        # print(adj_count.nnz, adj_count.max(), adj_count.data.min())

        if plot_bad_matches:
            if k in keys:
                ind = keys.index(k)
                x = 1 if wafer_id > 1 else 2
                nbad_matches[ind] = len(new_sequence)-x
    #for k, sequence in meta_dict['order_solving'][wafer_id].items():
    assert(not (adj_count.diagonal() > 0).any()) # you messed up

    if plot_bad_matches:
        if wafer_ind == 0:
            all_nbad_matches = [None]*nwafers
        all_nbad_matches[wafer_ind] = nbad_matches.copy() / region_include_cnts[wafer_id]
        if wafer_ind == nwafers-1:
            print(all_nbad_matches)
            # xxx - meh... kludgey just for a single paper figure
            x1 = [0.1, 0.15, 0.2, 0.3, 0.5, 1.]
            x2 = [17.3, 20, 21.5, 24.7, 38.1, 97.7]
            plt.figure()
            for wafer_ind in range(nwafers):
                plt.plot(x1, all_nbad_matches[wafer_ind], 'x-')
            ax1 = plt.gca()
            ax2 = ax1.twiny()
            ax2.plot(x1, np.zeros(len(x1)), 'k', alpha=0.) # Create a dummy plot
            xticks = np.arange(0.1, 1.01, 0.1)
            ax1.set_xticks(xticks)
            ax2.set_xticks(x1)
            ax2.set_xticklabels(x2)
            ax2.set_xlabel('runtime (m)')
            ax1.set_xlabel('% sampled sift features')
            ax1.set_ylabel('% bad matches')
            plt.gcf().set_size_inches([8,7])
            plt.show()

    _, labels = sp.csgraph.connected_components(adj_count, directed=False)
    # anything that is not part of the largest component can not be assimilated into the order.
    sizes = np.bincount(labels)
    mlabel = labels[np.argmax(sizes)]
    m = sizes.max()
    sel = (labels == mlabel)
    adj_count = adj_count[sel,:][:,sel]
    #adj_count.data /= nsequences
    adj_count.data /= adj_count.max()
    #print(mlabel, m, n, adj_count.shape, adj_count.data.min(), adj_count.max(), nsequences, adj_count.nnz)

    # xxx - could copy tsp "glue" routine locally for testing this script standalone
    print('Computing majority route with TSP solver based on {} sequences'.format(nsequences)); t = time.time()
    incl_solved_order, _ = wafer_solver.normxcorr_tsp_solver(adj_count)
    print('\tdone in %.4f s' % (time.time() - t, ))
    #incl_solved_order = np.arange(m, dtype=np.int64) # for test without msem package

    # created solved order using all slice indices
    solved_order = np.arange(n, dtype=np.int64)[sel][incl_solved_order]
    # create the excluded list (singletons or small components in all sequences)
    excluded = np.nonzero(np.logical_not(sel))[0]

    # add one because solved_order and excluded slices are stored 1-based
    solved_order += 1; excluded += 1
    tmp = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=sys.maxsize)
    if write_order:
        print('Writing out solved order')
        _, _, _, alignment_folder, _, _ = get_paths(wafer_id)
        order_txt_fn = os.path.join(alignment_folder, order_txt_fn_str.format(wafer_id))
        with open(order_txt_fn, "w") as text_file:
            text_file.write(' ' + np.array2string(solved_order, separator=' ',
                formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
        exclude_txt_fn = os.path.join(alignment_folder, exclude_txt_fn_str.format(wafer_id))
        with open(exclude_txt_fn, "w") as text_file:
            text_file.write(' ' + np.array2string(excluded, separator=' ',
                formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
        print('Excluded count == {}'.format(excluded.size))
    else:
        print(solved_order)
        print('Excluded (1-based) indices for wafer id {}, modify def_common_params'.format(wafer_id))
        print(excluded)
    np.set_printoptions(threshold=tmp)

#for wafer_id, wafer_ind in zip(wafer_ids, range(nwafers)):

print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
