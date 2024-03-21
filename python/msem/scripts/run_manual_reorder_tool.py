#!/usr/bin/env python3
"""run_manual_reorder_tool.py

Utility that assists in converting wafer orderings and excludes to / from
  a format that is easier to manually edit.

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

# import dill
import argparse
import os
import sys
# import time

try:
    from def_common_params import get_paths, order_txt_fn_str, exclude_txt_fn_str, wafer_region_prefix_str
    from def_common_params import all_wafer_ids, region_manifest_cnts, region_include_cnts
except ModuleNotFoundError:
    print('WARNING: no def_common_params (must specify wafer ids info)')
    order_txt_fn_str = 'wafer{:02d}_region_solved_order.txt'
    exclude_txt_fn_str = 'wafer{:02d}_region_excludes.txt'
    get_paths = all_wafer_ids = region_manifest_cnts = None

try:
    import webknossos as wk
except:
    print('WARNING: no webknossos (needed for --solved_zinds-nml)')

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
parser = argparse.ArgumentParser(description='run_manual_reorder_tool')
parser.add_argument('--wafer_ids', nargs='+', type=int, default=[1],
    help='specify all wafer ids for the dataset')
parser.add_argument('--wafer_manifest_counts', nargs='+', type=int, default=[0],
    help='specify region (slice) counts in manifests for each wafer')
parser.add_argument('--write-order', dest='write_order', action='store_true',
    help='write out the wafer orders (default write manual-reorder.txt')
parser.add_argument('--reorder-file', nargs=1, type=str, default=['manual-reorder.txt'],
    help='specify the name of the manual reorder file (to be edited or edited)')
parser.add_argument('--swap-file', nargs=1, type=str, default=[''],
    help='specify the name of a file with swap to incorporate into reorder file')
parser.add_argument('--sort-excludes', dest='sort_excludes', action='store_true',
    help='in write-order mode, sort the excludes by region')
parser.add_argument('--wafer-region_ids-lookup', nargs='+', type=int, default=[],
    help='convert list consisting of wafer_id, region_ind (base 1)')
parser.add_argument('--region_zinds-lookup', nargs='+', type=int, default=[],
    help='convert list of region zinds (base 0)')
parser.add_argument('--solved_zinds-lookup', nargs='+', type=int, default=[],
    help='convert list of solved zinds (base 0)')
parser.add_argument('--solved_zinds-nml', nargs=2, type=str, default=['', ''],
    help='create dummy nml file (and dataset name) to serve as solved zind lookup')
parser.add_argument('--solved_zinds-nml-pos', nargs=2, type=int, default=[0,0],
    help='specify position for --solved_zinds-nml node placement (2D)')
parser.add_argument('--solved_zinds-nml-voxel_size', nargs=3, type=int, default=[1,1,1],
    help='specify voxel size for the *dummy* nml solved zind lookup')
args = parser.parse_args()
args = vars(args)


### params that are set by command line arguments

# all wafer ids for this dataset (base 1)
wafer_ids = args['wafer_ids']

# the region (slice) counts from the manifests for each wafer in this dataset
wafer_manifest_counts = args['wafer_manifest_counts']

# optionally write the order out
write_order = args['write_order']

# filename for the edited / to be edited manual reordering
reorder_file = args['reorder_file'][0]

# optional filename for to perform a "batch" of swaps.
# format is one swap per line (base 0), existing_index new_index
swap_file = args['swap_file'][0]

# optionally write the order out
sort_excludes = args['sort_excludes']

# special mode for converting indices
wafer_region_ids_lookup = args['wafer_region_ids_lookup']
region_zinds_lookup = args['region_zinds_lookup']
solved_zinds_lookup = args['solved_zinds_lookup']

# special mode for converting all solved indices into a dummy nml.
# can use this as a lookup in webknossos.
solved_zinds_nml = args['solved_zinds_nml']
solved_zinds_nml_pos = args['solved_zinds_nml_pos']
solved_zinds_nml_voxel_size = args['solved_zinds_nml_voxel_size']

## fixed parameters not exposed in def_common_params

exclude_regions = None # normal operation, do not define
# xxx - hacky, for porting the old method, which defined the excludes in def_common_params
#   just copy the exclude_regions assignment out of def_common_params

#delimiter = '-' # stupid because of the -1 exclude delimiters
delimiter = ','

## local functions

def get_order_exclude_txt_fns(wafer_id):
    order_txt_fn = order_txt_fn_str.format(wafer_id)
    exclude_txt_fn = exclude_txt_fn_str.format(wafer_id)
    if get_paths is not None:
        _, _, _, alignment_folder, _, _ = get_paths(wafer_id)
        order_txt_fn = os.path.join(alignment_folder, order_txt_fn)
        exclude_txt_fn = os.path.join(alignment_folder, exclude_txt_fn)
    return order_txt_fn, exclude_txt_fn


### inits based on params

if all_wafer_ids is None: all_wafer_ids = wafer_ids
if region_manifest_cnts is None: region_manifest_cnts = [None] + wafer_manifest_counts

nwafers = len(all_wafer_ids)

if solved_zinds_nml[0]:
    solved_zinds_lookup = range(sum(region_manifest_cnts[1:]))

# special modes for that convert wafer ids / region inds back and forth between region and solved order z indices
if len(wafer_region_ids_lookup) > 0 or len(region_zinds_lookup) > 0 or len(solved_zinds_lookup) > 0:
    cum_manifest_cnts = np.concatenate(([0], np.cumsum(region_manifest_cnts[1:])))
    cum_include_cnts = np.concatenate(([0], np.cumsum(region_include_cnts[1:])))
    cum_exclude_cnts = cum_manifest_cnts - cum_include_cnts

    if solved_zinds_nml[0]:
        annotation = wk.Annotation(name="zind-to-region-ind-lookup",
            dataset_name=solved_zinds_nml[1], voxel_size=solved_zinds_nml_voxel_size)
        group = annotation.skeleton.add_group("solved_zind_to_region")

    if len(wafer_region_ids_lookup) > 0:
        tmp = np.array(wafer_region_ids_lookup).reshape(-1,2)
        wafer_ids = tmp[:,0]
        use_region_inds = tmp[:,1]
    else:
        wafer_ids = None
        use_region_inds = region_zinds_lookup if len(region_zinds_lookup) > 0 else solved_zinds_lookup

    print('{: <25} {: <7} {: <7} {: <7} {: <7} {}'.format('prefix', 'wafer', 'region', 'zind', 'zslvd', 'excl?'))
    for wafer_id, wafer_ind in zip(all_wafer_ids, range(nwafers)):
        if wafer_ids is not None and wafer_id not in wafer_ids: continue
        _, _, _, alignment_folder, _, region_strs = get_paths(wafer_id)
        nregions = sum([len(x) for x in region_strs])
        # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
        region_strs_flat = [item for sublist in region_strs for item in sublist]
        order_txt_fn, exclude_txt_fn = get_order_exclude_txt_fns(wafer_id)
        if os.path.isfile(order_txt_fn):
            solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ')-1 # saved order is 1-based
        else:
            solved_order = None
        if os.path.isfile(exclude_txt_fn) or exclude_regions is not None:
            if exclude_regions is not None:
                # for porting the old method, which defined the excludes in def_common_params
                excludes = np.array(exclude_regions[wafer_id], dtype=np.uint32)-1 # defined excludes are 1-based
            else:
                excludes = np.fromfile(exclude_txt_fn, dtype=np.uint32, sep=' ')-1 # saved excludes are 1-based

        for slice_or_region_ind in use_region_inds:
            if len(region_zinds_lookup) > 0 or len(solved_zinds_lookup) > 0:
                if len(region_zinds_lookup) > 0:
                    wafer_zind = slice_or_region_ind - cum_manifest_cnts[wafer_ind]
                    if wafer_zind >= 0 and wafer_zind < region_manifest_cnts[wafer_id]:
                        region_ind = wafer_zind + 1
                    else:
                        continue
                else:
                    wafer_zind = slice_or_region_ind - cum_include_cnts[wafer_ind]
                    if wafer_zind >= 0 and wafer_zind < region_include_cnts[wafer_id]:
                        region_ind = solved_order[wafer_zind] + 1
                    else:
                        wafer_zind = slice_or_region_ind - cum_include_cnts[-1] - cum_exclude_cnts[wafer_ind]
                        if wafer_zind >= 0 and wafer_zind < excludes.size:
                            region_ind = excludes[wafer_zind] + 1
                        else:
                            continue
            else:
                region_ind = slice_or_region_ind
            region_str = region_strs_flat[region_ind-1]
            zind = cum_manifest_cnts[wafer_ind] + region_ind - 1

            prefix = wafer_region_prefix_str.format(wafer_id, region_str)
            exclude_str = ''
            if solved_order is not None:
                try:
                    solved_zind = np.nonzero(solved_order == region_ind-1)[0][0] + cum_include_cnts[wafer_ind]
                except:
                    solved_zind = np.nonzero(excludes == region_ind-1)[0][0]
                    solved_zind += (cum_include_cnts[-1] + cum_exclude_cnts[wafer_ind])
                    exclude_str = 'excluded'
            else:
                solved_zind = -1
            print('{: <25} {: <7} {: <7} {: <7} {: <7} {}'.format(
                prefix, wafer_id, region_ind, zind, solved_zind, exclude_str))

            if solved_zinds_nml[0]:
                tree = group.add_tree(prefix)
                node_1 = tree.add_node(
                        position=(solved_zinds_nml_pos[0], solved_zinds_nml_pos[1], solved_zind),
                        comment=f'{prefix} {wafer_id=} {region_ind=} {zind=} {solved_zind=} {exclude_str}',
                        )
                # node_2 = tree.add_node(position=(icnodes[i,1,0], icnodes[i,1,1], solved_zind))
                # tree.add_edge(node_1, node_2)
        #for slice_or_region_ind in use_region_inds:
    #for wafer_id, wafer_ind in zip(all_wafer_ids, range(nwafers)):

    if solved_zinds_nml[0]: annotation.save(solved_zinds_nml[0])
    sys.exit(0)
# if len(wafer_region_ids_lookup) > 0 or len(region_zinds_lookup) > 0 or len(solved_zinds_lookup) > 0:

# util either re-creates the wafer solved order / exclude files based on the manual reordering (write_order==True)
#   or creates the manual reorder file based on the wafer solved order / exclude files (write_order==False).

if not write_order:
    loaded_solved_order = [None]*nwafers
    nincluded = [None]*nwafers
    # do two passes. first pass load all the solved_orders and get the included region counts
    for wafer_id, w in zip(all_wafer_ids, range(nwafers)):
        order_txt_fn, exclude_txt_fn = get_order_exclude_txt_fns(wafer_id)
        solved_order = np.fromfile(order_txt_fn, dtype=np.uint32, sep=' ') #-1 # saved order is 1-based
        assert( np.unique(solved_order).size == solved_order.size )
        assert( (solved_order > 0).all() )
        loaded_solved_order[w] = solved_order
        nincluded[w] = solved_order.size

    # optionally do a "batch set" of swaps, typically comes from an excel spreadsheet (semi-manual),
    #   or generated by the local tsp re-ordering method of detecting swaps.
    swaps = np.fromfile(swap_file, dtype=np.int32, sep=' ').reshape(-1,2) if swap_file else None

    # this creates the manual reordering file to be edited from the wafer solved orders and excludes
    print('Converting solved order and excludes to manual edit format')
    all_solved_order = np.empty((0,), dtype=np.int32)
    all_region_indices = np.empty((0,), dtype=np.int32)
    all_indices = np.empty((0,), dtype=np.int32)
    cum_manifest = 0
    cum_nincluded = 0
    cum_nexcluded = sum(nincluded) # excludes are saved at the end of the dataset
    for wafer_id, w in zip(all_wafer_ids, range(nwafers)):
        order_txt_fn, exclude_txt_fn = get_order_exclude_txt_fns(wafer_id)
        print('wafer {} included {}'.format(wafer_id, nincluded[w]))
        next_cum_nincluded = cum_nincluded + nincluded[w]
        csolved_order = loaded_solved_order[w]
        cregion_indices = loaded_solved_order[w] + cum_manifest - 1
        cindices = np.arange(cum_nincluded, next_cum_nincluded, dtype=np.int32)
        if swaps is not None:
            sel = np.logical_and(swaps[:,0] >= cum_nincluded, swaps[:,0] < next_cum_nincluded)
            cswaps = swaps[sel,:] - cum_nincluded
            csolved_order[cswaps[:,0]] = csolved_order[cswaps[:,1]]
            cregion_indices[cswaps[:,0]] = cregion_indices[cswaps[:,1]]
            cindices[cswaps[:,0]] = cindices[cswaps[:,1]]
        all_solved_order = np.concatenate((all_solved_order, csolved_order))
        all_region_indices = np.concatenate((all_region_indices, cregion_indices))
        all_indices = np.concatenate((all_indices, cindices))
        all_solved_order = np.concatenate((all_solved_order, [-1]))
        all_region_indices = np.concatenate((all_region_indices, [-1]))
        all_indices = np.concatenate((all_indices, [-1]))
        if os.path.isfile(exclude_txt_fn) or exclude_regions is not None:
            if exclude_regions is not None:
                # for porting the old method, which defined the excludes in def_common_params
                excludes = np.array(exclude_regions[wafer_id], dtype=np.uint32)
            else:
                excludes = np.fromfile(exclude_txt_fn, dtype=np.uint32, sep=' ')
            assert( (excludes > 0).all() )
            all_solved_order = np.concatenate((all_solved_order, excludes))
            all_region_indices = np.concatenate((all_region_indices, excludes + cum_manifest - 1))
            next_cum_nexcluded = cum_nexcluded + excludes.size
            all_indices = np.concatenate((all_indices, np.arange(cum_nexcluded, next_cum_nexcluded, dtype=np.int32)))
            assert( np.unique(excludes).size == excludes.size )
            assert( nincluded[w] + excludes.size == region_manifest_cnts[wafer_id] )
            print('wafer {} excluded {}'.format(wafer_id, excludes.size))
            cum_nexcluded = next_cum_nexcluded
        else:
            assert( nincluded[w] == region_manifest_cnts[wafer_id] )
        all_solved_order = np.concatenate((all_solved_order, [-1]))
        all_region_indices = np.concatenate((all_region_indices, [-1]))
        all_indices = np.concatenate((all_indices, [-1]))

        cum_nincluded = next_cum_nincluded
        cum_manifest += region_manifest_cnts[wafer_id]
    # for wafer_id, w in zip(all_wafer_ids, range(nwafers)):
    manual_reorder = np.concatenate(
            (all_solved_order[:,None], all_region_indices[:,None], all_indices[:,None]), axis=1)
    np.savetxt(reorder_file, manual_reorder, fmt='%5d {}%6d {}%6d'.format(*([delimiter]*2)))

else: # if not write_order:
    print('Converting manual edit format to solved order and excludes')
    # consult with qbert
    regexp = r'\s*?(?:#.*?\n)?\s*?(-?\d+)\s*?{}\s*?(-?\d+)\s*?{}\s*?(-?\d+)\s*?(?:#.*?\n)?'.format(*([delimiter]*2))
    data = np.fromregex(reorder_file, regexp,
            dtype=[('region', np.int32), ('cumregion', np.int32), ('index', np.int32)])
    manual_reorder = np.concatenate(
            (np.array([x[0] for x in data], dtype=np.int32)[:,None],
             np.array([x[1] for x in data], dtype=np.int32)[:,None],
             np.array([x[2] for x in data], dtype=np.int32)[:,None]), axis=1)
    delims = np.nonzero((manual_reorder == -1).all(1))[0]
    assert( delims.size == 2*nwafers )
    prev_delim = 0
    for wafer_id, w in zip(all_wafer_ids, range(nwafers)):
        solved_order = manual_reorder[prev_delim:delims[2*w],0]
        excludes = manual_reorder[delims[2*w]+1:delims[2*w+1],0]
        print('wafer {} included {} excluded {}'.format(wafer_id, solved_order.size, excludes.size))
        assert( np.unique(solved_order).size == solved_order.size )
        assert( np.unique(excludes).size == excludes.size )
        assert( (solved_order > 0).all() )
        assert( (excludes > 0).all() )
        assert( solved_order.size + excludes.size == region_manifest_cnts[wafer_id] )
        if sort_excludes: excludes = np.sort(excludes)

        order_txt_fn, exclude_txt_fn = get_order_exclude_txt_fns(wafer_id)
        tmp = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=sys.maxsize)
        with open(order_txt_fn, "w") as text_file:
            text_file.write(' ' + np.array2string(solved_order, separator=' ',
                formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
        with open(exclude_txt_fn, "w") as text_file:
            text_file.write(' ' + np.array2string(excludes, separator=' ',
                formatter={'int_kind':'{:4d}'.format}, max_line_width=120).strip('[]'))
        np.set_printoptions(threshold=tmp)

        prev_delim = delims[2*w+1] + 1
    #for wafer_id, w in zip(all_wafer_ids, range(nwafers)):

print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
