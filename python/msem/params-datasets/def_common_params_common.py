
# these are common routines shared by all the def_common_params for the new acquisition format.
# began to get cumbersome having these defined separately in each def_common_params.

# NOTE: for the Zeiss acquisition format there is no manifest file, so a function that generates
#   region_strs_all needs to be specified to init_region_info with the parameter generate_manifest.
#   see ZF_retinaa_test def_common_params for an example of how to do this.

import os
import glob
import numpy as np
import re

# access function for creating full path locations for a particular wafer id.
# used by most scripts to get locations to data and processed data.
def cget_paths(wafer_id, root_raw, root_align, root_thumb, raw_folders_all, align_folders_all, region_strs_all,
        experiment_subfolder, index_protocol_folders, proc_subdirs, meta_subdirs):
    sb = experiment_subfolder
    experiment_folders = [os.path.join(root_raw, sb, x) for x in raw_folders_all[wafer_id]]
    thumbnail_folders = [os.path.join(root_thumb, sb, x) for x in raw_folders_all[wafer_id]]
    inds = np.arange(len(experiment_folders))
    pinds = np.array(index_protocol_folders[wafer_id])
    sel = (pinds < 0); pinds[sel] = inds[sel]
    protocol_folders = [experiment_folders[x] for x in pinds]
    alignment_folder = os.path.join(root_align, sb, align_folders_all[wafer_id])
    os.makedirs(alignment_folder, exist_ok=True)
    for x in proc_subdirs:
        os.makedirs(os.path.join(alignment_folder, x), exist_ok=True)

    meta_folder = os.path.join(root_align, sb, align_folders_all[0])
    os.makedirs(meta_folder, exist_ok=True)
    for x in meta_subdirs:
        os.makedirs(os.path.join(meta_folder, x), exist_ok=True)

    return experiment_folders, thumbnail_folders, protocol_folders, alignment_folder, \
        meta_folder, region_strs_all[wafer_id]

# <<< utility functions needed for defining and processing region_strs_all

# utility functions to:
#   (1) dump the regions to a meta text file in the order they are in region_strs_all,
#       which is the order they are indexed in all the msem modules.
#   (2) load the regions from the meta text file
#   These are now driven by the manifest file provided on the experimental side
#     when loading msem data in the new acquisition format.
region_str_meta_fn_str = 'wafer{:02d}_region_indexed_order.txt'
def csave_region_strs_to_meta(wafer_id, lget_paths, raw_folders_all, alignment_folders, legacy_zen_format,
        reimage_beg_inds, fullres_dir, manifest_suffixes, _region_strs_all):
    experiment_folders, _, _, alignment_folder, _, region_strs = lget_paths(wafer_id)
    if legacy_zen_format:
        region_strs = _region_strs_all[wafer_id]
        other_fields = [[[] for y in x] for x in region_strs]
        reimage_max = np.array([np.iinfo(np.int64).max])
        #reimage_max = np.array([len(x)+1 for x in region_strs]) # xxx - could make sense for two pass align
    else:
        # for the new data format, load the manifest from the experimental folder and save to alignment folder
        fn = os.path.join(experiment_folders[0], raw_folders_all[wafer_id][0] + manifest_suffixes[wafer_id])
        region_strs, other_fields = cload_region_strs_from_meta(wafer_id, fn, raw_folders_all, alignment_folders)
        reimage_max = np.array(reimage_beg_inds[wafer_id])

    fn = os.path.join(alignment_folder, 'rough_alignment', region_str_meta_fn_str.format(wafer_id))
    with open(fn, 'w') as f:
        for experiment_folder, raw_folder, ind in \
                zip(experiment_folders, raw_folders_all[wafer_id], range(len(experiment_folders))):
            if not legacy_zen_format: experiment_folder = os.path.join(experiment_folder, fullres_dir)
            rns = [x for x in glob.glob(os.path.join(experiment_folder, '*' )) if os.path.isdir(x)]
            for region_str,cother in zip(region_strs[ind],other_fields[ind]):
                tmp = [s for s in rns if region_str == s[-len(region_str):]]
                if len(tmp) != 1:
                    print(tmp)
                    print('found %d folders matching %s in folder %s' % (len(tmp), region_str, experiment_folder))
                    assert(False)
                bn = os.path.basename(tmp[0])
                iord = int(bn.split('_')[0])
                ireimage = np.nonzero(iord < reimage_max)[0]
                if ireimage.size == 0: ireimage = [reimage_max.size]
                f.write(os.path.join(raw_folder, bn) + ' ' + ' '.join(cother) + ' ' + str(ireimage[-1]) + '\n')
                #print(bn)

def cload_region_strs_from_meta(wafer_id, fn, raw_folders_all, alignment_folders):
    alignment_folder = alignment_folders[wafer_id]
    if fn is None:
        fn = os.path.join(alignment_folder, 'rough_alignment', region_str_meta_fn_str.format(wafer_id))
    raw_folders = raw_folders_all[wafer_id]

    region_strs = []; region_substrs = []; raw_folders_ind = 0
    cother_fields = []; other_fields = []
    with open(fn, 'r') as f:
        for line in f:
            sline = line.strip()
            if not sline or sline[0]=='#': continue
            # for the new manifest format, includes limi angles.
            # should not matter for the old format.
            sline = sline.split()
            sline = [x.strip() for x in sline if len(x.strip()) > 0]
            cother = sline[1:]
            # the original manifests are done on windows, so replace windows path seperators.
            # NOTE: this means that backslash is not an allowed filename character.
            sline = sline[0].replace('\\', '/') # replace windows path seperators

            if raw_folders[raw_folders_ind] not in sline:
                raw_folders_ind += 1
                assert(raw_folders[raw_folders_ind] in sline) # bad region strs meta file
                region_strs.append(region_substrs); other_fields.append(cother_fields)
                region_substrs = []; cother_fields = []

            region_substrs.append(sline.split(os.sep)[-1])
            cother_fields.append(cother)
        if region_substrs:
            region_strs.append(region_substrs); other_fields.append(cother_fields)
    return region_strs, other_fields

def _glob_stack_imgs(dn, ext):
    #dn = os.path.join(root_align, experiment_subfolder, align_folders_all[wafer_id])
    assert( os.path.isdir(dn) ) # probably check root_raw
    fns = glob.glob(os.path.join(dn, '*'+ext ))
    fns = [os.path.splitext(os.path.basename(x))[0] for x in fns]
    # the sort is very important, as the order without it is only based on the glob
    fns.sort(key=str.lower)
    return fns

# <<< legacy for backporting automatically creating manifest for the Zen format
# these glob functions are needed because for some experimental folders the slices in the wafer parts
#   are divided into unordered subsets by slice number (yay!)
# NOTE: these glob'ing functions assume that the slice and region number are the same.
def cglob_regions(wafer_id, inds, root_raw, raw_folders_all, experiment_subfolder):
    all_slices = []
    for ind in inds:
        dn = os.path.join(root_raw, experiment_subfolder, raw_folders_all[wafer_id][ind])
        assert( os.path.isdir(dn) ) # probably check root_raw
        rns = [x for x in glob.glob(os.path.join(dn, '*' )) if os.path.isdir(x)]
        fns = [os.path.basename(x).split('_') for x in rns]
        fns = [x[1] for x in fns if len(x) > 1]
        fns = [re.search(r'\d+', x) for x in fns]
        slices = [int(x.group()) for x in fns if x]
        all_slices += slices
    # the sort is is very important, as the order without it is only based on the glob
    all_slices.sort()
    return all_slices
def cglob_regions_exclude(wafer_id, inds, exclude_inds, root_raw, raw_folders_all, experiment_subfolder):
    exclude_slices = cglob_regions(wafer_id, exclude_inds, root_raw, raw_folders_all, experiment_subfolder)
    slices = [x for x in cglob_regions(wafer_id, inds, root_raw, raw_folders_all,
                experiment_subfolder) if x not in exclude_slices]
    return slices
# legacy for backporting automatically creating manifest for the Zen format >>>

def init_region_info(all_wafer_ids, root_raw, lget_paths, raw_folders_all, alignment_folders, legacy_zen_format,
        reimage_beg_inds, fullres_dir, manifest_suffixes, stack_ext, region_strs_all, region_rotations_all,
        region_reimage_index, region_manifest_cnts, region_include_cnts, exclude_regions, generate_manifest=None):
    _region_strs_all = None
    for x in all_wafer_ids:
        if not root_raw:
            _, _, _, alignment_folder, _, _ = lget_paths(x)
            region_strs_all[x] = [_glob_stack_imgs(alignment_folder, stack_ext)]
            continue # all the other inits are unnecessary for image stack alignments

        # read the region strings from the manifest file
        try:
            region_strs_all[x], other_fields = cload_region_strs_from_meta(x, None, raw_folders_all, alignment_folders)
        except OSError:
            print('wafer {} manifest not found, exporting'.format(x))
            if legacy_zen_format and _region_strs_all is None:
                _region_strs_all = generate_manifest()

            # dump the manifest for current wafer, errors indicate missing data or issues with region_strs_all
            csave_region_strs_to_meta(x, lget_paths, raw_folders_all, alignment_folders, legacy_zen_format,
                    reimage_beg_inds, fullres_dir, manifest_suffixes, _region_strs_all)
            # rerun the load now that we know the alignment manifest is there
            region_strs_all[x], other_fields = cload_region_strs_from_meta(x, None, raw_folders_all, alignment_folders)

        region_strs_all_flat = [item for sublist in region_strs_all[x] for item in sublist]
        region_manifest_cnts[x] = len(region_strs_all_flat)
        region_include_cnts[x] = region_manifest_cnts[x] - len(exclude_regions[x])

        if not legacy_zen_format:
            # region_strs (and pther_fields) is a list of lists seperated by experiment folders. flatten.
            other_fields_flat = [item for sublist in other_fields for item in sublist]
            # for the new format, the "microscope alignment" slice angles are also stored in the manifest
            region_rotations_all[x] = [float(x[0]) for x in other_fields_flat]
            region_reimage_index[x] = np.array([float(x[-1]) for x in other_fields_flat])

# utility functions needed for defining and processing region_strs_all >>>

def cimport_exclude_regions(exclude_txt_fn_str, all_wafer_ids, root_align, experiment_subfolder, align_folders_all):
    exclude_regions = [[] for x in range(max(all_wafer_ids)+1)]
    for wafer_id in all_wafer_ids:
        # can not use get_paths because dependencies become circular
        alignment_folder = os.path.join(root_align, experiment_subfolder, align_folders_all[wafer_id])
        fn = os.path.join(alignment_folder, exclude_txt_fn_str.format(wafer_id))
        if os.path.isfile(fn):
            # use fromregex instead of fromfile so that comments can be used, loadtxt requires equal columns
            regexp = r'\s*?(?!#)\s*?(\d+)\s*?'
            data = np.fromregex(fn, regexp, dtype=[('regions', np.uint32)])
            exclude_regions[wafer_id] = np.array([x[0] for x in data])

    return exclude_regions
