#!/usr/bin/env python3
"""run_downsample_wafer.py

Top level command-line interface for downsampling images from a wafer from
  native resolution (4nm) to the pipeline resolution (typically 16 nm).

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

import logging
import pathlib
import shutil
import argparse
import os
import sys
import csv
import time

import multiprocessing as mp
import queue

import numpy as np
import cv2
import tifffile

import skimage.measure as measure

# this script was originally developed as independent from the msem package.
# decided to leave it as "half-in", meaning it still should be able to run in the old "dumb" mode.
try:
    #from msem.utils import isInt_str
    from def_common_params import all_wafer_ids, raw_folders_all, get_paths, legacy_zen_format
    from def_common_params import root_raw, experiment_subfolder
    from msem import zimages
    msem_imported = True
except:
    msem_imported = False

try:
    import mkl
    mkl_imported=True
except:
    mkl_imported=False
    print('WARNING: mkl module unavailable, can not mkl.set_num_threads')

# https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def isInt_str(v):
    v = str(v).strip()
    return v=='0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()

#logging
#    FORMAT = '%(asctime)-15s %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger.info('starting')


maxsize_queue = 6400 # xxx - parameter?
read_queue = mp.Queue(maxsize_queue); proc_queue = mp.Queue(maxsize_queue)
write_queue = mp.Queue(maxsize_queue); done_queue = mp.Queue(maxsize_queue)

# this is the timeout for multiprocessing queues before checking for dead workers.
# did not see a strong need for this to be drive from command line.
queue_timeout = zimages.queue_timeout if msem_imported else 180 # in seconds

def downsample_job_read(ind, verbose):
    #logger.info("read job %d started", ind)
    img_shape = img_dtype = None
    q = read_queue.get()
    while q is not None:
        ifn, ofn = q
        _, ext = os.path.splitext(ifn); ext = ext.lower()
        try:
            if ext == '.tif' or ext == '.tiff':
                img = tifffile.imread(ifn)
            else:
                img = cv2.imread(ifn, cv2.IMREAD_GRAYSCALE)
        except:
            img = None
        if img is not None:
            if img.ndim > 2: img = img[:,:,0]
            if img.ndim < 2:
                print('BAD IMG NDIM {}: {}'.format(img.ndim, ifn))
                img = None
            elif img_shape is not None:
                if not all([x==y for x,y in zip(img.shape, img_shape)]):
                    print('BAD IMG SHAPE {}: {}'.format(img.shape, ifn))
                    img = None
                elif not np.issubdtype(img.dtype, img_dtype):
                    print('BAD IMG DTYPE {}: {}'.format(img.dtype, ifn))
                    img = None
            if img is not None:
                img_shape = img.shape; img_dtype = img.dtype
        else:
            print('BAD IMG: {}'.format(ifn))
        bn, ext = os.path.splitext(ofn); ext = ext.lower()
        proc_queue.put((ofn, img))
        q = read_queue.get()
    # add to queue to shut all the proc workers down when finished
    for x in range(nproc_workers):
        proc_queue.put(None)
    #logger.info("read job %d done", ind)

def downsample_job_proc(ind, sample_factor, use_block_reduce, zero_border, verbose):
    bad_img_list = []
    ds_img_shape = img_dtype = None
    #logger.info("proc job %d started", ind)
    q = proc_queue.get()
    while q is not None:
        ofn, img = q
        if img is None and ds_img_shape is None:
            bad_img_list.append(q)
        else:
            if img is None:
                ds_img = np.zeros(ds_img_shape, dtype=img_dtype)
            else:
                if zero_border[0] > 0: img[:zero_border[0],:] = 0
                if zero_border[1] > 0: img[-zero_border[1]:,:] = 0
                if zero_border[2] > 0: img[:,:zero_border[2]] = 0
                if zero_border[3] > 0: img[:,-zero_border[3]:] = 0
                if not use_block_reduce:
                    ds_img = cv2.resize(img, None, 0, 1/sample_factor, 1/sample_factor, interpolation=cv2.INTER_AREA)
                else:
                    pad = (sample_factor - np.array(img.shape) % sample_factor) % sample_factor
                    ds_img = measure.block_reduce(np.pad(img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                 block_size=(sample_factor, sample_factor), func=np.median).astype(img.dtype)
                ds_img_shape = ds_img.shape; img_dtype = img.dtype
            write_queue.put((ofn, ds_img))
        q = bad_img_list.pop() if ds_img_shape is not None and len(bad_img_list) > 0 else proc_queue.get()
    # add to queue to shut all the write workers down when finished
    for x in range(nwrite_workers):
        write_queue.put(None)
    #logger.info("proc job %d done", ind)

def downsample_job_write(ind, verbose):
    #logger.info("write job %d started", ind)
    q = write_queue.get()
    while q is not None:
        ofn, ds_img = q
        cv2.imwrite(ofn, ds_img)
        done_queue.put(None)
        q = write_queue.get()
    #logger.info("write job %d done", ind)


def modify_metadata_file(metadata_file, output_dir, sample_factor):
    """ read metadata.txt, modify and write to output mfov dir(s)"""
    with open(metadata_file.as_posix(), 'r') as mdfile:
        md_data = list(csv.reader(mdfile))
    #skip 6 lines
    line = md_data[7][0]
    pixelsize = float(line.split('\t')[2][:-2])
    line = line[:-7]
    pixelsize = pixelsize * sample_factor
    line = line + "{:4.3f}".format(pixelsize) + "nm"
    md_data[7][0] = line
    outputfile = output_dir.joinpath(metadata_file.name)
    with open(outputfile.as_posix(), 'w') as mdfile:
        writer = csv.writer(mdfile)
        writer.writerows(md_data)


def modify_image_coord_file(full_img_coord_file, output_dir, sample_factor):
    """ modify region_stage_coords.csv and save to output region dir"""
    modified_data = []
    with open(full_img_coord_file.as_posix(), 'r') as mdfile:
        md_data = csv.reader(mdfile, delimiter='\t')
        for row in md_data:
            filename = row[0]
            coord1 = float(row[1]) / sample_factor
            coord2 = float(row[2]) / sample_factor
            coord3 = 0
            modified_data.append((filename, coord1, coord2, coord3))
    outputfile = output_dir.joinpath(full_img_coord_file.name)
    with open(outputfile.as_posix(), 'w') as mdfile:
        writer = csv.writer(mdfile, delimiter='\t')
        writer.writerows(modified_data)



def handle_mfov(mfovdir, outputdir, sample_factor, verbose=True, pretend=False,
                thumbnails_only=False, use_block_reduce=False, zero_border=[0,0,0,0]):
    thumbstr = 'thumbnail_' if thumbnails_only else ''
    """copy/convert mfov directory"""
    dscnt = 0
    for mfovobj in mfovdir.iterdir():
        #if verbose:
        #    logger.debug("processing file %s", mfovobj)
        #    logger.debug("processing file suffix: %s", mfovobj.suffix)
        if mfovobj.suffix == ".bmp" or mfovobj.suffix[:4] == ".tif":
            if verbose:
                logger.debug("processing image file %s", mfovobj)
            if mfovobj.name.startswith("thumbnail"):
                if verbose:
                    logger.debug("ignoring thumbnail file %s", mfovobj)
            else:
                outputfile = outputdir.joinpath(thumbstr + mfovobj.name)
                if not pretend:
                    read_queue.put((mfovobj.as_posix(), outputfile.as_posix()))
                    dscnt += 1
        elif mfovobj.name == "metadata.txt":
            if verbose:
                logger.debug("processing %s", mfovobj)
            if not pretend and not thumbnails_only:
                modify_metadata_file(mfovobj, outputdir, sample_factor)
        elif mfovobj.name == "image_coordinates.txt":
            if verbose:
                logger.debug("processing %s", mfovobj)
            if not pretend and not thumbnails_only:
                modify_image_coord_file(mfovobj, outputdir, sample_factor)
        else:
            if verbose:
                logger.warning("unexpected file %s", mfovobj)
        if verbose:
            logger.debug("---")
    return dscnt


def handle_region(regiondir, outputdir, sample_factor, verbose=False, pretend=False,
                  thumbnails_only=False, use_block_reduce=False, zero_border=[0,0,0,0], read_only=False):
    """copy/convert region directory"""
    tmp = regiondir.name.split('_')
    slice_name = tmp[1] if len(tmp) > 1 else tmp[0]
    fn_stitched_coords = slice_name + '_stitched_imagepositions.txt'
    dscnt = 0

    for regionobj in regiondir.iterdir():
        if regionobj.is_dir():
            if len(regionobj.name) != (6 if legacy_zen_format else 3) or not isInt_str(regionobj.name): continue
            if verbose:
                logger.debug("processing mFoV dir %s", regionobj)
            targetdir = outputdir.joinpath(regionobj.name)
            if not pretend and not read_only:
                os.makedirs(targetdir.as_posix(), exist_ok=True)
            dscnt += handle_mfov(regionobj, targetdir, sample_factor, thumbnails_only=thumbnails_only,
                    verbose=verbose, pretend=pretend, use_block_reduce=use_block_reduce,
                    zero_border=zero_border)
        elif regionobj.name == "region_stage_coords.csv":
            if verbose:
                logger.debug("processing %s", regionobj)
            if not pretend and not thumbnails_only:
                shutil.copy2(regionobj, outputdir.joinpath(regionobj.name))
        elif regionobj.name == "full_image_coordinates.txt":
            if verbose:
                logger.debug("processing %s", regionobj)
            if not pretend and not thumbnails_only:
                modify_image_coord_file(regionobj, outputdir, sample_factor)
        elif regionobj.name == fn_stitched_coords:
            if verbose:
                logger.debug("processing %s", regionobj)
            if not pretend and not thumbnails_only:
                modify_image_coord_file(regionobj, outputdir, sample_factor)
        elif regionobj.name == "protocol.txt":
            if verbose:
                logger.debug("processing %s", regionobj)
            if not pretend and not thumbnails_only:
                shutil.copy2(regionobj, outputdir.joinpath(regionobj.name))
        elif regionobj.suffix == ".txt":
            if verbose:
                logger.debug("copying text file %s", regionobj)
            if not pretend and not thumbnails_only:
                shutil.copy2(regionobj, outputdir)
    return dscnt

if __name__ == "__main__":

    #argparse
    parser = argparse.ArgumentParser(description='downsample dataset')
    parser.add_argument('datadir', help='top level directory of input dataset')
    parser.add_argument('outputdir', help='output directory')
    parser.add_argument('samplefactor', nargs='+',
                        help='downsampling factor (per dimension)', type=int)
    parser.add_argument('--pretend', action='store_true',
                        default=False, help='for testing, do not actually write output')
    parser.add_argument('--verbose', action='store_true',
                        default=False, help='verbose output')
    parser.add_argument('--region-range', nargs=2, default=None,
                        help='python-style range for regions to process', type=int)
    parser.add_argument('--inter-area', action='store_true', default=False,
                        help='use cv2 inter_area (pixel averaging) instead of default median block reduce')
    parser.add_argument('--nworkers', nargs=4, type=int, default=[1,1,1,1],
                        help='number of read/process/write/mkl workers repsectively')
    parser.add_argument('--thumbnails-only', action='store_true', default=False,
                        help='only save downsampled images and not other files, save using thumbnail name')
    parser.add_argument('--no-manifest', action='store_true', default=False,
                        help='the old dumb mode that does not care about the manifest')
    parser.add_argument('--zero-border', nargs=4, type=int, default=[0,0,0,0],
                        help='zero out this number of border lines, in order top,bottom,left,right')
    # parser.add_argument('--legacy-zen', action='store_true', default=False,
    #                     help='use the legacy zen acquisition format')

    args = vars(parser.parse_args())
    verbose = args["verbose"]
    pretend = args["pretend"]
    sample_factor = args["samplefactor"][0]
    use_block_reduce = not args['inter_area']
    zero_border = args['zero_border']
    thumbnails_only = args['thumbnails_only']
    no_manifest = args['no_manifest']
    nworkers = args['nworkers']
    #legacy_zen = args["legacy_zen"]

    nread_workers = nworkers[0]
    nwrite_workers = nworkers[2]
    nproc_workers = nworkers[1]
    read_only = (nproc_workers < 1)
    mkl.set_num_threads(nworkers[3])

    datapath = pathlib.PosixPath(args['datadir'])

    # this script was developed originally independent of the msem package.
    use_wafer_id = -1
    if msem_imported:
        topdir = os.path.join(root_raw, experiment_subfolder)
        if topdir in args['datadir']:
            waferdir = args['datadir'][len(topdir)+1:]
            if waferdir.endswith(os.sep): waferdir = waferdir[:-1]
            for wafer_id in all_wafer_ids:
                tmp = [p[:-1] if p.endswith(os.sep) else p for p in raw_folders_all[wafer_id]]
                tmp = [p[1:] if p.startswith(os.sep) else p for p in tmp]
                if waferdir in tmp:
                    use_wafer_id = wafer_id
                    break
            if use_wafer_id > -1:
                _, _, _, _, _, region_strs = get_paths(use_wafer_id)
                # region_strs is a list of lists unfortunately, seperated by experiment folders. flatten.
                region_fns = [item for sublist in region_strs for item in sublist]
                region_fns = [os.path.split(x)[1] for x in region_fns]
    assert( use_wafer_id > -1 or no_manifest ) # can not find waferdir in manifest, use --no-manifest for old mode

    if not legacy_zen_format: datapath /= 'fullres'
    output_base_path = pathlib.PosixPath(args['outputdir'])

    region_range = args['region_range']

    # most likely do not need czifiles
    copy_czifiles = False

    if verbose:
        logger.debug("data path: %s", datapath)
        logger.debug("sample factor: %s", sample_factor)

    if not datapath.is_absolute():
        datapath = datapath.absolute()

    if not output_base_path.is_absolute():
        output_base_path = output_base_path.absolute()

    dataset_dirname = datapath.name

    if not output_base_path.is_dir() and not read_only:
        logger.info("creating dir %s", output_base_path)
        os.makedirs(output_base_path, exist_ok=True)

    if dataset_dirname == output_base_path.parts[-1]:
        # do not make another path dataset_dirname if it was already specified on command line
        output_path = output_base_path
    else:
        output_path = output_base_path.joinpath(dataset_dirname)
    if not output_path.is_dir() and not read_only:
        logger.info("creating dir %s", output_path)
        os.makedirs(output_path, exist_ok=True)

    # start up the downsample worker processes
    read_workers = [None]*nread_workers
    for x in range(nread_workers):
        read_workers[x] = mp.Process(target=downsample_job_read, daemon=True, args=(x,verbose,))
        read_workers[x].start()
    proc_workers = [None]*nproc_workers
    for x in range(nproc_workers):
        proc_workers[x] = mp.Process(target=downsample_job_proc, daemon=True,
                    args=(x, sample_factor, use_block_reduce, zero_border, verbose))
        proc_workers[x].start()
    write_workers = [None]*nwrite_workers
    for x in range(nwrite_workers):
        write_workers[x] = mp.Process(target=downsample_job_write, daemon=True, args=(x,verbose,))
        write_workers[x].start()

    t = time.time(); dscnt = 0
    for fsobj in sorted(datapath.iterdir()):
        if verbose:
            logger.info(fsobj)
        if fsobj.suffix == ".czi" and copy_czifiles:
            #logger.debug("copying %s to %s", fsobj, output_path)
            if not pretend:
                shutil.copy2(fsobj, output_path.joinpath(fsobj.name))
        elif fsobj.is_dir():
            str_region_num = fsobj.name.split('_')[0]
            if not isInt_str(str_region_num): continue
            int_region_num = int(str_region_num)
            if region_range is not None:
                if use_wafer_id > -1:
                    # new mode, use manifest indexing, region_strs is zero-based after it's loaded
                    if fsobj.name not in region_fns[region_range[0]-1:region_range[1]-1]: continue
                elif (int_region_num < region_range[0] or int_region_num >= region_range[1]):
                    # old mode, use sorted indexing
                    continue

            logger.info("processing region %s", fsobj)
            target_region_dir = output_path.joinpath(fsobj.name)
            if not pretend and not read_only:
                os.makedirs(target_region_dir.as_posix(), exist_ok=True)
            dscnt += handle_region(fsobj, target_region_dir, sample_factor, thumbnails_only=thumbnails_only,
                         verbose=verbose, pretend=pretend, use_block_reduce=use_block_reduce,
                         zero_border=zero_border, read_only=read_only)

    logger.info("waiting for workers to complete %d downsamples", dscnt)
    nupdate = 100
    dt = time.time()
    #for cnt in range(dscnt):
    cnt = 0
    while cnt < dscnt:
        try:
            if nproc_workers < 1:
                # mechanism to read-only without actually downsampling,
                #   specify zero for number of proc and write workers
                proc_queue.get(block=True, timeout=queue_timeout)
            else:
                done_queue.get(block=True, timeout=queue_timeout)
        except queue.Empty:
            for x in range(nread_workers):
                if not read_workers[x].is_alive():
                    print('read worker {} is dead and not finished'.format(x))
                    assert(False) # a worker exitted with an error or was killed without finishing
            for x in range(nproc_workers):
                if not proc_workers[x].is_alive():
                    print('proc worker {} is dead and not finished'.format(x))
                    assert(False) # a worker exitted with an error or was killed without finishing
            for x in range(nwrite_workers):
                if not write_workers[x].is_alive():
                    print('write worker {} is dead and not finished'.format(x))
                    assert(False) # a worker exitted with an error or was killed without finishing
            continue

        if (cnt > 0) and (cnt % nupdate==0):
            logger.info('%d downsampled in %.3f s' % (nupdate,time.time()-dt,))
            logger.info('remaining: %d read, %d proc, %d write',
                         read_queue.qsize(), proc_queue.qsize(), write_queue.qsize())
            dt = time.time()
        cnt += 1

    # signal to shut down the read workers, this cascades to the proc and write workers.
    # nothing further is added to the done queue.
    for x in range(nread_workers):
        read_queue.put(None)

    [x.join() for x in read_workers]
    [x.join() for x in proc_workers]
    [x.join() for x in write_workers]
    [x.close() for x in read_workers]
    [x.close() for x in proc_workers]
    [x.close() for x in write_workers]

    logger.info('finished in %.3f', time.time()-t)
    print('Twas brillig, and the slithy toves') # with --check-msg swarm reports slurm failure without message
