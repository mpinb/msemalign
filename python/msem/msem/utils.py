"""utils.py

Miscellaneous dumping ground. Buyer beware.

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
from pathlib import Path
import time
import re
from enum import IntEnum
import warnings
import dill
import fcntl
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw
import tifffile

import math
import sympy

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import scipy.ndimage as nd
import scipy.sparse as sp
import scipy.spatial as spatial
import scipy.spatial.distance as scidist
import scipy.stats as st
import scipy.optimize as opt
from scipy import signal

import networkx as nx

import itertools

# from sklearnex import patch_sklearn
# patch_sklearn()
#from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
# xxx - sometimes intel version completely crashes or hangs, not worth the modest speedup
#from sklearnex.neighbors import NearestNeighbors

import hdf5plugin
import h5py

from contextlib import nullcontext

from scipy import fft as scipy_fft
try:
    import pyfftw
    _pyfftw_imported = True
except:
    print('WARNING: pyfftw unavailable, needed for one method for template matching with cross-correlations')
    _pyfftw_imported = False

try:
    from mkl_fft import _scipy_fft_backend as mkl_fft
    _mkl_fft_imported = True
except:
    print('WARNING: mkl fft unavailable, needed for one method for template matching with cross-correlations')
    _mkl_fft_imported = False

try:
    import mkl
    _mkl_imported=True
except:
    print('WARNING: mkl module unavailable, can not mkl.set_num_threads')
    _mkl_imported=False

# try:
#     import cupy as cp
# except:
#     pass

# <<< helper functions to retrieve environment variables to set some msem runtime parameters globally

def get_num_threads(default=8, set_mkl=True):
    """ read num_threads from environment variable MSEM_NUM_THREADS
        otherwise set default value"""
    try:
        num_threads: int = int(os.environ['MSEM_NUM_THREADS'])
    except:
        num_threads: int = default
    assert 0 < num_threads < 1024
    if set_mkl and _mkl_imported:
        mkl.set_num_threads(num_threads)
    return num_threads

def get_gpu_index(default=0):
    """ read gpu_index from environment variable MSEM_GPU_INDEX
        otherwise set default value"""
    try:
        gpu_index: int = int(os.environ['MSEM_GPU_INDEX'])
    except:
        gpu_index: int = default
    assert 0 <= gpu_index < 8
    return gpu_index

# this ideally is a memory mapped locations that can allow for some
#   file loads to be sped up (particularly useful on gpfs).
# was originally intended for large tiff loads, but this was depreceated
#   after moving to the hdf5 intermediary.
def get_cache_dir(default=None):
    """ read cache_dir from environment variable MSEM_CACHE_DIR
        otherwise set default value"""
    try:
        cache_dir: str = str(os.environ['MSEM_CACHE_DIR'])
    except:
        shmvar = 'JOB_SHMTMPDIR' # special user/job location in shm on soma cluster
        if default is None:
            default = os.environ[shmvar] if shmvar in os.environ else '/dev/shm'
        cache_dir: str = default
    return cache_dir

class FFT_types(IntEnum):
    scipy_fft = 0; numpy_fft = 1; pyfftw_fft = 2; cupy_fft = 3; rcc_xcorr = 4

    # SO how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch/43634746
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def get_fft_types(default=[int(FFT_types.pyfftw_fft), int(FFT_types.pyfftw_fft)]):
    """ read fft type to use from environment variable MSEM_FFT_TYPE
        types are expected comma separated (no spaces), first is CPU fft type, second GPU fft type
        otherwise set default values"""
    try:
        fft_types: list[int] = [int(x) for x in os.environ['MSEM_FFT_TYPE'].split(',')]
    except:
        fft_types: list[int] = default
    # NOTE: can not do query cuda devices in a global init because cuda can not be forked,
    #   meaning any processes that try to use cuda will fail. another option is 'spawn'
    #   but all the conditional code required for this is a nightmare.
    # Querying for cuda devices has been moved into zimages.query_cuda_devices and must be explicitly called.
    assert( all([FFT_types.has_value(x) for x in fft_types]) ) # bad fft type in env var MSEM_FFT_TYPE
    return fft_types

class FFT_backend_types(IntEnum):
    none = 0; mkl = 1; fftw = 2; cupy = 3

    # SO how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch/43634746
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

# have this in addition to fft types because for some of the "types" different backends are possible.
# xxx - likely possible to clean up this "secondary" option enumeration...
def get_fft_backend(default=FFT_backend_types.none):
    """ read fft type to use from environment variable MSEM_FFT_BACKEND
        otherwise set default values"""
    try:
        fft_backend: int = int(os.environ['MSEM_FFT_BACKEND'])
    except:
        fft_backend: int = int(default)

    assert( FFT_backend_types.has_value(fft_backend) ) # bad interp type in env var MSEM_FFT_BACKEND
    return fft_backend

class block_reduce_type(IntEnum):
    mean = 0; median = 1

    # SO how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch/43634746
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def get_block_reduce_func(default=block_reduce_type.mean):
    """ read block reduce type to use from environment variable MSEM_BLKRDC_TYPE
        return the numpy function to use for the block reduce
        otherwise set default value
        this is used for all downsamplings that use the block reduce (pixel averaging) method"""
    try:
        blkrdc_type: int = int(os.environ['MSEM_BLKRDC_TYPE'])
    except:
        blkrdc_type: int = int(default)
    assert( block_reduce_type.has_value(blkrdc_type) ) # bad value for env var MSEM_BLKRDC_TYPE

    if blkrdc_type == block_reduce_type.mean:
        blkrdc_func = np.mean
    elif blkrdc_type == block_reduce_type.median:
        blkrdc_func = np.median

    return blkrdc_func

class delta_interp_methods(IntEnum):
    MLS = 0; TPS = 1

    # SO how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch/43634746
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def get_delta_interp_method(default=delta_interp_methods.MLS):
    """ read fft type to use from environment variable MSEM_INTERP
        otherwise set default value"""
    try:
        delta_interp_method: int = int(os.environ['MSEM_INTERP'])
    except:
        delta_interp_method: int = int(default)

    assert( delta_interp_methods.has_value(delta_interp_method) ) # bad interp type in env var MSEM_INTERP
    return delta_interp_method

def get_process_uuid(default=str(uuid.uuid4())):
    """ read process_uuid from environment variable MSEM_UUID
        otherwise set default value"""
    try:
        process_uuid: str = str(os.environ['MSEM_UUID'])
    except:
        process_uuid: str = default
    uuid.UUID(process_uuid) # throws exception for invalid uuid
    return process_uuid

# helper functions to retrieve environment variables to set some msem runtime parameters globally >>>

# not clear if these are re-entrant or not, so recreate on each usage.
def create_scipy_fft_context_manager(use_gpu, use_fft, use_fft_backend, nthreads):
    if not use_gpu and (use_fft == FFT_types.scipy_fft or use_fft == FFT_types.rcc_xcorr):
        ctx_managers = [scipy_fft.set_workers(nthreads)]
        # change fft backend for scipy, determines which library fftconvolve utilizes for ffts
        if use_fft_backend == FFT_backend_types.mkl:
            ctx_managers += [scipy_fft.set_backend(mkl_fft)]
        elif use_fft_backend == FFT_backend_types.fftw:
            ctx_managers += [scipy_fft.set_backend(pyfftw.interfaces.scipy_fft)]
            pyfftw.interfaces.cache.enable() # Turn on the cache for optimum performance
            #pyfftw.config.NUM_THREADS = _template_match_nthreads # does not do anything
    else:
        ctx_managers = [nullcontext()]

    return ctx_managers

# in some cases loading really large image files from GPFS is really really slow.
# avoid this by first copying the file to a memory mapped or local drive location
#   and then loading the image from there. delete the cache file afterwards.
def cached_image_load(loadpath, loadfn=None, cache_dir='', return_pil=False):
    if loadfn is None:
        fn = loadpath
        loadpath, loadfn = os.path.split(loadpath)
    else:
        fn = os.path.join(loadpath, loadfn)
    if not cache_dir: cache_dir = get_cache_dir()

    if cache_dir:
        cacheloadfn = os.path.join(cache_dir, uuid.uuid4().hex + '_' + loadfn)
        shutil.copyfile(fn, cacheloadfn)
    else:
        cacheloadfn = fn

    ext = os.path.splitext(loadfn)[1].lower()
    if ext == '.tiff' or ext == '.tif':
        img = tifffile.imread(cacheloadfn)
        if return_pil: img = Image.fromarray(img)
    else:
        img = Image.load(cacheloadfn); img.load(); img.close()
        # need copy to avoid ValueError: assignment destination is read-only
        if not return_pil: img = np.asarray(img).copy()

    # immediately remove the temp file after it's loaded.
    if cache_dir: os.remove(cacheloadfn)

    return img


# <<< access and helper methods to support tiling (to allow for tile processing) of very large images

def big_img_info(fn, dataset='image', attrs=None):
    fh = h5py.File(fn, 'r')
    image = fh[dataset]
    img_shape = image.shape; img_dtype = image.dtype
    if attrs is not None:
        for k in attrs.keys(): attrs[k] = image.attrs[k]
    fh.close()
    return img_shape, img_dtype

def big_img_load(fn, nblks=[1,1], iblk=[0,0], novlp_pix=[0,0], dataset='image', attrs=None, custom_rng=None,
        return_rng=False, img_blk=None, custom_drng=None, custom_slc=None, custom_dslc=None):
    single_block = all([x==1 for x in nblks])
    # load only the data for the requested block from the hdf5 file.
    fh = h5py.File(fn, 'r')
    image = fh[dataset]
    img_shape = image.shape
    if custom_rng is None:
        _, _, _, rng = tile_nblks_to_ranges(img_shape, nblks, novlp_pix, iblk)
    else:
        assert( single_block )
        rng = custom_rng
    shape = [x[1] - x[0] for x in rng]
    if img_blk is None:
        img_blk = np.empty(shape, dtype=image.dtype)
        assert(custom_drng is None)
    if single_block and custom_rng is None and custom_drng is None and custom_slc is None and custom_dslc is None:
        source_sel = dest_sel = None
    else:
        if custom_slc:
            source_sel = custom_slc
        else:
            source_sel = np.s_[rng[0][0]:rng[0][1],rng[1][0]:rng[1][1]]
        if custom_dslc:
            dest_sel = custom_dslc
        else:
            drng = custom_drng
            dest_sel = None if drng is None else np.s_[drng[0][0]:drng[0][1],drng[1][0]:drng[1][1]]
    image.read_direct(img_blk, source_sel=source_sel, dest_sel=dest_sel)
    if attrs is not None:
        for k in attrs.keys(): attrs[k] = image.attrs[k]
    fh.close()

    if return_rng:
        return img_blk, img_shape, rng
    else:
        return img_blk, img_shape

# the init only needs to be performed when utilizing the h5 locking mechanism.
def big_img_init(fn):
    # deleting the h5 file allows for zero'ing write count so we know how many blocks have been written.
    if os.path.isfile(fn): os.remove(fn)
    dfn, pfn = os.path.split(fn)
    fn_lock_file = os.path.join(dfn, '.lock-' + pfn)
    # this touch is multiprocess hell on gpfs, so only do it if the file is not there
    if not os.path.isfile(fn_lock_file): Path(fn_lock_file).touch()

def big_img_save(fn, img_blk, img_shape=None, nblks=[1,1], iblk=[0,0], novlp_pix=[0,0], save_ovlp=False,
        dataset='image', compression=False, recreate=False, overwrite_dataset=False, attrs=None, custom_slc=None,
        chunk_shape=None, truncate=False, sleep=None, wait=False, lock=False, f1=None, f2=None, keep_locks=False,
        img_shape_blk=None, bcrop=None):
    assert( ((f1 is None) and (f2 is None)) or ((f1 is not None) and f2 is not None) )
    open_locks = (f1 is not None)
    # Without file locking, this is the old method, logic (should still be) intact:
    #    IMPORTANT: made the logic create a new dataset if specified and only for the first block.
    #      Also, the dataset called image is intended to be the actual image data.
    #      The hdf5 file itself is only truncated if dataset is being recreated and writing image data.
    #      Other image metadata can be saved by specifying another name for dataset.
    #      So, when doing any export, the **first block of the image data should be saved first**.
    #      Additionally, hdf5 is not easily supporting multiple writers, so the blocks
    #        need to be exported serially.
    # With file locking, the order is no longer important. Typically the unlocking is handled externally
    #   after all the datasets for a given hdf5 file / block are written. This method is lock_recreate==True.
    lock_recreate = (recreate and (lock or open_locks))
    recreate = (all([x==0 for x in iblk]) and recreate)
    # force truncate when recreating image dataset, unless we are using locking.
    #   locking should always init, which creates the locking files and deletes the existing h5 files.
    truncate = ((recreate and dataset == 'image' and not lock_recreate) or truncate)
    single_block = all([x==1 for x in nblks])
    #tblks = np.prod(np.array(nblks))

    # hook to support locking but using a different file in the same directory.
    #   this is painfully slow for even a few writers for hdf5, so recommend only having this as
    #   a safety mechanism for rare cases, not truly for multiple writers support.
    # NOTE: this is because we could never get locking directly on the hdf5 file to work.
    #   if the file is locked first, h5py complains that it is locked. if hdf5 file locking is disabled,
    #   it causes all kinds of OS Errors. xxx - is there some way of doing this properly?
    dfn, pfn = os.path.split(fn)
    fn_lock_file = os.path.join(dfn, '.lock-' + pfn)
    # multiple writers still causes problems basically during any of the h5py calls,
    #   so just lock the file for the duration of the whole function for atomicity.
    if lock:
        if not open_locks: f1, f2 = gpfs_file_lock(fn_lock_file, sleep=sleep)

    lock_recreate_file_exists = lock_recreate and os.path.isfile(fn) and not truncate
    if (not recreate and not lock_recreate) or (lock_recreate and lock_recreate_file_exists):
        if sleep is None: csleep = [5,10]
        if wait and not lock_recreate_file_exists:
            found = False
            while not found:
                try:
                    fh = h5py.File(fn, 'r+')
                    found = True
                except:
                    if lock and not open_locks: gpfs_file_unlock(f1,f2)
                    time.sleep(csleep[0] if csleep[0] == csleep[1] else np.random.uniform(csleep[0], csleep[1]))
                    if lock and not open_locks: f1, f2 = gpfs_file_lock(fn_lock_file, sleep=sleep)
        else:
            fh = h5py.File(fn, 'r+')
        lock_recreate_dataset_exists = lock_recreate and (dataset in fh)
        if not lock_recreate or lock_recreate_dataset_exists:
            image = fh[dataset]
            img_shape = image.shape
            write_mask = image.attrs['write_mask']
        else:
            assert( img_shape is not None ) # need to specify shape with locking recreate
    else:
        lock_recreate_dataset_exists = False
        assert( img_shape is not None ) # need to specify shape with recreate
    if custom_slc is None:
        if img_shape_blk is None: img_shape_blk = img_shape
        #tile_nblks_to_ranges - return rngs, max_shape, min_shape, rng
        # get the ranges for slicing the entire image to get requested block.
        _, _, _, rng = tile_nblks_to_ranges(img_shape_blk, nblks, novlp_pix, iblk, bcrop=bcrop)
        if not save_ovlp and not all([x==0 for x in novlp_pix]):
            # remove the overlap from the block and adjust the ranges for no overlap
            _, _, _, rng_novlp = tile_nblks_to_ranges(img_shape_blk, nblks, [0,0], iblk, bcrop=bcrop)
            rmv_ovlp_pix = [[y[0]-x[0],x[1]-y[1]] for x,y in zip(rng, rng_novlp)]
            img_blk = img_blk[rmv_ovlp_pix[0][0]:img_blk.shape[0]-rmv_ovlp_pix[0][1],
                              rmv_ovlp_pix[1][0]:img_blk.shape[1]-rmv_ovlp_pix[1][1]]
            rng = rng_novlp
    else:
        assert( single_block )

    if (recreate and not lock_recreate) or \
            (lock_recreate and (not lock_recreate_file_exists or not lock_recreate_dataset_exists)):
        # force chunking on if compression is specified
        if chunk_shape is not None or compression:
            if chunk_shape is None:
                # xxx - specific to our gpfs system, for very large 2D images, force a chunk
                #   size that is the same size or some small multiple of the file system block size.
                if len(img_shape) == 2 and img_shape[0]*img_shape[1] >= 2**31:
                    chunk_shape = (4096, 4096)
                else:
                    # otherwise stick to autochunking; seems to work reasonably well in most cases
                    chunk_shape = True
        img_shape = tuple([x for x in img_shape]) # h5py only takes shape as a tuple
        if truncate:
            if os.path.isfile(fn): os.remove(fn) # somehow this helps sometimes on gpfs
            fh = h5py.File(fn, 'w')
        else:
            fh = h5py.File(fn, 'a')
        if overwrite_dataset and dataset in fh: del fh[dataset]
        image = fh.create_dataset(dataset, img_shape, dtype=img_blk.dtype, chunks=chunk_shape,
            **hdf5plugin.Blosc(cname='blosclz', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        write_mask = np.zeros(nblks, dtype=np.int64)
    write_mask[iblk[0], iblk[1]] += 1; image.attrs['write_mask'] = write_mask
    write_count = (write_mask > 0).sum() # just return a count of the unique written blocks instead of mask

    # write data
    # xxx - the blocking methods only work for 2d data
    # xxx - unclear at the moment if write_direct is worth it or not. most likely the data in memory will have
    #   to be copied, which is wasteful. however, write_direct is faster, but it seems not by enough to
    #   make the extra memory for the copy worth it. put a "hard parameter" here to leave open both options.
    #   decided for now to tie it to whether we're using blocking or not.
    use_write_direct = single_block
    if use_write_direct: img_blk = np.ascontiguousarray(img_blk)
    if single_block:
        # workaround for arbitrary dimensions, but blocking not supported (except with custom defined slice).
        if use_write_direct:
            image.write_direct(img_blk, dest_sel=custom_slc)
        else:
            image[:] = img_blk
    else:
        # for the normal range blocking the size must match,
        #   this means the indexing for making the block should also have been created by tile_nblks_to_ranges.
        if use_write_direct:
            image.write_direct(img_blk, dest_sel=np.s_[rng[0][0]:rng[0][1], rng[1][0]:rng[1][1]])
        else:
            image[rng[0][0]:rng[0][1],rng[1][0]:rng[1][1]] = img_blk

    # set some optional attributes
    if attrs is not None:
        for k,v in attrs.items(): image.attrs[k] = v

    fh.close()
    if lock and not keep_locks: gpfs_file_unlock(f1,f2)
    return write_count, f1, f2

# creates the ranges for overlapping blocks.
def tile_nblks_to_ranges(shape, nblks, novlp_pix, iblk, ignore_bounds=False, bcrop=None):
    ndims = len(shape); rndims = range(len(shape))
    single_block = all([x==1 for x in nblks])
    if single_block:
        nblks = [1]*ndims; iblk = [0]*ndims; novlp_pix = [0]*ndims
    else:
        assert( len(nblks) == ndims and len(iblk) == ndims and len(novlp_pix) == ndims )
    rngs = [[[x[0],x[-1]+1] for x in np.array_split(np.arange(y), z)] for y,z in zip(shape, nblks)]
    # add in overlap for blocks that are not along the edges
    for x in rndims:
        for y in range(nblks[x]):
            if ignore_bounds or y != 0:
                rngs[x][y][0] -= novlp_pix[x]
            if bcrop is not None and y != 0:
                rngs[x][y][0] -= bcrop[x][0]
            assert(ignore_bounds or rngs[x][y][0] >= 0) # overlap or crop bigger than chunk size
            if ignore_bounds or y != nblks[x]-1:
                rngs[x][y][1] += novlp_pix[x]
            if bcrop is not None:
                rngs[x][y][1] -= bcrop[x][0]
                if y == nblks[x]-1:
                    rngs[x][y][1] -= bcrop[x][1]
            assert(ignore_bounds or rngs[x][y][1] <= shape[x]) # overlap or crop bigger than chunk size
    max_shape = [max([rngs[x][y][1] - rngs[x][y][0] for y in range(nblks[x])]) for x in rndims]
    min_shape = [min([rngs[x][y][1] - rngs[x][y][0] for y in range(nblks[x])]) for x in rndims]
    rng = [rngs[x][y] for x,y in zip(rndims,iblk)]
    return rngs, max_shape, min_shape, rng

# access and helper methods to support tiling (to allow for tile processing) of very large images >>>


# <<< generic locking functions on GPFS, works for thread safe and process safe, BUT not very efficient

# NOTE: file locking is known to be problematic, this is very much os and file system
#   dependent, but works on GPFS. also this was a much easier solution for now than
#   switching to using a database or a server-based locking or mpi.
# inspired from "Old post, but if anyone else finds it, I get this behaviour:" here:
#   https://stackoverflow.com/questions/9907616/python-fcntl-does-not-lock-as-expected
# lockf and flock behave differently on GPFS, see:
#   https://www.ibm.com/mysupport/s/question/0D50z00006LKy2a/flock-on-gpfs?language=en_US
#   flock works on GPFS for file descriptors within the same processes,
#     but not for file descriptors in different processes.
#   lockf works on GPFS for different processes (same or different nodes),
#     but not for multilple file descriptors within the same process (i.e., threads)
#   implementation here uses both locks so it is both thread and process safe.
# mode 'r+' for lockf handle is because opening 'r' causes lockf to throw Bad File Descriptor (why?).

def gpfs_file_lock(fn, allow_create=False, sleep=None):
    if sleep is None: sleep = [5,10]
    if allow_create and not os.path.isfile(fn): Path(fn).touch()
    busy = True
    while busy:
        try:
            fthread = open(fn, 'rb'); fproc = open(fn, 'rb+')
            fcntl.flock(fthread, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.lockf(fproc, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = False
        except BlockingIOError:
            gpfs_file_unlock(fthread, fproc)
            # if many processes are trying to access the same set of files, polling a lot here
            #   is a disaster. since usually swarms are setup to prevent too many processes from
            #   concurrent file access, and this is an inefficient solution anyways, default
            #   to a relatively long randomized sleep range before trying again.
            time.sleep(sleep[0] if sleep[0] == sleep[1] else np.random.uniform(sleep[0], sleep[1]))
    return fthread, fproc

def gpfs_file_unlock(fthread, fproc):
    # the order here is important (?), close in the opposite order they were locked.
    fproc.close(); fthread.close()

# generic locking functions on GPFS, works for thread safe and process safe, BUT not very efficient >>>


# <<< access functions for reading and writing dills using file lock on GPFS

def dill_lock_and_load(fn, keep_locks=False):
    f1, f2 = gpfs_file_lock(fn)
    d = dill.load(f1)
    if keep_locks:
        # NOTE: returning open file handles in this case; they need to be closed eventually.
        #   This will block any other threads or processes from accessing this file
        #     if they access the file via these locking functions.
        return d, f1, f2
    else:
        gpfs_file_unlock(f1,f2)
        return d

def dill_lock_and_dump(fn, d, f1=None, f2=None):
    assert( ((f1 is None) and (f2 is None)) or ((f1 is not None) and f2 is not None) )
    open_locks = (f1 is not None)
    if not open_locks:
        f1, f2 = gpfs_file_lock(fn)
    # use a separate file handle from the locks for writing.
    f3 = open(fn, 'wb'); dill.dump(d, f3); f3.close()
    gpfs_file_unlock(f1,f2)

# access functions for reading and writing dills using file lock on GPFS >>>


# better grid point interpolation (and extrapolation) method than using normal interpolation.
# used for filling in outliers deltas as part of the fine alignment process.
# NOTE: this is a fully vectorized implementation, meaning that this will be very memory inefficient
#   if you try to apply it at too many points (i.e., all pixel locations in a large image).
def mls_rigid_transform(_v, p, q, alpha=1.0, tol=1e-6):
    ''' Rigid transform
    Image deformation using moving least squares
       see paper of same name by Schaefer et al
    ### Params:
        * v - ndarray: an array with size [m, 2], points to transform
        * p - ndarray: an array with size [n, 2], control points
        * q - ndarray: an array with size [n, 2], deformed control points
        * alpha - float: parameter used by weights
    ### Return:
        f_r - ndarray: an array with size [m, 2], deformed points from v
    '''

    # <<< avoid divide by zero in the weight calculation
    # use a tolerance (something less than 1), because points in v that are very close to those in p
    #   can also cause MLS to explode.
    knbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(p)
    kdist, knnbrs = knbrs.kneighbors(_v, return_distance=True)
    nsel_v = (kdist < tol).reshape(-1)
    any_v_in_p = nsel_v.any()
    if any_v_in_p:
        #print('{} points in query ({}) within {} of control ({})'.format(nsel_v.sum(), _v.shape[0], tol, p.shape[0]))
        closest_p_in_v = knnbrs.reshape(-1)[nsel_v]
        sel_v = np.logical_not(nsel_v)
        v = _v[sel_v,:]
    else:
        del nsel_v
        v = _v
    del knbrs, kdist, knnbrs
    # avoid divide by zero in the weight calculation >>>

    m, n = v.shape[0], p.shape[0]
    assert( q.shape[0] == n )

    # shape (n,m,2)
    p_ij = p[:,None,:]
    q_ij = q[:,None,:]
    v_ij = v[None,:,:]

    # weights, shape (n,m)
    w = 1.0 / np.sum((p_ij - v_ij)**2, axis=2)**alpha
    w_ij = w[:,:,None]
    sum_w = np.sum(w, axis=0)

    # centroids
    p_star = (w_ij * p_ij).sum(0) / sum_w[:,None] # shape (m,2)
    q_star = (w_ij * q_ij).sum(0) / sum_w[:,None] # shape (m,2)
    p_star_ij = p_star[None,:,:]
    q_star_ij = q_star[None,:,:]
    p_hat_ij = p_ij - p_star_ij
    q_hat_ij = q_ij - q_star_ij

    # similarity transform solved matrix
    p_hat_ij_uT = np.zeros_like(p_hat_ij)
    # from paper: (_uT) is an operator on 2D vectors such that (x, y) = (−y, x).
    p_hat_ij_uT[:,:,0] = -p_hat_ij[:,:,1]
    p_hat_ij_uT[:,:,1] = p_hat_ij[:,:,0]
    v_minus_p_star = v_ij - p_star_ij
    v_minus_p_star_uT = np.zeros_like(v_minus_p_star)
    v_minus_p_star_uT[:,:,0] = -v_minus_p_star[:,:,1]
    v_minus_p_star_uT[:,:,1] = v_minus_p_star[:,:,0]
    # transform matrix, shape (n,m,2,2)
    A_ij = w_ij[:,:,:,None] * np.matmul(\
            np.concatenate((p_hat_ij[:,:,:,None], -p_hat_ij_uT[:,:,:,None]), axis=3),
            np.concatenate((v_minus_p_star[:,:,:,None], -v_minus_p_star_uT[:,:,:,None]), axis=3).transpose(0,1,3,2))

    # f_r, shape (m,2)
    f_bar_r = np.matmul(q_hat_ij[:,:,None,:], A_ij).sum(0).reshape(m,2)
    v_minus_p_star = v_minus_p_star.reshape(m,2)
    _f_r = np.sqrt((v_minus_p_star**2).sum(1))[:,None] * f_bar_r / np.sqrt((f_bar_r**2).sum(1))[:,None] + q_star

    # <<< avoid divide by zero in the weight calculation
    if any_v_in_p:
        f_r = np.zeros_like(_v)
        f_r[sel_v,:] = _f_r
        # just the deformed control points for points in v that are very close to points in p
        f_r[nsel_v,:] = q[closest_p_in_v,:]
    else:
        f_r = _f_r
    # avoid divide by zero in the weight calculation >>>

    return f_r


# https://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
def constrainAngle(a):
    if sp.issparse(a):
        a = a.copy() # do not modify original inpput
        a.data = np.fmod(a.data + np.pi, 2*np.pi)
        sel = (a.data < 0); a.data[sel] += 2*np.pi
        a.data -= np.pi
        return a
    else:
        a = np.fmod(a + np.pi, 2*np.pi)
        if np.isscalar(a):
            if a < 0: a += 2*np.pi
        else:
            sel = (a < 0); a[sel] += 2*np.pi
        return a - np.pi

# NOTE: these two polygon functions assume closed polygons, so start point should not be repeated at the end.

# http://paulbourke.net/geometry/polygonmesh/
# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolySignedArea(x,y):
    xn = np.roll(x,-1); yn = np.roll(y,-1) # sign matters, make counterclockwise positive
    return (np.dot(x,yn)-np.dot(xn,y))/2.

# http://paulbourke.net/geometry/polygonmesh/
# validated against prettier python code at:
#   https://github.com/pwcazenave/pml-git/blob/master/python/centroids.py
# and also against matlab 'centroid' function for polygons.
def PolyCentroid(x,y):
    xn = np.roll(x,1); yn = np.roll(y,1)
    coeff = 1/3.0/(np.dot(x,yn)-np.dot(xn,y)) # sign does not matter
    common = x*yn - xn*y
    return coeff*np.dot( x+xn, common ), coeff*np.dot( y+yn, common )

# xxx - keep here for reference, but got too complex for grids with lots of points
#   and also is not necessary in the current workflow, as the spacing is provided as a parameter anyways.
# # "estimates" the distance between neighboring points on a mostly regular hexagonal (or cartesian) grid.
# def regular_grid_distance(grid_points):
#     delaunay_tris = spatial.Delaunay(grid_points)
#     simplices = delaunay_tris.simplices
#     ntris = simplices.shape[0]
#
#     # get the primary grid unit distance and all the triangle areas.
#     D = np.zeros((ntris,3), dtype=np.double)
#     for i in range(ntris):
#         pts = grid_points[simplices[i,:],:]
#         D[i,:] = scidist.pdist(pts)
#     grid_dist, _ = st.mode(np.round(D.flat[:]))
#
#     return grid_dist, simplices, D


# https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def isInt_str(v):
    v = str(v).strip()
    return v=='0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()

# https://stackoverflow.com/questions/7085512/check-what-number-a-string-ends-with-in-python
def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)

# nearly square divisors
def nsd(n):
    s = math.isqrt(n)
    a = max(d for d in sympy.divisors(n) if d <= s)
    return a, n // a

# # translates the image center by specified amount using padding.
# # return image shift amount in coordinate space.
# def translate_image_center(img, trans):
#     sz = np.array(img.size)
#     shift = np.round(2*trans).astype(np.int64)
#     sz += np.abs(shift); shift[shift < 0] = 0
#     shifted_img = Image.new(img.mode, tuple(sz.tolist()))
#     shifted_img.paste(img, tuple(shift.tolist()))
#     return shifted_img, shift

# makes centered "hexagonal grid" points, meaning all the nearest neighbors are equally spaced.
# scale is the spacing between points along the x-axis.
# corner of the grid is aligned on the most negative point (point furthest from origin in third quadrant).
# this construction method is not unique.
# set distance_power to non-zero value to make points spacing increase as distance from origin.
#   distance_power==1. and distance_norm==2. means point spacing scales as euclidean distance
def make_hex_xy(size_x,size_y,scale,distance_power=0.,distance_norm=2.):
    x,y = np.mgrid[0:size_x,0:size_y]*1.0
    x[:,1::2] += 0.5; y *= np.sqrt(3)/2
    rngx = x.max()/2; rngy = y.max()/2
    x = x - rngx; y = y - rngy
    if distance_power > 0:
        d = (np.abs(x)**distance_norm + np.abs(y)**distance_norm)**(distance_power/distance_norm)
        d = d/rngx**distance_power; x = x*d; y = y*d
        sel = np.logical_and(np.abs(x) <= rngx, np.abs(y) <= rngy)
        x = x[sel]; y = y[sel]; d = d[sel]
    x = x.flat[:]; y = y.flat[:]
    return x*scale,y*scale

# more convenient make_hex_xy wrapper for how it is used in msem package.
def make_hex_points(size_x,size_y,scale,trans=[0,0],bbox=False):
    if bbox and scale <= 0:
        # this is to allow bounding box to be specified directly.
        ret_val = [size_x, size_y]
    else:
        x,y = make_hex_xy(size_x, size_y, scale)
        pts = np.concatenate((x[:,None], y[:,None]), axis=1) + trans
        if bbox:
            ret_val = [pts.min(0), pts.max(0)]
        else:
            # move the point that is closest to the image center to the first position.
            # some msem package plots assume this is the "center point".
            i = np.argmin((pts*pts).sum(1)); sel = np.ones((x.size,), dtype=bool); sel[i]=0
            ret_val = np.concatenate( (pts[i,:][None,:], np.delete(pts,i,axis=0)) )
    return ret_val

def mad_zscore(x):
    x_minus_med = x - np.median(x)
    mad = np.median(np.abs(x_minus_med))
    return (x_minus_med/1.4826/mad if mad > 0 else np.zeros(x.shape))

def mad_angle_zscore(a):
    aflip = a.copy()
    sel = (aflip < 0); aflip[sel] = np.pi + aflip[sel]
    sel = np.logical_not(sel); aflip[sel] = -(np.pi - aflip[sel])
    mad_a = mad_zscore(a); mad_aflip = mad_zscore(aflip)
    sel = (np.abs(mad_aflip) < np.abs(mad_a))
    mad_a[sel] = mad_aflip[sel]
    return mad_a

def l2_and_delaunay_distance_select(cinliers, inlier_radius, blk_center, griddist_pixels, use_delaunay=True):
    ninliers = cinliers.shape[0]
    # first use a normal radius distance based on the center of the block
    sel_rad = (((cinliers - blk_center)**2).sum(1) < inlier_radius**2)

    if use_delaunay:
        # second use the radius as a cutoff for a "delaunay distance"
        # the triangulation is a bit compute intensive and single threaded, hence the multiprocessing here
        # compute triangulation that includes the block center point as first point
        cinliers_blk_ctr = np.concatenate((blk_center[None,:], cinliers), axis=0)
        dln = spatial.Delaunay(cinliers_blk_ctr)
        # remove long edges that are not connected to the block center
        edge_lens = np.array([scidist.pdist(x) for x in cinliers_blk_ctr[dln.simplices]])
        # xxx - additional parameter that controls the tri edge distance cutoff?
        sel = np.logical_or((edge_lens < inlier_radius/2).all(1), (dln.simplices == 0).any(1))
        simplices = dln.simplices[sel,:]
        #make_grid_plot(simplices, cinliers_blk_ctr); plt.show()
        #plt.show()
        # create a graph out of the delaunay triangulation
        dlg = sp.dok_array((ninliers+1,ninliers+1), dtype=bool)
        dlg[simplices[:,0], simplices[:,1]] = 1
        dlg[simplices[:,1], simplices[:,2]] = 1
        dlg[simplices[:,0], simplices[:,2]] = 1
        # use shortest path algorithm to calculate the "delaunay distance"
        d = sp.csgraph.shortest_path(dlg, method='D', directed=False,
                return_predecessors=False, unweighted=True, overwrite=True,
                indices=[0]).reshape(-1)[1:]
        # convert inlier_radius to delaunay unit distance,
        #   i.e., the distance between grid points == 1.
        # then combine the select with the l2 distance select.
        sel_rad = np.logical_or(sel_rad, d < (inlier_radius / griddist_pixels))
    #if use_delaunay:

    return sel_rad


# <<< simplex circumcenter averaging utility functions

# SO python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d
def get_voronoi_circumcenters_from_delaunay_tri(points, simplices):
    #points = np.random.rand(30, 2) # original written as npts x ndims
    p = points[simplices]

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T

    # See http://en.wikipedia.org/wiki/Circumscribed_circle#Circumscribed_circles_of_triangles
    # The following is just a direct transcription of the formula there
    a = A - C
    b = B - C

    def dot2(u, v):
        return u[0]*v[0] + u[1]*v[1]

    def cross2(u, v, w):
        """u x (v x w)"""
        return dot2(u, w)*v - dot2(u, v)*w

    def ncross2(u, v):
        """|| u x v ||^2"""
        return sq2(u)*sq2(v) - dot2(u, v)**2

    def sq2(u):
        return dot2(u, u)

    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C

    return cc.T

def get_voronoi_to_simplex_mapping(simplex_circumcenters, voronoi_vertices):
    nvertices = voronoi_vertices.shape[0]
    nsimplices = simplex_circumcenters.shape[0]

    if nvertices==nsimplices:
        # we can use the single nearest neighbor search because:
        #   only a many circumcenters to single voronoi vertex mapping is possible,
        #     not a many voronoi vertices to single circumcenter.
        #   so if nvertices==nsimplices there must be a one-to-one mapping.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(simplex_circumcenters)
        voronoi_to_simplex = nbrs.kneighbors(voronoi_vertices, return_distance=False).flat[:]
        # just to make that statement above is correct...
        assert((np.unique(voronoi_to_simplex)==np.arange(nsimplices)).all()) # multiple voronoi vectors to cc?!?
    else:
        # first find the minimum distance between the voronoi vertices.
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(voronoi_vertices)
        dmin, _ = nbrs.kneighbors(voronoi_vertices, return_distance=True)
        radius = dmin[:,1].min()/2

        # here we have to use a distance search because there is no bound on
        #   the number of circumcenters that could map to the same voronoi vertex.
        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(simplex_circumcenters)
        voronoi_to_simplex = nbrs.radius_neighbors(voronoi_vertices, return_distance=False)
        mapped = np.concatenate(voronoi_to_simplex)
        assert( mapped.size <= nsimplices ) # this is totally wack... help me
        if mapped.size < nsimplices:
            print('WARNING: very "co-circular" grid points, recommend not doing this')
            # apparently qhull has some liberty about where it puts "crazy" voronoi vertices.
            # if all the circumcenters are not mapped, used k=1 nearest neighbor to map the remainder...
            # xxx - not sure if this is completely correct, but many "co-circular" distributed points
            #   is not a common use case for the msem package anyways.
            unmapped = np.setdiff1d(np.arange(nsimplices), mapped)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(voronoi_vertices)
            simplex_to_voronoi = nbrs.kneighbors(simplex_circumcenters[unmapped,:], return_distance=False).flat[:]
            # append the unmapped circumcenters to the nearest voronoi vertex.
            for i in range(unmapped.size):
                voronoi_to_simplex[simplex_to_voronoi[i]] = \
                    np.append(voronoi_to_simplex[simplex_to_voronoi[i]], unmapped[i])
            # uncomment for sanity check
            # mapped = np.concatenate(voronoi_to_simplex)
            # unmapped = np.setdiff1d(np.arange(nsimplices), mapped)
            # assert( unmapped.size == 0 )

    return voronoi_to_simplex

# this function is only need for remapping all the points at once, and even then only
#   in the complex case below. if iterating over the vertices, or in the case of the
#   one-to-one mapping, one can just use the voronoi_to_simplex remapping directly.
def vector_simplex_circumcenters_to_voronoi(circumcenter_vectors, voronoi_to_simplex):
    nvertices = voronoi_to_simplex.size
    nsimplices, nd = circumcenter_vectors.shape

    if nvertices==nsimplices:
        # simple case: just remap with indexing
        voronoi_vectors = circumcenter_vectors[voronoi_to_simplex,:]
    else:
        # complex case: for multiple circumcenters that map to a single voronoi vertex, take the mean.
        voronoi_vectors = np.zeros((nvertices,2), dtype=np.double)
        for i in range(nvertices):
            # see http://qhull.org/html/qvoronoi.htm option Qz, point "at infinity"
            # just ignore these points.
            if voronoi_to_simplex[i].size > 0:
                voronoi_vectors[i,:] = circumcenter_vectors[voronoi_to_simplex[i],:].mean(0)

    return voronoi_vectors

def voronoi_vectors_to_point_vectors(vectors_voronoi, points, voronoi, idw_p=2., bcast=False):
    npts = points.shape[0]

    # convert back to the original points by taking the mean of all the voronoi vertices
    #   that define the region for that point, weighted by the inverse distance to each voronoi vertex.
    if bcast:
        vectors_points = np.zeros((vectors_voronoi.shape[0],npts,2), dtype=np.double)
    else:
        vectors_points = np.zeros((npts,2), dtype=np.double)
    # pad the regions out to create 2d array with -1 fillvalue for easier lookup.
    regions = np.array(list(itertools.zip_longest(*voronoi.regions, fillvalue=-1))).T
    for i in range(npts):
        cipts = regions[voronoi.point_region[i],:]
        cipts = cipts[cipts >= 0]
        # weight the average by the inverse distance to each point.
        # this is inverse distance weighting interpolation.
        W = 1/scidist.cdist(voronoi.vertices[cipts,:], points[[i],:])**idw_p
        if bcast:
            vectors_points[:,i,:] = (vectors_voronoi[:,cipts,:]*W).sum(1)/W.sum()
        else:
            vectors_points[i,:] = (vectors_voronoi[cipts,:]*W).sum(0)/W.sum()

    return vectors_points

# simplex circumcenter averaging utility functions >>>


def label_fill_image_background(aimg, bg_fill_type='noise', constant=128, psize_cutoff=0.0016, bgsel=None):
    amax = aimg.max(); amin = aimg.min(); arng = amax - amin
    if bgsel is None:
        # this is a method to try to "estimate" the background around the slice (i.e., never contained data).
        # this is somewhat deprecated, the normal workflow now is to just save it along with the image.
        lbls, nlbls = nd.label(aimg==0)
        if nlbls==0: return aimg
        obgsel = np.zeros(aimg.shape, dtype=bool)
        sizes = np.bincount(np.ravel(lbls))[1:] # component sizes
        bglbls = np.nonzero(sizes/aimg.size > psize_cutoff)[0]
    else:
        bglbls = [0]; obgsel = None
    for lbl in bglbls:
        sel = (lbls==lbl+1) if bgsel is None else bgsel
        if bgsel is None: obgsel = np.logical_or(obgsel,sel)
        if bg_fill_type == 'noise':
            if np.issubdtype(aimg.dtype, np.integer):
                fillvals = np.random.randint(amin, high=amax+1, size=(sel.sum(),), dtype=aimg.dtype)
            else:
                fillvals = (np.random.rand(sel.sum())*arng + amin).astype(aimg.dtype)
        elif bg_fill_type == 'constant':
            fillvals = constant
        elif bg_fill_type == 'none':
            pass
        else:
            print(bg_fill_type)
            assert(False) # bad bg_fill_type
        if bg_fill_type != 'none': aimg[sel] = fillvals
    return obgsel

def fill_outside_polygon(img, roi_points, imgpil=None, fill=0, outline=0, docrop=False):
    img = Image.fromarray(img) if imgpil is None else imgpil
    if docrop:
        crpmin = roi_points.min(0); crpmax = roi_points.max(0)
        img = img.crop((crpmin[0], crpmin[1], crpmax[0], crpmax[1]))
        pts = roi_points - crpmin
    else:
        pts = roi_points
    img_mask = Image.new('L', img.size, color=255)
    pdraw = ImageDraw.Draw(img_mask)
    pdraw.polygon([tuple(x) for x in pts], fill=fill, outline=outline)
    img.paste(0, mask=img_mask)
    # need copy to avoid ValueError: assignment destination is read-only
    if imgpil is None: img = np.asarray(img).copy()
    return img if imgpil is None else None

# image integer upsampling without interpolation (nearest).
# simple block construction method using slices.
def block_construct(img, factor):
    oimg = np.empty(np.array(img.shape)*factor, img.dtype)
    for i in range(factor):
        for j in range(factor):
            oimg[i::factor,j::factor] = img
    return oimg

# heuristical method of finding "tissue mode" in EM grayscale data.
# attempts to essentially to deal with mutliple peaks based on slices with little tissue
#   relative to other things in the slice (epon, wafer, iron, artifacts, etc).
def find_histo_mode(histo, mode_limits, mode_rel, histo_smooth_size):
    # meh, some small slices (a few mfovs or less) can still have non-smooth full slice
    #   histograms, so apply a small smoothing.
    smoothed_histo = histo.astype(np.double)
    if histo_smooth_size > 0: smoothed_histo = nd.uniform_filter1d(smoothed_histo, histo_smooth_size)

    # determine grayscale cutoffs for heuristics based on the summed histograms
    peaks, _ = signal.find_peaks(smoothed_histo) # this returns all the local maxima
    # this was meant for another heuristic that did not work very well
    #prominences = signal.peak_prominences(smoothed_histo, peaks)[0]

    # go through the "preferred ranges" for the slice histo mode.
    # this is another heuristic to deal with multimodal histo slices.
    # take a peak as mode if it's in the highest preferred range,
    #   and is also greater than some percentage of the overall peak.
    histo_mode = -1 # default if mode finding fails
    histo_max = histo.max()
    for i in range(len(mode_limits)):
        ipeaks = np.nonzero(np.logical_and(peaks >= mode_limits[i][0], peaks <= mode_limits[i][1]))[0]
        if ipeaks.size > 0:
            #iproms = prominences[ipeaks] # sometimes results in disasterous results
            ipeaks = peaks[ipeaks]
            # take largest peak in the range, if there are multiple
            j = np.argmax(histo[ipeaks]); cpeak = ipeaks[j] #; cprom = iproms[j]
            # take only if it's greater than some percentage of the max peak
            if histo[cpeak] / histo_max > mode_rel:
                histo_mode = cpeak; break
                # take only if it's the max peak,
                #   or if it's prominence is greater than some percentage of the max peak.
                #if histo[cpeak] == histo_max or cprom / histo_max > mode_rel:

    return histo_mode, smoothed_histo

# <<< plotting functions for deltas / grid points

def make_delta_plot(cgrid_points, deltas=None, grid_sel_r=None, grid_sel_b=None, center=False, figno=2):
    scl = 0.1; msz = 36
    #scl = 1.; msz = 36
    min_cgrid = cgrid_points.min(0); max_cgrid = cgrid_points.max(0)
    print('npts at xmin (nypts) {} ({})'.format((cgrid_points[:,0] == min_cgrid[0]).sum(),
        2*(cgrid_points[:,0] == min_cgrid[0]).sum()-1))
    print('npts at ymin (nxpts) {}'.format((cgrid_points[:,1] == min_cgrid[1]).sum()))
    plt.figure(figno); plt.clf()
    plt.scatter(cgrid_points[:,0], cgrid_points[:,1], c='g', s=msz, marker='.')
    if grid_sel_r is not None:
        plt.scatter(cgrid_points[grid_sel_r,0], cgrid_points[grid_sel_r,1], c='r', s=msz, marker='.')
    if grid_sel_b is not None:
        plt.scatter(cgrid_points[grid_sel_b,0], cgrid_points[grid_sel_b,1], c='b', s=msz, marker='.')

    #plt.plot(0, 0, 'r.')
    if deltas is not None:
        d = (deltas - deltas[0,:]) if center else deltas
        plt.quiver(cgrid_points[:,0], cgrid_points[:,1], d[:,0], d[:,1],
                   angles='xy', scale_units='xy', scale=scl, color='k')
    plt.xlim([min_cgrid[0], max_cgrid[0]])
    plt.ylim([min_cgrid[1], max_cgrid[1]])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(8,6)
    plt.axis('off')


def make_grid_plot(simplices, cgrid_points, ctrs=None, deltas=None, figno=1):
    msz = 12
    min_cgrid = cgrid_points.min(0); max_cgrid = cgrid_points.max(0)
    if deltas is None:
        deltas = np.zeros_like(cgrid_points)
    # Create the matplotlib Triangulation object
    x = cgrid_points[:,0] + deltas[:,0] - deltas[0,0]
    y = cgrid_points[:,1] + deltas[:,1] - deltas[0,1]
    triang = mtri.Triangulation(x=x, y=y, triangles=simplices)
    plt.figure(figno); plt.clf()
    plt.triplot(triang, 'k-', lw=0.5)
    #plt.plot(0, 0, 'r.')
    if ctrs is not None:
        plt.scatter(ctrs[:,0], ctrs[:,1], c='r', s=msz, marker='.')
    plt.xlim([min_cgrid[0], max_cgrid[0]])
    plt.ylim([min_cgrid[1], max_cgrid[1]])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(8,6)
    plt.axis('off')

# plotting functions for deltas / grid points >>>


# <<< graph color code

def color_from_edgelist(edgelist, chromatic=None):
    G = nx.from_edgelist(edgelist)
    return color(G, chromatic)

def color(G, chromatic=None):
    # try greedy first using all strategies, then try optimal (slow) if chromatic is specified
    #   meaning that we want an exact number of colors for coloring the non-neighboring labels.
    strategies = [
        'strategy_largest_first','strategy_smallest_last','strategy_independent_set',
        'strategy_independent_set','strategy_connected_sequential_dfs',
        'strategy_connected_sequential_bfs','strategy_saturation_largest_first',
    ]
    interchange = np.ones((len(strategies),),dtype=bool); interchange[[2,6]] = 0; minclrs = G.number_of_nodes()
    for i in range(len(strategies)):
        tmp = nx.coloring.greedy_color(G, strategy=eval('nx.coloring.'+strategies[i]),
            interchange=bool(interchange[i]))
        nclrs = max(tmp.values())+1
        if nclrs < minclrs: Gclr = tmp; minclrs = nclrs
        if chromatic is not None and minclrs <= chromatic: break
    if chromatic is not None and minclrs > chromatic:
        Gclr = optimal_color(G,chromatic); minclrs = chromatic

    return Gclr

# http://www.geeksforgeeks.org/backttracking-set-5-m-coloring-problem/

# A utility function to check if the current color assignment is safe for vertex v
def isSafe(v, graph, color, c):
    for i in graph.neighbors(v):
        if c == color[i]: return False
    return True

def graphColoringUtil(graph, m, color, v):
    # base case: If all vertices are assigned a color then
    if v == graph.number_of_nodes(): return True

    # consider vertex v and try different colors
    for c in range(1,m+1):
        # Check if assignment of color c to v is fine
        if isSafe(v, graph, color, c):
            color[v] = c

            # recur to assign colors to rest of the vertices
            if graphColoringUtil(graph, m, color, v+1): return True

            # If assigning color c doesn't lead to a solution then remove it
            color[v] = 0

    # If no color can be assigned to this vertex then return false
    return False

# This function solves the m Coloring problem using Backtracking.
# It mainly uses graphColoringUtil() to solve the problem. It returns
# false if the m colors cannot be assigned, otherwise return true and
# prints assignments of colors to all vertices. Please note that there
# may be more than one solutions, this function prints one of the
# feasible solutions
def optimal_color(graph, chromatic):
    # Initialize all color values as 0. This initialization is needed correct functioning of isSafe()
    color = dict((node_id, 0) for node_id in graph.nodes())

    solution_exists = graphColoringUtil(graph, chromatic, color, 0)
    return dict((n, c-1) for n,c in color.items()) if solution_exists else None

# graph color code >>>

# <<< histogram fitting code
# this code is here for fitting some good slice histograms in the effort of creating
#   a target histogram for the histogram matching method of contrast equalization.

# <<< modified from
#   https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# Create models from data
#def best_fit_distribution(data, bins=200, ax=None):
def best_fit_distribution(data=None, data_dtype=None, histo=None, fit_hist=True, nrepeats=1, rng=None):
    """Model data by finding best fit distribution to data"""
    assert( data is not None or (data_dtype is not None and histo is not None and fit_hist) )
    if data_dtype is None: data_dtype = data.dtype
    assert( np.issubdtype(data_dtype, np.integer) )
    datamax = np.iinfo(data_dtype).max
    if histo is None:
        histo = np.bincount(np.ravel(data), minlength=datamax+1)
    y = histo/histo.sum()
    x = np.arange(datamax+2)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    if rng:
        x = x[rng[0]:rng[1]]; y = y[rng[0]:rng[1]]

    # Distributions to check, all as of scipy==1.5.1
    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.argus, st.beta, st.betaprime, st.bradford, st.burr, st.burr12, st.cauchy,
        st.chi, st.chi2, st.cosine, st.crystalball, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm,
        st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.frechet_r,
        st.frechet_l, st.genlogistic, st.gennorm, st.genpareto, st.genexpon, st.genextreme, st.gausshyper,
        st.gamma, st.gengamma, st.genhalflogistic, st.geninvgauss, st.gilbrat, st.gompertz, st.gumbel_r, st.gumbel_l,
        st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.kappa4, st.kappa3, st.ksone, st.kstwo, st.kstwobign, st.laplace,
        st.levy, st.levy_l, st.levy_stable, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.loguniform,
        st.lomax, st.maxwell, st.mielke, st.moyal, st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.norminvgauss,
        st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.rayleigh, st.rice,
        st.recipinvgauss, st.semicircular, st.skewnorm, st.t, st.trapz, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max,
        st.wrapcauchy,
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for r in range(nrepeats):
        for distribution in DISTRIBUTIONS:
            #print(distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    if fit_hist:
                        p0 = np.concatenate((3*np.random.randn(distribution.numargs),
                                datamax*np.random.rand(2)))
                        params,pcov = opt.curve_fit(lambda x,*kwargs: distribution.pdf(x,*kwargs), x, y, p0=p0,
                                maxfev=1000)
                        params = np.concatenate((params[:distribution.numargs], params[-2:]))
                    else:
                        # fit dist to data
                        params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            #except Exception as e: raise e
            except: pass

    return (best_distribution.name, best_params, getattr(st, best_distribution.name))

def make_pdf(dist, params, size=None, rng=None):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    if rng is None:
        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
    else:
        start, end = rng

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, end-start if size is None else size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)

    return x,y

#   https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
# modified from >>>

def best_fit_piecewise_distribution(data_dtype, histo, rngs, nrepeats=1):
    nrngs = len(rngs)

    maxrng = max([x[1] for x in rngs])
    minrng = min([x[0] for x in rngs])
    templ_pdf = np.zeros((maxrng-minrng,), dtype=np.double)
    cbins = np.zeros((maxrng-minrng,), dtype=np.double)
    dist_strs = [None]*nrngs
    for rng,i in zip(rngs,range(nrngs)):
        best_fit_name, best_fit_params, best_dist = best_fit_distribution(data_dtype=data_dtype, histo=histo,
                nrepeats=nrepeats, rng=rng)

        # Make PDF with best params
        cbins[rng[0]:rng[1]],templ_pdf[rng[0]:rng[1]] = make_pdf(best_dist, best_fit_params, rng=rng)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:g}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_strs[i] = '{}({})'.format(best_fit_name, param_str)

    # not guaranteed to sum to 1 so renormalize
    templ_pdf = templ_pdf / templ_pdf.sum()

    return templ_pdf, cbins, dist_strs

# histogram fitting code >>>
