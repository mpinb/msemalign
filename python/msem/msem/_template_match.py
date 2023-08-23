"""_template_match.py

Implements image template matching using normalized cross corrrelations.

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

# Fourier domain implementations roughly equivalent to matlab's normxcorr2.
# Two versions:
#   (1) for mfov alignment with adjacency matrix
#   (2) for alignment between regions including brute-force rotation search

# xxx - this module could use some refactoring for readability. is there a way to combine the adjacency method
#   with the non-adjacency method? much of the code is repeated...

import numpy as np
import time
import os
import glob

from contextlib import ExitStack
import functools

from numpy import fft as numpy_fft
from scipy import fft as scipy_fft
import scipy.ndimage as nd
import scipy.spatial.distance as scidist
import cv2

try:
    import pyfftw
    _pyfftw_imported = True
except:
    print('WARNING: pyfftw unavailable, needed for one method for template matching with cross-correlations')
    _pyfftw_imported = False

try:
    import cupy as cp
    import cupyx.scipy.fft as cupy_fft
except:
    print('WARNING: cupy unavailable, needed for one method for template matching with cross-correlations')

try:
    from rcc_xcorr.xcorr import BatchXCorr
except:
    print('WARNING: rcc-xcorr unavailable, needed for one method for template matching with cross-correlations')




from scipy.ndimage import filters
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from msem.utils import get_num_threads, get_fft_types, FFT_types, get_trailing_number
from msem.utils import get_fft_backend, create_scipy_fft_context_manager

# fft type to use by default
_use_fft_types=get_fft_types()

# fft backend to use by default
_use_fft_backend=get_fft_backend()

# byte alignment to use for pyfftw
pyfftw_aligned=8

# number of threads by default
_template_match_nthreads=get_num_threads()

# xxx - make types configurable?
_dtype_fft = np.float32; _dtype_cfft = np.complex64



# The meat of the code in normxcorr2_fft_adj and normxcorr2_fft and also the function local_sum
#   are translated and modified from matlab code from kb that was in turn taken out of matlab's normxcorr2.
# xxx - potential licensing issue here

# Comment header from matlab's normxcorr2:
#NORMXCORR2 Normalized two-dimensional cross-correlation.
#   C = NORMXCORR2(TEMPLATE,A) computes the normalized cross-correlation of
#   matrices TEMPLATE and A. The matrix A must be larger than the matrix
#   TEMPLATE for the normalization to be meaningful. The values of TEMPLATE
#   cannot all be the same. The resulting matrix C contains correlation
#   coefficients and its values may range from -1.0 to 1.0.

#   We normalize the cross correlation to get correlation coefficients using the
#   definition of Haralick and Shapiro, Volume II (p. 317), generalized to
#   two-dimensions.
#
#   Lewis explicitly defines the normalized cross-correlation in two-dimensions
#   in this paper (equation 2):
#
#      "Fast Normalized Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.
#      http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html
#
#   Our technical reference document on NORMXCORR2 shows how to get from
#   equation 2 of the Lewis paper to the code below.

def local_sum(A,m,n):
    # We thank Eli Horn for providing this code, used with his permission,
    # to speed up the calculation of local sums. The algorithm depends on
    # precomputing running sums as described in "Fast Normalized
    # Cross-Correlation", by J. P. Lewis, Industrial Light & Magic.
    # http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html

    B = np.lib.pad(A, ((m,m),(n,n)), 'constant', constant_values=0)
    s = np.cumsum(B, axis=0)
    c = s[m:-1,:]-s[:-m-1,:]
    s = np.cumsum(c, axis=1)
    return s[:,n:-1]-s[:,:-n-1]

# xxx - in an ideal world I never wrote this as accepting the adjacency matrix.
#   Reasons that normxcorr2_fft_adj and normxcorr2_fft are not assimilated into a single function:
#     (1) the triu thing is tied into how the cropping of the images is done in mfov (also unideal).
#     (2) besides ffts, the main speed savings here is by precomputing the image ffts outside the main xcorr loop.
#         doing this row by row would require ffts for many images to be computed twice.
def normxcorr2_fft_adj(templates, templates_shape, imgs, imgs_shape, adj_matrix,
                       template_offset=None, img_offset=None, crp=np.array([0,0]),
                       nthreads=_template_match_nthreads, use_fft_types=_use_fft_types, use_gpu=False,
                       # these are for whether templates / images are passed in differently for upper/lower adjancencies
                       use_tri_templates=False, use_tri_images=False,
                       # these are for preserving fft precomputations on the images between template calls
                       return_precomputes=False,
                       Fa=None, Fb=None, T=None, F=None, fftT=None, fftF=None, local_sum_A=None, denom_A=None,
                       doplots=False, dosave_path='', return_comps=False):
    nimgs = len(imgs[0]) # python array (for tri) of python array of numpy images (of same dimensions)
    assert(adj_matrix.shape[0] == nimgs and adj_matrix.shape[1] == nimgs)
    use_fft = use_fft_types[int(use_gpu)]

    # parameters - some just to preserve original matlab variable names
    Aimgs = imgs; Aimgs_shape = imgs_shape
    A_size = np.array(Aimgs_shape)
    T_size = np.array(templates_shape)
    assert( (T_size <= A_size).all() ) # apparently this algorithm is not intended for template bigger than image
    m,n = T_size; mn = m*n
    outsize = A_size + T_size - 1
    assert( (outsize > 2*crp).all() ) # specified cropping is bigger than the output xcorr size
    toutsize = tuple(outsize.tolist()) # cupy fails without explicit tuple
    Tntri = int(use_tri_templates) + 1
    Antri = int(use_tri_images) + 1

    # xxx - marginally better performance with real ffts, decided not worth it for now
    #   because the implication for precision for using real ffts versus full ffts is unclear.
    real_fft = False
    # whether to zero pad for faster fft performance
    # xxx - same story here, did not really help that much, unclear if the normalization is affected.
    calc_fast_len = False
    # divide by zero tolerance
    #tol = np.finfo(_dtype_fft).eps
    tol = 1e-6

    # do not remove these, not worth the slight time savings
    assert(all([x is None or (x.shape == A_size).all() for y in imgs for x in y]))
    assert(all([x is None or (x.shape == T_size).all() for y in templates for x in y]))
    assert(all([x is None or x.dtype == _dtype_fft for y in imgs for x in y]))
    assert(all([x is None or x.dtype == _dtype_fft for y in templates for x in y]))

    # most of the code for cupy/scipy/numpy is shared.
    # set the module to use appropriately here.
    if use_fft == FFT_types.pyfftw_fft:
        fftw_effort = 'FFTW_MEASURE'
        fft_module = pyfftw.builders; xp = np
        dtype_fftw = str(np.dtype(_dtype_fft))
        zero_fftw = np.array(0., dtype=_dtype_fft)
        dtype_cfftw = str(np.dtype(_dtype_cfft))
        #zero_cfftw = np.array(0. + 0j, dtype=_dtype_cfft)
    elif use_fft == FFT_types.scipy_fft:
        fft_module = scipy_fft; xp = np
    elif use_fft == FFT_types.numpy_fft:
        fft_module = numpy_fft; xp = np
    elif use_fft == FFT_types.cupy_fft:
        fft_module = cupy_fft; xp = cp
    else:
        assert(False) # bad use_fft specified

    if calc_fast_len:
        poutsize = np.array([fft_module.next_fast_len(x, real=real_fft) for x in toutsize])
        tpoutsize = tuple(poutsize.tolist()) # cupy fails without explicit tuple
    else:
        poutsize = outsize
        tpoutsize = toutsize

    if real_fft:
        fftsize = (poutsize[0], poutsize[1]//2 + 1)
        f_fft2 = fft_module.rfft2; f_ifft2 = fft_module.irfft2
    else:
        fftsize = tpoutsize
        if use_fft == FFT_types.pyfftw_fft:
            f_fft2 = functools.partial(fft_module.fft2, overwrite_input=True)
            f_ifft2 = functools.partial(fft_module.ifft2, overwrite_input=True)
        else:
            f_fft2 = functools.partial(fft_module.fft2, overwrite_x=True)
            f_ifft2 = functools.partial(fft_module.ifft2, overwrite_x=True)

    # a method for dumping the images/templates and results for validation of other implementations.
    if return_comps:
        assert( not use_tri_images ) # basically always off now, did not implement
        assert( use_tri_templates ) # basically always on now, did not implement
        ncomps = (adj_matrix > 0).sum()
        comps_export = -np.ones((ncomps,2), dtype=np.int64)
        Ci_export = [None]*ncomps
        D_export = np.zeros((ncomps,2), dtype=np.int64)
        C_export = np.zeros((ncomps,), dtype=np.double)
        ccomp = 0

    # NOTE: the precomputes are only for the images, meaning when passing in precomputes the images must
    #   stay the same, but the template can change. The template-related values that are returned are
    #   only so the arrays do not have to be re-allocated, meaning the template size must also remain
    #   the same between calls using the same pre-computes.
    # NOTE: for pyfftw fft inputs and outputs need to be allocated and copied, do not optimize this.
    #   minor difference for other methods and would require far more diverging code paths.
    if Fa is None:
        # fft input allocation only used in fft precompute loop
        if use_fft == FFT_types.pyfftw_fft:
            fftA = f_fft2(pyfftw.empty_aligned(A_size, dtype=dtype_fftw, n=pyfftw_aligned),
                          s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
            # input array is resized to output, so take a slice for the input array assignment
            _A = fftA.input_array; _A[:] = zero_fftw; A = _A[:A_size[0],:A_size[1]]
        else:
            A = xp.empty(A_size, dtype=_dtype_fft)

        # image fft precompute loop
        local_sum_A = [None]*Antri; denom_A = [None]*Antri; Fb = [None]*Antri
        for z in range(Antri):
            local_sum_A[z] = [None]*nimgs; denom_A[z] = [None]*nimgs; Fb[z] = [None]*nimgs
            for x in range(nimgs):
                if Aimgs[z][x] is None: continue
                Ad = Aimgs[z][x].astype(_dtype_fft)
                local_sum_A[z][x] = local_sum(Ad,m,n)
                local_sum_A2 = local_sum(Ad*Ad,m,n)
                diff_local_sums = ( local_sum_A2 - (local_sum_A[z][x]**2)/mn )
                denom_A[z][x] = np.sqrt( np.maximum(diff_local_sums, 0) )
                A[:] = xp.asarray(Aimgs[z][x])
                Fb[z][x] = xp.empty(fftsize, dtype=_dtype_cfft)
                if use_fft == FFT_types.pyfftw_fft:
                    Fb[z][x][:] = fftA()
                else:
                    Fb[z][x][:] = f_fft2(A, s=tpoutsize)

        if use_fft == FFT_types.pyfftw_fft:
            T = Tntri*[None]; F = Tntri*[None]; fftT = Tntri*[None]; fftF = Tntri*[None]
            for z in range(Tntri):
                fftT[z] = f_fft2(pyfftw.empty_aligned(T_size, dtype=dtype_fftw, n=pyfftw_aligned),
                                 s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
                # input array is resized to output, so take a slice for the input array assignment
                _T = fftT.input_array; _T[:] = zero_fftw; T[z] = _T[:T_size[0],:T_size[1]]

                fftF[z] = f_ifft2(pyfftw.empty_aligned(fftsize, dtype=dtype_cfftw, n=pyfftw_aligned),
                                  s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
                F = fftF[z].input_array
        else:
            T = [xp.empty(T_size, dtype=_dtype_fft) for x in range(Tntri)]
            F = [xp.empty(fftsize, dtype=_dtype_cfft) for x in range(Tntri)]

        Fa = [None]*Tntri
        for z in range(Tntri):
            Fa[z] = xp.empty(fftsize, dtype=_dtype_cfft)
    #if Fa is None:

    # the returned results, peak correlation value and location.
    C = np.zeros((nimgs,nimgs), dtype=_dtype_fft)
    D0 = np.zeros((nimgs,nimgs), dtype=np.int64)
    D1 = np.zeros((nimgs,nimgs), dtype=np.int64)

    # compute correlations over all test_data tiles using precomputed train_data ffts
    denom_T = [None]*Tntri; Tdsum_mn = [None]*Tntri
    for x in range(nimgs):
        # slice the comparisons to do out of the adjacency matrix for current image
        comps = np.transpose(np.nonzero(adj_matrix[x,:])).reshape(-1)
        if comps.size == 0: continue

        for z in range(Tntri):
            T[z][:] = xp.rot90(xp.asarray(templates[z][x]), k=2) # for cupy, copy templates to device.
            if use_fft == FFT_types.pyfftw_fft:
                Fa[z][:] = fftT[z]()
            else:
                Fa[z][:] = f_fft2(T[z], s=tpoutsize)

            Td = templates[z][x].astype(_dtype_fft)
            denom_T[z] = np.sqrt(mn-1)*Td.std(dtype=_dtype_fft)
            Tdsum_mn[z] = Td.sum(dtype=_dtype_fft)/mn

        for y in comps:
            istriu = (x < y)
            Tistriu = (istriu and use_tri_templates)
            Aistriu = (istriu and use_tri_images)

            F[Tistriu][:] = Fa[Tistriu]*Fb[Aistriu][y]
            if use_fft == FFT_types.pyfftw_fft:
                xcorr_TA = np.real(fftF[Tistriu]())
            else:
                xcorr_TA = xp.real(f_ifft2(F[Tistriu], s=tpoutsize))
            cxcorr_TA = xcorr_TA[tuple([slice(sz) for sz in toutsize])] if calc_fast_len else xcorr_TA
            cxcorr_TA = cp.asnumpy(cxcorr_TA) if use_fft == FFT_types.cupy_fft else cxcorr_TA

            denom = denom_T[Tistriu]*denom_A[Aistriu][y]
            numerator = (cxcorr_TA - local_sum_A[Aistriu][y]*Tdsum_mn[Tistriu])
            # remove divide by zeros using specified tolerance, replace with zero correlation
            Ci = np.zeros(outsize, dtype=_dtype_fft)
            sel = (denom > tol); Ci[sel] = numerator[sel] / denom[sel]

            mCi = Ci[crp[0]:outsize[0]-crp[0],crp[1]:outsize[1]-crp[1]]
            C[x,y] = np.max(mCi) # the correlation peak magnitude
            # the correlation peak location in the correlation image
            deltaC = np.array(np.unravel_index(np.argmax(mCi), outsize - 2*crp)) + crp
            deltaA = deltaC - T_size + 1 # the correlation peak location in the image A
            # the shift require for peak correlation of the image that the template came from relative to the image A
            D0[x,y], D1[x,y] = deltaA - (template_offset[Tistriu][x] if template_offset is not None else 0) \
                + (img_offset[Tistriu][x] if img_offset is not None else 0)

            if doplots:
                # plotting
                print('Tsize %d %d, Asize %d %d' % (T_size[0],T_size[1],A_size[0],A_size[1]))
                print('template %d to img %d is %g at %d %d' % (x,y,C[x,y],D0[x,y],D1[x,y]))
                print('use_fft is {}'.format(str(FFT_types(use_fft)))); print()
                plt.figure(1); plt.gcf().clf()
                plt.subplot(2, 2, 1); plt.imshow(templates[Tistriu][x], cmap='gray'); plt.title("template")
                ax = plt.subplot(2, 2, 2); plt.imshow(imgs[Aistriu][y], cmap='gray'); plt.title("A")
                rect = patches.Rectangle(deltaA[::-1],T_size[1],T_size[0],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                ax = plt.subplot(2, 2, 4); plt.imshow(Ci, cmap='gray'); plt.title("xcorr")
                ax.text(deltaC[1], deltaC[0], 'x', color='red')

                if dosave_path:
                    tmp = glob.glob(os.path.join(dosave_path, '*.png'))
                    vals = [get_trailing_number(os.path.splitext(os.path.basename(x))[0]) for x in tmp]
                    vals = [x for x in vals if x is not None]
                    inext = (max(vals) + 1) if len(vals) > 0 else 0
                    # to save instead of show figures
                    fignums = plt.get_fignums()
                    for f,fi in zip(fignums, range(len(fignums))):
                        plt.figure(f)
                        plt.savefig(os.path.join(dosave_path,'xcorr_plots_{}.png'.format(inext+fi)))
                else:
                    plt.show()
            #if doplots:

            if return_comps:
                comps_export[ccomp,:] = [y, Tistriu*nimgs + x]
                Ci_export[ccomp] = mCi
                D_export[ccomp,:] = deltaC
                C_export[ccomp] = C[x,y]
                ccomp += 1

    if return_comps:
        comps_dict = {'comps':comps_export, 'Cmax':C_export, 'Camax':D_export, 'C':Ci_export}
    else:
        comps_dict = None

    if return_precomputes:
        return (C,D0,D1, Fa,Fb,T,F,fftT,fftF,local_sum_A,denom_A,comps_dict)
    else:
        return (C,D0,D1,comps_dict)

# xxx - the intention ultimately is to replace normxcorr2_fft_adj with this conversion and
#   then always use the rcc-xcorr utility for running the cross-correlations.
def normxcorr2_adj_to_comps(nimgs, adj_matrix, use_tri_templates=False, use_tri_images=False):
    #nimgs = len(imgs[0]) # python array (for tri) of python array of numpy images (of same dimensions)
    assert(adj_matrix.shape[0] == nimgs and adj_matrix.shape[1] == nimgs)

    assert( not use_tri_images ) # basically always off now, did not implement
    assert( use_tri_templates ) # basically always on now, did not implement
    ncomps = (adj_matrix > 0).sum()
    comps_export = -np.ones((ncomps,2), dtype=np.int64)
    ccomp = 0

    for x in range(nimgs):
        # slice the comparisons to do out of the adjacency matrix for current image
        comps = np.transpose(np.nonzero(adj_matrix[x,:])).reshape(-1)
        if comps.size == 0: continue

        for y in comps:
            istriu = (x < y)
            Tistriu = (istriu and use_tri_templates)
            #Aistriu = (istriu and use_tri_images)

            comps_export[ccomp,:] = [y, Tistriu*nimgs + x]
            ccomp += 1

    return comps_export

def normxcorr2_fft(template, imgs, imgs_shape=None, crp=np.array([0,0]),
                   nthreads=_template_match_nthreads, use_fft_types=_use_fft_types, use_gpu=False,
                   # these are for preserving fft precomputations on the images between template calls
                   return_precomputes=False,
                   Fa=None, Fb=None, T=None, F=None, fftT=None, fftF=None, local_sum_A=None, denom_A=None,
                   doplots=False, dosave_path=''):
    nimgs = len(imgs)
    if imgs_shape is None: imgs_shape = imgs[0].shape
    use_fft = use_fft_types[int(use_gpu)]

    # parameters - some just to preserve original matlab variable names
    Aimgs = imgs
    A_size = np.array(imgs_shape)
    T_size = np.array(template.shape)
    assert( (T_size <= A_size).all() ) # apparently this algorithm is not intended for template bigger than image
    m,n = T_size; mn = m*n
    outsize = A_size + T_size - 1
    assert( (outsize > 2*crp).all() ) # specified cropping is bigger than the output xcorr size
    toutsize = tuple(outsize.tolist()) # cupy fails without explicit tuple

    # xxx - marginally better performance with real ffts, decided not worth it for now
    #   because the implication for precision for using real ffts versus full ffts is unclear.
    real_fft = False
    # whether to zero pad for faster fft performance
    # xxx - same story here, did not really help that much, unclear if the normalization is affected.
    calc_fast_len = False
    # divide by zero tolerance
    #tol = np.finfo(_dtype_fft).eps
    tol = 1e-6

    # do not remove these, not worth the slight time savings
    assert(all([x is None or (x.shape == A_size).all() for x in imgs]))
    assert(all([x is None or x.dtype == _dtype_fft for x in imgs]))
    assert(template.dtype == _dtype_fft)

    # most of the code for cupy/scipy/numpy is shared.
    # set the module to use appropriately here.
    if use_fft == FFT_types.pyfftw_fft:
        fftw_effort = 'FFTW_MEASURE'
        fft_module = pyfftw.builders; xp = np
        dtype_fftw = str(np.dtype(_dtype_fft))
        zero_fftw = np.array(0., dtype=_dtype_fft)
        dtype_cfftw = str(np.dtype(_dtype_cfft))
        #zero_cfftw = np.array(0. + 0j, dtype=_dtype_cfft)
    elif use_fft == FFT_types.scipy_fft:
        fft_module = scipy_fft; xp = np
    elif use_fft == FFT_types.numpy_fft:
        fft_module = numpy_fft; xp = np
    elif use_fft == FFT_types.cupy_fft:
        fft_module = cupy_fft; xp = cp
    else:
        assert(False) # bad use_fft specified

    if calc_fast_len:
        poutsize = np.array([fft_module.next_fast_len(x, real=real_fft) for x in toutsize])
        tpoutsize = tuple(poutsize.tolist()) # cupy fails without explicit tuple
    else:
        poutsize = outsize
        tpoutsize = toutsize

    if real_fft:
        fftsize = (poutsize[0], poutsize[1]//2 + 1)
        f_fft2 = fft_module.rfft2; f_ifft2 = fft_module.irfft2
    else:
        fftsize = tpoutsize
        if use_fft == FFT_types.pyfftw_fft:
            f_fft2 = functools.partial(fft_module.fft2, overwrite_input=True)
            f_ifft2 = functools.partial(fft_module.ifft2, overwrite_input=True)
        else:
            f_fft2 = functools.partial(fft_module.fft2, overwrite_x=True)
            f_ifft2 = functools.partial(fft_module.ifft2, overwrite_x=True)

    # NOTE: the precomputes are only for the images, meaning when passing in precomputes the images must
    #   stay the same, but the template can change. The template-related values that are returned are
    #   only so the arrays do not have to be re-allocated, meaning the template size must also remain
    #   the same between calls using the same pre-computes.
    # NOTE: for pyfftw fft inputs and outputs need to be allocated and copied, do not optimize this.
    #   minor difference for other methods and would require far more diverging code paths.
    if Fa is None:
        # fft input allocation only used in fft precompute loop
        if use_fft == FFT_types.pyfftw_fft:
            fftA = f_fft2(pyfftw.empty_aligned(A_size, dtype=dtype_fftw, n=pyfftw_aligned),
                          s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
            # input array is resized to output, so take a slice for the input array assignment
            _A = fftA.input_array; _A[:] = zero_fftw; A = _A[:A_size[0],:A_size[1]]
        else:
            A = xp.empty(A_size, dtype=_dtype_fft)

        # image fft precompute loop
        local_sum_A = [None]*nimgs; denom_A = [None]*nimgs; Fb = [None]*nimgs
        for x in range(nimgs):
            Ad = Aimgs[x].astype(_dtype_fft)
            local_sum_A[x] = local_sum(Ad,m,n)
            local_sum_A2 = local_sum(Ad*Ad,m,n)
            diff_local_sums = ( local_sum_A2 - (local_sum_A[x]**2)/mn )
            denom_A[x] = np.sqrt( np.maximum(diff_local_sums, 0) )
            A[:] = xp.asarray(Aimgs[x])
            Fb[x] = xp.empty(fftsize, dtype=_dtype_cfft)
            if use_fft == FFT_types.pyfftw_fft:
                Fb[x][:] = fftA()
            else:
                Fb[x][:] = f_fft2(A, s=tpoutsize)

        if use_fft == FFT_types.pyfftw_fft:
            fftT = f_fft2(pyfftw.empty_aligned(T_size, dtype=dtype_fftw, n=pyfftw_aligned),
                          s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
            # input array is resized to output, so take a slice for the input array assignment
            _T = fftT.input_array; _T[:] = zero_fftw; T = _T[:T_size[0],:T_size[1]]

            fftF = f_ifft2(pyfftw.empty_aligned(fftsize, dtype=dtype_cfftw, n=pyfftw_aligned),
                           s=tpoutsize, threads=nthreads, planner_effort=fftw_effort)
            F = fftF.input_array
        else:
            T = xp.empty(T_size, dtype=_dtype_fft)
            F = xp.empty(fftsize, dtype=_dtype_cfft)
        Fa = xp.empty(fftsize, dtype=_dtype_cfft)
    #if Fa is None:

    # the returned results, peak correlation value and location.
    C = np.empty((nimgs,), dtype=_dtype_fft)
    D0 = np.empty((nimgs,), dtype=np.int64)
    D1 = np.empty((nimgs,), dtype=np.int64)

    T[:] = xp.rot90(xp.asarray(template), k=2) # for cupy, copy template to device.
    if use_fft == FFT_types.pyfftw_fft:
        Fa[:] = fftT()
    else:
        Fa[:] = f_fft2(T, s=tpoutsize)

    Td = template.astype(_dtype_fft)
    denom_T = np.sqrt(mn-1)*Td.std(dtype=_dtype_fft)
    Tdsum_mn = Td.sum(dtype=_dtype_fft)/mn

    # compute correlations over all test_data tiles using precomputed train_data ffts
    for x in range(nimgs):
        F[:] = Fa*Fb[x]
        if use_fft == FFT_types.pyfftw_fft:
            xcorr_TA = np.real(fftF())
        else:
            xcorr_TA = xp.real(f_ifft2(F, s=tpoutsize))
        cxcorr_TA = xcorr_TA[tuple([slice(sz) for sz in toutsize])] if calc_fast_len else xcorr_TA
        cxcorr_TA = cp.asnumpy(cxcorr_TA) if use_fft == FFT_types.cupy_fft else cxcorr_TA

        denom = denom_T*denom_A[x]
        numerator = (cxcorr_TA - local_sum_A[x]*Tdsum_mn)
        # remove divide by zeros using specified tolerance, replace with zero correlation
        Ci = np.zeros(outsize, dtype=_dtype_fft)
        sel = (denom > tol); Ci[sel] = numerator[sel] / denom[sel]

        mCi = Ci[crp[0]:outsize[0]-crp[0],crp[1]:outsize[1]-crp[1]]
        C[x] = np.max(mCi) # the correlation peak magnitude
        # the correlation peak location in the correlation image
        deltaC = np.array(np.unravel_index(np.argmax(mCi), outsize - 2*crp)) + crp
        deltaA = deltaC - T_size + 1 # the correlation peak location in the image A
        D0[x], D1[x] = deltaA

        if doplots:
            # plotting
            print('Tsize %d %d, Asize %d %d' % (T_size[0],T_size[1],A_size[0],A_size[1]))
            print('template to img %d is %g at %d %d' % (x,C[x],D0[x],D1[x]))
            print('use_fft is {}'.format(str(FFT_types(use_fft)))); print()
            plt.figure(1); plt.gcf().clf()
            plt.subplot(2, 2, 1); plt.imshow(template, cmap='gray'); plt.title("template")
            ax = plt.subplot(2, 2, 2); plt.imshow(imgs[x], cmap='gray'); plt.title("A")
            rect = patches.Rectangle(deltaA[::-1],T_size[1],T_size[0],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax = plt.subplot(2, 2, 3); plt.imshow(Ci, cmap='gray'); plt.title("xcorr")
            ax.text(deltaC[1], deltaC[0], 'x', color='red')

            nzC = Ci[crp[0]:outsize[0]-crp[0],crp[1]:outsize[1]-crp[1]].flat[:]
            Chist,Cbins = np.histogram(nzC, 100)
            Ccbins = Cbins[:-1] + (Cbins[1]-Cbins[0])/2
            plt.figure(2); plt.gcf().clf()
            u,s = np.mean(nzC),np.std(nzC, ddof=1)
            Chist = Chist.astype(np.double); sel = Chist > 0
            Chist[sel] = np.log10(Chist[sel]); Chist[np.logical_not(sel)] = -1
            ax = plt.subplot(1, 1, 1); plt.plot(Ccbins, Chist, 'b.-')
            plt.plot([u, u], [-1, Chist.max() + 0.5], 'g-')
            plt.plot([s, s], [-1, Chist.max() + 0.5], 'r-')
            plt.plot([6*s, 6*s], [-1, Chist.max() + 0.5], 'r--')
            plt.xlabel('correlations'); plt.ylabel('count')
            plt.title('total %d, %.3f, u %g, s %g' % (nzC.size,C[x],u,s))

            if dosave_path:
                tmp = glob.glob(os.path.join(dosave_path, '*.png'))
                vals = [get_trailing_number(os.path.splitext(os.path.basename(x))[0]) for x in tmp]
                vals = [x for x in vals if x is not None]
                inext = (max(vals) + 1) if len(vals) > 0 else 0
                # to save instead of show figures
                fignums = plt.get_fignums()
                for f,fi in zip(fignums, range(len(fignums))):
                    plt.figure(f)
                    plt.savefig(os.path.join(dosave_path,'xcorr_plots_{}.png'.format(inext+fi)))
            else:
                plt.show()

    if return_precomputes:
        return C,D0,D1, Fa,Fb,T,F,fftT,fftF,local_sum_A,denom_A
    else:
        return C,D0,D1

def template_match_preproc(_img, clahe_clipLimit=0., clahe_tileGridSize=(0,0), whiten_sigma=0.,
                           filter_size=0, convert_float=True, normalize=True, aligned_copy=True,
                           xcorr_img_size=None):
    assert(_img.ndim == 2) # expects grayscale images only

    # handle corner case of empty or very small image, particularly because this will cause the cv2 CLAHE to hang.
    if _img.shape[0] < clahe_tileGridSize[1] or _img.shape[1] < clahe_tileGridSize[0]:
        return (np.zeros(_img.shape, dtype=np.float32) if convert_float else _img)

    if filter_size > 0:
        _img = nd.median_filter(_img, size=filter_size)

    if clahe_clipLimit > 0.:
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
        _img = clahe.apply(_img)

    if convert_float:
        _img = _img.astype(np.float32); _img -= _img.min(); imgmax = _img.max()
        if imgmax > 0: _img /= imgmax

    if whiten_sigma > 0.:
        # appoximate whitening with laplacian of gaussian filter
        _img = filters.gaussian_laplace(_img, whiten_sigma)

    if convert_float and normalize:
        if xcorr_img_size is None: xcorr_img_size = _img.size
        # SO - why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize
        # see also https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
        #   you can alternatively divide each image by sqrt of number of pixels (as below).
        _img = _img - _img.mean()
        std = _img.std()
        if std > 0: _img /= (std * np.sqrt(xcorr_img_size))

    if aligned_copy and any([x == FFT_types.pyfftw_fft for x in _use_fft_types]):
        try:
            aimg = pyfftw.empty_aligned(_img.shape, _img.dtype, n=pyfftw_aligned)
            aimg.flat[:] = _img.flat[:]; _img = aimg
        except NameError:
            pass

    return _img

def pyfftw_alloc_like(x):
    return pyfftw.empty_aligned(x.shape, x.dtype, n=pyfftw_aligned)

def template_match_rotate_images(templates, imgs, rotation_step, rotation_range, interp_type=Image.NEAREST,
                                 imgs_dtype=None, imgs_shape=None, tpls_shape=None, verbose=False, doplots=False,
                                 dosave_path='', print_xcorr_every=1, return_ctemplate0=False,
                                 use_fft_types=_use_fft_types, use_gpu=False, num_gpus=0, nworkers=None,
                                 use_fft_backend=_use_fft_backend):
    ntemplates = len(templates); nimgs = len(imgs)
    if imgs_dtype is None: imgs_dtype = imgs[0].dtype
    if imgs_shape is None: imgs_shape = imgs[0].shape
    if tpls_shape is None: tpls_shape = templates[0].shape
    if nworkers is None: nworkers = _template_match_nthreads
    use_fft = use_fft_types[int(use_gpu)]

    imgs_shape = np.array(imgs_shape); imgs_size = imgs_shape[::-1]
    tpls_shape = np.array(tpls_shape)

    # xxx - assuming floating point normalized [0,1] grayscale image / template inputs
    #   old code was using other template types, decided to move everything to float for PIL support
    #     and consistency with normxcorr2_fft
    assert( issubclass(imgs_dtype.type, np.floating) ) # operate on floating point images

    # do not remove these, not worth the slight time savings
    # xxx - not implemented but this code could be modified for different shaped templates / images
    assert(all([x is None or x.dtype == imgs_dtype for x in imgs]))
    assert(all([x is None or x.dtype == imgs_dtype for x in templates]))
    assert(all([x is None or (x.shape == imgs_shape).all() for x in imgs]))
    assert(all([x is None or (x.shape == tpls_shape).all() for x in templates]))

    # get actual rotations from rotation range parameter
    rotations = np.arange(rotation_range[0],rotation_range[1] + rotation_step/2,rotation_step)
    nrotations = len(rotations)

    # iterate templates and template rotations and run template matching
    if use_fft == FFT_types.rcc_xcorr:
        natemplates = ntemplates*nrotations
        atemplates = [None]*natemplates
    else:
        Ct = np.zeros((nimgs, ntemplates, nrotations), dtype=np.double)
        D = np.zeros((nimgs, ntemplates, nrotations, 2), dtype=np.double)
    for j in range(ntemplates):
        if verbose:
            print('Matching template %d' % (j,)); t = time.time()

        # resize image here because the rotation with expand=True and scipy interpolate rotate are very slow
        #   and could not find a way to pad with nonzero value in PIL rotate.
        # pad by twice the diag of the original template, so that when the middle is cropped out the non-zero
        #   pad remains in any rotation.
        # NOTE: just so this comment is clear, without cropping out the result, the zero-padded rotation area will
        #   be visible around the edges in the rotated templates (do not want any PIL-padding in the result).
        sz = np.array(templates[j].shape)
        # calculate the diagonal and half-diagonal (used to crop the output of the PIL-rotation).
        diag = np.ceil(np.sqrt((sz*sz).sum())).astype(np.int64); diag += (diag % 2); hdiag = diag//2
        # now calculate the "double-diagonal" to use for padding the image before rotation.
        diag2 = diag*2; pad = (diag2 + ((diag2 - sz) % 2) - sz)//2
        template_shape = sz + 2*pad - diag; template_size = template_shape[::-1]
        # if the padding causes the template to get bigger than the image, then skip this template.
        if (imgs_shape < template_shape).any(): continue
        # xxx - issue of how best to pad to avoid spurious correlations
        # noise padding:
        #img_pad = 2*np.random.random(sz + 2*pad)-1; img_pad[pad[0]:-pad[0], pad[1]:-pad[1]] = templates[j]
        # constant:
        # NOTE: this is the best appraoch, but images MUST be zero mean, use normalize==True in template_match_preproc
        img_pad = np.lib.pad(templates[j], ((pad[0],pad[0]),(pad[1],pad[1])), 'constant', constant_values=0.)
        # checkerboard:
        #img_pad = np.empty(sz + 2*pad, dtype=imgs_dtype); img_pad.fill(-1.)
        #img_pad[1::2, ::2] = 1.; img_pad[::2, 1::2] = 1.; img_pad[pad[0]:-pad[0], pad[1]:-pad[1]] = templates[j]

        img_template = Image.fromarray(img_pad)

        crp = np.array([hdiag, hdiag]) # crop out the half diagonal to avoid spurious edge correlations

        for a,cnt in zip(rotations, range(nrotations)):
            degrees = a/np.pi*180
            if verbose and cnt % print_xcorr_every == 0:
                print('\tnormxcorr against template angle %.4f degrees' % (degrees,)); t = time.time()
            # PIL rotate with expand=True and scipy interpolation.rotate are very slow
            ctemplate0 = np.asarray(img_template.rotate(degrees, expand=False,
                resample=interp_type))[hdiag:-hdiag,hdiag:-hdiag]

            assert( (np.array(ctemplate0.shape) == template_shape).all() )
            # xxx - this method suffers from spurious matches right at the very edge of the full correlation output.
            #   set this crop value to ignore the diag length along the border.

            if use_fft == FFT_types.rcc_xcorr:
                atemplates[nrotations*j + cnt] = ctemplate0
            else: # if use_fft == FFT_types.rcc_xcorr:
                ctx_managers = create_scipy_fft_context_manager(use_gpu, use_fft, use_fft_backend,
                        _template_match_nthreads)
                with ExitStack() as stack:
                    for mgr in ctx_managers: stack.enter_context(mgr)

                    if cnt==0:
                        c,dy,dx, Fa,Fb,T,F,fftT,fftF,local_sum_A,denom_A = normxcorr2_fft(ctemplate0, imgs, crp=crp,
                            doplots=doplots, dosave_path=dosave_path, return_precomputes=True,
                            use_gpu=use_gpu, use_fft_types=use_fft_types)
                    else:
                        c,dy,dx = normxcorr2_fft(ctemplate0, imgs, doplots=doplots, dosave_path=dosave_path, crp=crp,
                            Fa=Fa,Fb=Fb,T=T,F=F,fftT=fftT,fftF=fftF,local_sum_A=local_sum_A,denom_A=denom_A,
                            use_gpu=use_gpu, use_fft_types=use_fft_types)

                if verbose and cnt % print_xcorr_every == (print_xcorr_every-1):
                    print('\t\tdone in %.4f s' % (time.time() - t, ))

                Ct[:,j,cnt] = c; D[:,j,cnt,:] = np.concatenate((dx[:,None], dy[:,None]), axis=1)
            #else: # if use_fft == FFT_types.rcc_xcorr:
        #for a,cnt in zip(rotations, range(nrotations)):
    #for j in range(ntemplates):

    if use_fft == FFT_types.rcc_xcorr:
        iimgs = np.tile(np.arange(nimgs), (natemplates,1)).T.reshape(-1,1)
        itmpls = np.tile(np.arange(natemplates), nimgs).reshape(-1,1)
        correlations = np.concatenate((iimgs,itmpls), axis=1)

        # it does not make sense to utilize group_correlations or multiple workers (threads) here
        #   because we are only using small batches (across angles).
        group_correlations = False; nworkers = 1

        ctx_managers = create_scipy_fft_context_manager(use_gpu, use_fft, use_fft_backend, _template_match_nthreads)
        with ExitStack() as stack:
            for mgr in ctx_managers: stack.enter_context(mgr)

            batch_correlations = BatchXCorr.BatchXCorr(imgs, atemplates, correlations,
                    normalize_input=False, group_correlations=group_correlations, crop_output=crp,
                    use_gpu=use_gpu, num_gpus=num_gpus, num_workers=nworkers,
                    disable_pbar=True, override_eps=True, custom_eps=1e-6)
            coords, peaks = batch_correlations.execute_batch()

        #deltaA = deltaC - T_size + 1 # the correlation peak location in the image A
        # swap back to normal x/y coords and convert to the correlation peak locations in the images
        coords = coords[:,::-1] - template_size[None,:] + 1

        Ct = peaks.reshape(nimgs, ntemplates, nrotations)
        D = coords.reshape(nimgs, ntemplates, nrotations, 2).astype(np.double)

    # return only the min xcorr for each image/template over the rotations
    Ca = np.argmax(Ct,2)
    Ct = np.take_along_axis(Ct, Ca[:,:,None], axis=2)[:,:,0]
    D = np.take_along_axis(D, Ca[:,:,None,None], axis=2)[:,:,0,:]

    # return the rotation invariant delta between the centers (not corners).
    D += (template_size/2 - imgs_size/2)[None,None,::-1]

    if not return_ctemplate0: ctemplate0 = None
    return Ct, Ca, D, ctemplate0

def _rasterize(pts, delta, pad):
    rpts = (pts - pts.min(0)) / delta; szA = np.ceil(rpts.max(0) + 2*pad).astype(np.int64)
    ras = np.zeros(szA[::-1], dtype=_dtype_fft) # memory error here from points too close, probably duplicate slice
    inds = np.round(rpts).astype(np.int64) + pad; ras[inds[:,1], inds[:,0]] = 1
    return ras

# xxx - this is probably more properly replaced with a set-subset rigid point matching algorithm.
#   was not able to find a straightforward candidate for this.
# this function does template matching on rasterized versions of the points to get the best-correspondence.
def points_match_trans(ptsA, ptsB, rmvduplA=False, rmvduplB=False, dperc_cutoff=1e-4, doplots=False):
    d = 10 # number of pixels to leave between the minimum point distance when rasterizing

    # move origin to top-corner (min value over points) for both point sets.
    ptsA = ptsA - ptsA.min(0); ptsB = ptsB - ptsB.min(0)
    # max range of the point sets over both dimensions.
    mrngA = ptsA.max(); mrngB = ptsB.max()

    # this is a way to remove effect of points that are basically duplicates of each other,
    #   based on the dperc_cutoff threshold of the range of the point sets.
    # a warning is printed below for points that are not addressed here.
    # xxx - replace with nearest neighbors search
    distA = scidist.pdist(ptsA); distB = scidist.pdist(ptsB)
    if rmvduplA:
        sel = np.logical_and(distA > 0, distA/d < dperc_cutoff*mrngA)
        distA[sel] = distA[distA/d >= dperc_cutoff*mrngA].min()
    if rmvduplB:
        sel = np.logical_and(distB > 0, distB/d < dperc_cutoff*mrngB)
        distB[sel] = distB[distB/d >= dperc_cutoff*mrngB].min()

    # rasterization is based on the minimum point distance in both sets.
    # exclude zero distance between identical points.
    dist = np.concatenate((distA, distB))
    sel = (dist > 0); delta = dist[sel].min() / d

    #if doplots:
    #    #step = 0.1; bins = np.arange(5,10,step)
    #    #hist,bins = np.histogram(dist.flat[:], bins)
    #    hist,bins = np.histogram(np.log10(dist.flat[:]), 100)
    #    cbins = bins[:-1] + (bins[1]-bins[0])/2
    #    plt.plot(cbins, hist)
    #    plt.xlabel('log10 dist'); plt.ylabel('count')
    #    plt.show()

    # this is to flag what is basically an error condition of two points that are very close to each
    #   other relative to other points (but not identical). this code does not handle this very well.
    if delta/max([mrngA, mrngB]) < dperc_cutoff:
        print('WARNING: min delta {} in points_match_trans very small percentage of max of point range'.format(delta))
        print('\tMost likely a very similar point is repeated.')
        print('\tAlso this will probably hang due to very large rasterized image sizes')
        sel = np.logical_and(distA > 0, distA/d < dperc_cutoff*mrngA)
        badA = np.transpose(np.nonzero(np.triu(scidist.squareform(sel))))
        sel = np.logical_and(distB > 0, distB/d < dperc_cutoff*mrngB)
        badB = np.transpose(np.nonzero(np.triu(scidist.squareform(sel))))
        print('Close points in {} ptsA (1-based):'.format(ptsA.shape[0]))
        print(badA+1)
        print('Close points in {} ptsB (1-based):'.format(ptsB.shape[0]))
        print(badB+1)

    szA = ptsA.max(0); szB = ptsB.max(0)
    intszA = np.ceil(szA/delta).astype(np.int64); intszB = np.ceil(szB/delta).astype(np.int64)
    if (intszA < intszB).all() or (intszA > intszB).all():
        padA = padB = d
    else:
        # try to force one rasterized image larger than the other.
        # xcorr template matching requires this.
        if (intszA < intszB).any():
            padA = d; padB = 2*d
        else:
            padA = 2*d; padB = d

    # rasterize the points based on the minimum point distance in both sets
    rasA = _rasterize(ptsA, delta, padA)
    rasB = _rasterize(ptsB, delta, padB)

    # blur the points incase they are not exactly aligned
    rasA = filters.gaussian_filter(rasA, d//3.); rasB = filters.gaussian_filter(rasB, d//3.)
    szA = np.array(rasA.shape); szB = np.array(rasB.shape)

    Aistemplate = (szA <= szB).all()
    if Aistemplate:
        template = rasA; img = rasB; crp = szA//2; tpts = ptsA; spts = ptsB
        tsz = szA; isz = szB; tpad = padA; ipad = padB
    else:
        template = rasB; img = rasA; crp = szB//2; tpts = ptsB; spts = ptsA
        tsz = szB; isz = szA; tpad = padB; ipad = padA
    assert( (isz > tsz).all() ) # points sets are extremely mismatching, this algorithm is not intended for this case

    #_,_,dx,dy = normxcorr2_fft(template, [img], crp=crp, doplots=doplots)
    _,dx,dy = normxcorr2_fft(template, [img], crp=crp, doplots=doplots)
    # convert to delta between original point clouds without the padding.
    tpts = tpts + (np.array([dx[0],dy[0]]) + (tpad-ipad))[::-1]*delta

    if doplots:
        plt.figure()
        plt.scatter(tpts[:,0], tpts[:,1], s=12, edgecolors='b',facecolors='none')
        plt.scatter(spts[:,0], spts[:,1], s=12, edgecolors='r',facecolors='none')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', 'datalim')
        plt.axis('off')
        plt.show()

    # xxx - replace with nearest neighbors search
    dist = scidist.cdist(spts, tpts); mapttos = np.argmin(dist, 0); mapstot = np.argmin(dist, 1)
    if Aistemplate:
        return mapttos, mapstot
    else:
        return mapstot, mapttos
