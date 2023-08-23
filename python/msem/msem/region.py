"""region.py

Class representation and alignment / stitching procedure for Zeiss multi-SEM
  regions. Regions contain multiple contiguous mFOVs within a single section
  (i.e., from a single z-plane of the original tissue block).

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
import os
import time
import glob
import re

#import scipy.spatial.distance as scidist
import scipy.sparse as sp
from scipy import signal
import scipy.stats as stats
import scipy.ndimage as nd

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing

import multiprocessing as mp
import queue

import matplotlib.pyplot as plt

import logging
LOGGER = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

from .zimages import zimages
from .mfov import mfov
from .utils import PolyCentroid, color_from_edgelist, find_histo_mode


# <<< workers for parallelizing histogram cross-correlation calculations
def compute_histos_xcorrs_full_job(ind, inds, img_nsel, dhistos, midpt, maxlag, result_queue, verbose, doplot):
    if verbose: print('\tworker%d: started' % (ind,))
    nimgs = dhistos.shape[0]; ncomps = inds.size

    # for matrices that are bigger than 32 bit, found some kind of bug in multiprocessing:
    # struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    # D = np.zeros((ncomps,nimgs), dtype=np.double)
    # Dh = np.zeros((ncomps,nimgs), dtype=np.double)
    # returning the indices individually is horrendously ineffecient, so compromise by return per row.

    for i in range(ncomps):
        #if inds[i] != 457 or not img_nsel[inds[i]]: # for debug for looking at a specific tile
        if not img_nsel[inds[i]]:
            D_i = np.zeros((nimgs,), dtype=np.double)
            Dh_i = np.zeros((nimgs,), dtype=np.double)
            for j in range(nimgs):
                if img_nsel[j] or inds[i]==j: continue
                X = signal.correlate(dhistos[inds[i],:], dhistos[j,:], mode='same')
                Xmid = X[midpt-maxlag:midpt+maxlag+1]
                Xmidmax = Xmid.max()
                #assert(Xmidmax <= 1)
                # for the full approach save distance as 1 - normalized xcorr,
                #   ranging from 0 to 2 where 0 is min distance, so we can use argsort
                #     in order to get the ntop xcorrs below.
                D_i[j] = 1-Xmidmax
                Dh_i[j] = maxlag - np.argmax(Xmid)

                #if D[i,j] < 0.2 and doplot: # for debug to view specific comparisons
                if doplot:
                    print(inds[i], j, Dh_i[j], D_i[j])
                    region.plot_histo_xcorr(dhistos[inds[i],:], dhistos[j,:], Dh_i[j], D_i[j], X, midpt, maxlag)
            #for j in range(nimgs):
        else:
            D_i = Dh_i = None
        # if not img_nsel[inds[i]]:

        res = {'D':D_i, 'Dh':Dh_i, 'iworker':ind, 'i':inds[i]}
        result_queue.put(res)
    # for i in range(ncomps):
    if verbose: print('\tworker%d: completed' % (ind,))

def compute_histos_xcorrs_sparse_job(ind, subs, nimgs, dhistos, midpt, maxlag, progress_cnt, result_queue, verbose):
    if verbose: print('\tworker%d: started' % (ind,))
    ncomps = subs.shape[0]

    D = sp.lil_matrix((nimgs,nimgs), dtype=np.double)
    Dh = sp.lil_matrix((nimgs,nimgs), dtype=np.double)

    tprog=time.time()
    for i in range(ncomps):
        X = signal.correlate(dhistos[subs[i,0],:], dhistos[subs[i,1],:], mode='same')
        Xmid = X[midpt-maxlag:midpt+maxlag+1]
        Xmidmax = Xmid.max()
        #assert(Xmidmax <= 1)
        # for the sparse approach save the normalized xcorr in the sparse distance matrix,
        #   so we can simply threshold it below using a lower cutoff.
        D[subs[i,0],subs[i,1]] = Xmidmax
        Dh[subs[i,0],subs[i,1]] = maxlag - np.argmax(Xmid)

        if verbose and i > 0 and i % progress_cnt==0:
            print('\tworker%d: cnt %d/%d in %.3f s' % (ind, i, ncomps, time.time()-tprog,))
            tprog = time.time()

    res = {'D':D, 'Dh':Dh, 'iworker':ind}
    result_queue.put(res)
    if verbose: print('\tworker%d: completed' % (ind,))

# workers for parallelizing histogram cross-correlation calculations >>>


class region(mfov):
    """Zeiss mSEM region object.

    For loading / stitching all zeiss MFoVs into a single contiguous image.

    .. note::


    """

    ### fixed parameters not exposed

    # how many tiles into neighboring mfovs to load
    overlap_radius = 2


    def __init__(self, experiment_folders, protocol_folders, region_strs, region_ind, mfov_ids=None, dsstep=1,
                 thumbnail_folders=[], brightness_balancing=False, backload_roi_poly_raw=None, mfov_align_init=True,
                 blending_mode_feathering_dist_um=1, blending_mode_feathering_min_overlap_dist_um=None,
                 overlap_correction_borders=None, use_thumbnails_ds=0, false_color_montage=False,
                 D_cutoff=None, V_cutoff=None, W_default=[None]*3, overlap_radius=None, legacy_zen_format=False,
                 C_cutoff_soft_nGMM=0, nimages_per_mfov=None, scale_nm=None, init_region_coords=True, tissue_mask_ds=1,
                 #tissue_mask_path=None, tissue_mask_ds=1, tissue_mask_fn_str=None, tissue_mask_min_edge_um=0.,
                 tissue_mask_min_edge_um=0., tissue_mask_min_hole_edge_um=0., tissue_mask_bwdist_um=0., verbose=False):
        self.region_verbose = verbose

        # new brightness balancing method between tiles in the entire region
        self.brightness_balancing = brightness_balancing

        # allow the overlap radius (between mfovs) to be overriden, mostly intended for debug
        if overlap_radius is not None: self.overlap_radius = overlap_radius

        # use the parent class to store mfov parameters and initialize coords, filenames, etc
        self.region_W_default = W_default
        mfov.__init__(self, experiment_folders, region_strs, region_ind, 1 if mfov_ids is None else mfov_ids[0],
                      overlap_radius=self.overlap_radius, overlap_correction_borders=overlap_correction_borders,
                      dsstep=dsstep, use_thumbnails_ds=use_thumbnails_ds, false_color_montage=false_color_montage,
                      thumbnail_folders=thumbnail_folders, D_cutoff=D_cutoff, V_cutoff=V_cutoff,
                      W_default=W_default[0], legacy_zen_format=legacy_zen_format, scale_nm=scale_nm,
                      nimages_per_mfov=nimages_per_mfov, init_region_coords=init_region_coords,
                      C_cutoff_soft_nGMM=C_cutoff_soft_nGMM, verbose=False)
        if self.imfov_diameter <= 1: return # make empty directory a graceful error
        # xxx - currently doing this to init member variables for this region:
        #   max_delta_zeiss, mfov_hex_coords
        #   is there an easy way to fix / avoid this? would require semi-major surgery...
        #   see also comments in mfov, a few things that are needed for getting this stuff
        #     is buried at the bottom of the call stack in zimages.
        assert( init_region_coords or not mfov_align_init ) # you can't do that
        if mfov_align_init: self.align_and_stitch(init_only=True, init_only_load_images=False)

        self.mfov_coords = None

        # total unique tiles in entire region
        self.region_ntiles = self.nmfovs*self.niTiles

        # sort mfov ids if they were specified.
        self.mfov_ids = range(self.nmfovs) if mfov_ids is None else (np.sort(mfov_ids)-1)
        self.nmfov_ids = len(self.mfov_ids)
        self._get_mfov_ids_with_neighbors()

        if not self.legacy_zen_format:
            self.s2ics = None
        else:
            # read the region ROI as it was mapped to the mSEM coordinates.
            # these coordinates are originally defined in the limi overview and input to the mSEM software.
            self.s2ics = zimages.get_s2ics_matrix(self.region_folder)
            # some new Zeiss horse%^*#, the s2ics mapping might only be saved in the first region for a wafer.
            # worse problem that required workaround is for experiment folders where the first imaged region is
            #   an error, this file does not get saved at all. optionally load from another experimental folder.
            if self.s2ics is None:
                #tmp = glob.glob(os.path.join(os.path.dirname(self.region_folder), '001_*'))
                tmp = glob.glob(os.path.join(protocol_folders[self.experimental_ind], '001_*'))
                tmp = [x for x in tmp if os.path.isdir(x)]
                assert(len(tmp) > 0) # no region 001 in protocol_folder
                self.s2ics = zimages.get_s2ics_matrix(tmp[0])
        # try to load the stage coordinates as saved during acquisition.
        try:
            if self.legacy_zen_format:
                fn = None; sep = ';'; scl = 1000
            else:
                fn = os.path.join(self.region_folder, 'ROI_coordinates.txt'); sep = None; scl = 1
            self.roi_poly_raw = zimages.get_roi_coordinates(self.region_folder, coordinate_file=fn, scl=scl, sep=sep,
                    cache_dn=self.cache_dir)
        except OSError:
            assert(self.legacy_zen_format) # not using czi files for new format so can not backload
            if backload_roi_poly_raw is None:
                print('WARNING: Zeiss ROI polygon file "region_stage_coords.csv" not found and no backload')
                self.roi_poly_raw = None
            else:
                self.roi_poly_raw = backload_roi_poly_raw
        if self.roi_poly_raw is not None and init_region_coords:
            if legacy_zen_format:
                # convert to msem pixel coordinates: per info from zeiss, the coordinates need to be inverted first.
                pts = -self.roi_poly_raw/self.scale_nm
                pts = np.dot(self.s2ics, pts.T).T
            else:
                # roi coordiantes are already in image pixel coordinates, so only adjust by the downsampling
                pts = self.roi_poly_raw/use_thumbnails_ds if use_thumbnails_ds > 0 else self.roi_poly_raw
            self.roi_poly = (pts - self.region_coords.reshape((-1,2)).min(0))/self.dsstep
            self.roi_poly_rect_ctr = (self.roi_poly.max(0) + self.roi_poly.min(0))/2
            self.roi_poly_ctr = PolyCentroid(self.roi_poly[:,0], self.roi_poly[:,1])
        else:
            self.roi_poly = self.roi_poly_rect_ctr = self.roi_poly_ctr = None

        # convert feathering distance to pixels for feathering blending in montage
        self.blending_mode_feathering_dist_pix = blending_mode_feathering_dist_um / self.scale_nm / self.dsstep * 1000

        # for converting micron params to pixels
        dsscl = 1e3 / (self.scale_nm*self.dsstep)

        # convert feathering min overlap distance to pixels for feathering blending in montage
        if blending_mode_feathering_min_overlap_dist_um is None:
            self.blending_mode_feathering_min_overlap_dist_pix = self.blending_mode_feathering_dist_pix*np.ones((2,))
        else:
            self.blending_mode_feathering_min_overlap_dist_pix = \
                    np.array(blending_mode_feathering_min_overlap_dist_um) * dsscl

        # support for loading a tissue mask to filter the keypoints
        #self.tissue_mask_path = tissue_mask_path
        #self.tissue_mask_fn_str = tissue_mask_fn_str
        # this downsampling factor should be relative to region exports
        self.tissue_mask_ds = tissue_mask_ds
        assert( self.tissue_mask_ds % self.dsstep == 0 or self.dsstep % self.tissue_mask_ds == 0 )
        tmp = tissue_mask_min_edge_um * dsscl / self.tissue_mask_ds
        self.tissue_mask_min_size = tmp*tmp
        tmp = tissue_mask_min_hole_edge_um * dsscl / self.tissue_mask_ds
        self.tissue_mask_min_hole_size = tmp*tmp
        self.tissue_mask_bwdist = tissue_mask_bwdist_um * dsscl / self.tissue_mask_ds

    def align_and_stitch_mfovs(self, get_residuals=False, twopass=False, default_tol=None, cmp_method='C',
            twopass_firstpass=False, mfovDx=None, mfovDy=None, mfovC=None, delta_coords=None,
            mfovDx_med=None, mfovDy_med=None, nworkers=1, doplots=False, dosave_path=''):
        # iterate specified set of mfovs and stitch each individually
        self.mfov_coords = [None]*self.nmfovs #; self.mfov_adjust = [None]*self.nmfovs
        self.mfov_filenames = [None]*self.nmfovs
        self.mfov_adjust = np.zeros((self.nmfovs,self.niTiles,1), dtype=np.double)
        self.mfov_max_delta_zeiss = np.zeros((self.nmfovs,2), dtype=np.double)

        if get_residuals:
            residuals = [None]*self.nmfovs; residuals_orig = [None]*self.nmfovs
            residuals_triu = [None]*self.nmfovs; residuals_xy = [None]*self.nmfovs
            xcorrs = [None]*self.nmfovs; imgvars = [None]*self.nmfovs

        # for creating single giant adjacency matrix and deltas, see below
        self._alloc_mfov_to_region_adj()

        # this is for two pass mode that supports computing medians over multiple regions/slices
        twopass_multiple_regions = mfovDx is not None
        if twopass_multiple_regions:
            assert(mfovDy is not None and mfovC is not None and delta_coords is not None)
            assert(mfovDx_med is not None and mfovDy_med is not None)
            assert(not twopass_firstpass)
            print('mfov twopass second pass using multiple slices for medians')
        if twopass_firstpass:
            assert(not twopass_multiple_regions)
            print('mfov twopass first pass to get deltas only')

        # for two-pass mode, save all the mfov results and compute the median mfov deltas
        if not twopass_multiple_regions: mfovDx = mfovDy = None
        sel_btw_mfovs = None
        if twopass:
            if self.region_verbose:
                t = time.time()

            adj_matrix = self.get_full_adj()
            nimgs = adj_matrix.shape[0]
            nadjs = adj_matrix.count_nonzero() if sp.issparse(adj_matrix) else np.count_nonzero(adj_matrix)
            subs = adj_matrix.nonzero(); inds = np.ravel_multi_index(subs, (nimgs,nimgs)); subs = np.transpose(subs)
            #subs = np.transpose(adj_matrix.nonzero())
            if not twopass_multiple_regions:
                mfovC = np.zeros((nadjs,self.nmfovs), dtype=np.double)
                mfovDx = np.zeros((nadjs,self.nmfovs), dtype=np.double)
                mfovDy = np.zeros((nadjs,self.nmfovs), dtype=np.double)
                delta_coords = [None]*self.nmfovs
            mfov_subs = [None]*self.nmfovs
            adj_matrix_btw_mfovs = np.zeros(adj_matrix.shape, dtype=bool)

            for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
                # purposely disable variance cutoff for the first pass
                D_cutoff = [[-x for x in self.D_cutoff[y]] for y in range(2)]
                cmfov = mfov(self.experiment_folder, self.region_str, self.region_ind, mfov_id+1,
                    dsstep=self.dsstep, overlap_radius=self.overlap_radius, use_thumbnails_ds=self.use_thumbnails_ds,
                    overlap_correction_borders=self.overlap_correction_borders, mfov_tri=self.mfov_tri,
                    thumbnail_folders=self.thumbnail_folder, region_coords=self.region_coords,
                    region_filenames=self.region_filenames, D_cutoff=D_cutoff, W_default=self.region_W_default[0],
                    legacy_zen_format=self.legacy_zen_format, scale_nm=self.native_scale_nm,
                    C_cutoff_soft_nGMM=self.C_cutoff_soft_nGMM, nimages_per_mfov=self.nimages_per_mfov, verbose=True)
                # uncomment for hacky way to save xcorr inputs and outputs
                #cmfov.export_xcorr_comps_path = # export path
                # save the deltas without outliers removed for the second pass.
                cdelta_coords = cmfov.align_and_stitch(doplots=doplots, dosave_path=dosave_path,
                        get_delta_coords=True, init_only=twopass_multiple_regions, nworkers=nworkers)
                # deltas coords depends on mfov neighbors (adjs for missing neighbors are removed)
                mfov_subs[cnt] = np.transpose(cmfov.adj_matrix.nonzero())
                # also return an adj matrix that only contains adjacencies for between mfovs
                adj_matrix_btw_mfovs = np.logical_or(cmfov.adj_matrix_btw_mfovs,adj_matrix_btw_mfovs)

                if not twopass_multiple_regions:
                    delta_coords[cnt] = cdelta_coords
                    mfovC[:,cnt] = cmfov.C[subs[:,0], subs[:,1]]
                    # re-create deltas from solved mfov coordinates for computing the "mfov-median" deltas.
                    # creating the medians without the outliers removed does not work well.
                    mfovDx[:,cnt] = (cmfov.xy_fixed[subs[:,0],0] - cmfov.xy_fixed[subs[:,1],0])
                    mfovDy[:,cnt] = (cmfov.xy_fixed[subs[:,0],1] - cmfov.xy_fixed[subs[:,1],1])
            #for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):

            inds_btw_mfovs = np.ravel_multi_index(np.nonzero(adj_matrix_btw_mfovs), (nimgs,nimgs))
            # select on the inds for image comparisons between different mfovs
            sel_btw_mfovs = np.in1d(inds, inds_btw_mfovs, assume_unique=True)

            if self.region_verbose:
                print('mfov pass one stitch done in %.4f s' % (time.time() - t, ))

            # compute medians over all mfovs as the new defaults
            Dx_d = np.zeros((nimgs,nimgs), dtype=np.double)
            Dy_d = np.zeros((nimgs,nimgs), dtype=np.double)
            if get_residuals:
                mfov_deltas = np.zeros((nadjs,2), dtype=np.double)
            for i in range(nadjs):
                sel = (mfovC[i,:] > 0) # ignore missing or removed adjacencies
                if twopass_multiple_regions:
                    Dx_d[subs[i,0], subs[i,1]] = mfovDx_med[i]
                    Dy_d[subs[i,0], subs[i,1]] = mfovDy_med[i]
                else:
                    if sel.sum() > 0:
                        # use the medians instead of regression
                        Dx_d[subs[i,0], subs[i,1]] = np.median(mfovDx[i,sel])
                        Dy_d[subs[i,0], subs[i,1]] = np.median(mfovDy[i,sel])
                    else:
                        # the "double-default" scenario, will default to stage coords (in second pass).
                        Dx_d[subs[i,0], subs[i,1]] = Dy_d[subs[i,0], subs[i,1]] = np.nan
                nsel = np.logical_not(sel); mfovDx[i,nsel] = np.nan; mfovDy[i,nsel] = np.nan
                if get_residuals:
                    mfov_deltas[i,:] = (Dx_d[subs[i,0], subs[i,1]], Dy_d[subs[i,0], subs[i,1]])

            if get_residuals:
                coords_imgs_rect = np.zeros((self.nTilesRect,2), dtype=np.double)
                coords = self.ring['coords']*np.array(self.images[0].shape)[::-1]
                coords_imgs_rect[self.ring['hex_to_rect'],0] = coords[:,0]
                coords_imgs_rect[self.ring['hex_to_rect'],1] = coords[:,1]
                mfov_deltas_xy = (coords_imgs_rect[subs[:,0],:] + coords_imgs_rect[subs[:,1],:])/2
                mfov_deltas_triu = (subs[:,0] < subs[:,1])
                if not twopass_firstpass:
                    # return all deltas as the diff with default deltas
                    mfovDx = mfov_deltas[:,0][:,None] - mfovDx; mfovDy = mfov_deltas[:,1][:,None] - mfovDy
                # return the default deltas as the diff with tiled deltas
                mfov_deltas = (coords_imgs_rect[subs[:,0],:] - coords_imgs_rect[subs[:,1],:]) - mfov_deltas

            # allocate for the "restitching"
            C = np.zeros((nimgs,nimgs), dtype=np.double)
            Dx = np.zeros((nimgs,nimgs), dtype=np.double)
            Dy = np.zeros((nimgs,nimgs), dtype=np.double)
        else: # if twopass:
            C = Dx = Dy = Dx_d = Dy_d = mfov_deltas = mfov_deltas_xy = mfov_deltas_triu = None

        # this is a hook to only run the first pass, so that median deltas can be computed
        #   across multiple region/slices. this may be necessary for either regions that do
        #   not contain enough mfovs, or regions that contain a lot more epon
        #   (or any non-structured/flat areas) relative to tissue.
        if twopass_firstpass: return mfovDx, mfovDy, mfovC, delta_coords

        for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
            if self.region_verbose:
                t = time.time()

            D_cutoff = [[-x for x in self.D_cutoff[y]] for y in range(2)]
            cmfov = mfov(self.experiment_folder, self.region_str, self.region_ind, mfov_id+1,
                dsstep=self.dsstep, overlap_radius=self.overlap_radius, use_thumbnails_ds=self.use_thumbnails_ds,
                overlap_correction_borders=self.overlap_correction_borders, thumbnail_folders=self.thumbnail_folder,
                region_coords=self.region_coords, region_filenames=self.region_filenames, mfov_tri=self.mfov_tri,
                D_cutoff=D_cutoff, V_cutoff=self.V_cutoff, W_default=self.region_W_default[1 if twopass else 0],
                scale_nm=self.native_scale_nm, legacy_zen_format=self.legacy_zen_format,
                C_cutoff_soft_nGMM=self.C_cutoff_soft_nGMM, nimages_per_mfov=self.nimages_per_mfov, verbose=True)

            if twopass:
                C[subs[:,0], subs[:,1]] = mfovC[:,cnt]
                Dx[mfov_subs[cnt][:,0], mfov_subs[cnt][:,1]] = delta_coords[cnt][:,0]
                Dy[mfov_subs[cnt][:,0], mfov_subs[cnt][:,1]] = delta_coords[cnt][:,1]

                if default_tol is not None:
                    Dx_t = Dx_d; Dy_t = Dy_d
                else:
                    Dx_t = Dy_t = None
            else:
                Dx_t = Dy_t = default_tol = None

            cresiduals, cresiduals_xy, cresiduals_triu, cresiduals_orig = \
                cmfov.align_and_stitch(get_residuals=True, C=C, Dx=Dx, Dy=Dy, Dx_d=Dx_d, Dy_d=Dy_d,
                        Dx_t=Dx_t, Dy_t=Dy_t, default_tol=default_tol, save_weights=True,
                        doplots=doplots, dosave_path=dosave_path)
            if get_residuals:
                residuals[mfov_id], residuals_xy[mfov_id], residuals_triu[mfov_id], residuals_orig[mfov_id] = \
                    cresiduals, cresiduals_xy, cresiduals_triu, cresiduals_orig
                xcorrs[mfov_id] = cmfov.C
                if self.V_cutoff > 0:
                    imgvars[mfov_id] = cmfov.images_rect_var
                else:
                    imgvars[mfov_id] = np.empty((cmfov.nTilesRect,), dtype=np.double); imgvars[mfov_id].fill(np.nan)
            self.mfov_coords[mfov_id] = cmfov.xy_fixed
            self.mfov_filenames[mfov_id] = cmfov.mfov_filenames

            # xxx - forget why I did this. can the tile overlap within an mfov change during an experiment?
            self.mfov_max_delta_zeiss[mfov_id,:] = cmfov.max_delta_zeiss

            # replace the weights with those for the final region stitching
            mfov_weights = np.zeros(cmfov.mfov_weights.shape, dtype=np.double)
            mfov_weights[cmfov.mfov_weights==0] = 1.
            mfov_weights[cmfov.mfov_weights==1] = self.region_W_default[2][0]
            mfov_weights[cmfov.mfov_weights==2] = self.region_W_default[2][1]
            mfov_weights[cmfov.mfov_weights==3] = self.region_W_default[2][0]

            # cmp is the value the determines which delta is selected from the overlap areas,
            #   i.e. spots where the xcorrs were run twice (for each mfov with neighboring context).
            csubs = cmfov.adj_matrix.nonzero()
            cinds = np.ravel_multi_index(csubs, cmfov.adj_matrix.shape)
            csubs = np.transpose(csubs)
            if cmp_method == 'C':
                cmp = cmfov.C.flat[cinds]
            elif cmp_method == 'res':
                cmp = -(cresiduals_orig*cresiduals_orig).sum(1)
            else:
                assert(False) # bad cmp_method string
            self._add_mfov_to_region_adj(csubs, cmfov.region_tile_ids_rect, cmfov.xy_fixed, mfov_weights, cmp)

            if self.region_verbose:
                print('\tmfov stitch done in %.4f s' % (time.time() - t, ))
        #for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):

        # # for debug to bypass the alignment, do not delete, you'll want it again - xxxalgnbypass
        #self._save_mfov_to_region_adj()

        if get_residuals:
            return residuals, residuals_xy, residuals_triu, residuals_orig, xcorrs, imgvars, \
                mfov_deltas, mfov_deltas_xy, mfov_deltas_triu, mfovDx, mfovDy, sel_btw_mfovs
    #def align_and_stitch_mfovs(self,

    # average all the mfovs together in this region
    def average_region_mfovs(self, use_heuristics=False, scale_tiled_coords_factor=1., background=0., fit_decay=False,
            mode_ratio_block_shape_um=None, mean_mfov_return_type=None, res_nm=1., bwdist_heuristic_um=0.,
            histo_nsat=[1,1], mode_limits=[[48, 248]], mode_rel=0.2, absolute_rng=None, mean_mfovs_mode_ratio=None,
            decay_params=None, offset=None, slice_histo_max_rng=[0,0], histo_smooth_size=5, doplots=False):
        # xxx - expose heuristic parameters? some are hard-coded in first and second loops

        if offset is not None: offset = np.array(offset, dtype=np.double).reshape((1,1))
        assert( offset is None ) # abondoned this for now, leave hook

        if self.region_verbose:
            print('Iterating mfovs to get image info and histograms'); t = time.time()
        doinit = True; ihistos_all = [None]*self.nmfov_ids; ihistos_all_sat = [None]*self.nmfov_ids
        for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
            # NOTE: very important here that overlap_radius is zero.
            D_cutoff = [[-x for x in self.D_cutoff[y]] for y in range(2)]
            cmfov = mfov(self.experiment_folder, self.region_str, self.region_ind, mfov_id+1,
                dsstep=self.dsstep, overlap_radius=0, use_thumbnails_ds=self.use_thumbnails_ds,
                overlap_correction_borders=self.overlap_correction_borders, thumbnail_folders=self.thumbnail_folder,
                region_coords=self.region_coords, region_filenames=self.region_filenames, mfov_tri=self.mfov_tri,
                D_cutoff=D_cutoff, V_cutoff=self.V_cutoff, W_default=self.W_default, scale_nm=self.native_scale_nm,
                legacy_zen_format=self.legacy_zen_format, nimages_per_mfov=self.nimages_per_mfov,
                C_cutoff_soft_nGMM=self.C_cutoff_soft_nGMM, verbose=False)
            cmfov.align_and_stitch(init_only=True)

            if doinit:
                coords = cmfov.mfov_hex_coords.copy()
                if scale_tiled_coords_factor > 1:
                    coords = coords - coords.mean(0)[None,:]
                    coords *= scale_tiled_coords_factor
                scale = np.array(cmfov.images[0].shape)[::-1]
                image_dtype = cmfov.images[0].dtype
                image_shape = cmfov.images[0].shape
                image_size = cmfov.images[0].size
                imgmax = np.iinfo(image_dtype).max
                summed_histos = np.zeros((imgmax+1,), dtype=np.int64)
                mfov_valid_sel = [x is not None for x in cmfov.mfov_images]
                summed_imgs_histos = np.zeros((self.niTilesRect,imgmax+1,), dtype=np.int64)
                doinit = False

                # if the mode_ratio or decay params are provided, apply them to the images beforehand.
                # this allows the median tile values to be computed on top of the mode ratio or decay adjustments.
                mean_mfovs_mode_ratio_rect = None
                if mean_mfovs_mode_ratio is not None:
                    mean_mfovs_mode_ratio_rect = np.empty([cmfov.niTilesRect]+list(image_shape),
                            dtype=mean_mfovs_mode_ratio.dtype)
                    for k,j in zip(range(cmfov.niTiles), cmfov.iring['hex_to_rect']):
                        mean_mfovs_mode_ratio_rect[j,:,:] = mean_mfovs_mode_ratio[k,:,:]
                decay_params_rect = None
                if decay_params is not None:
                    decay_params_rect = np.zeros((cmfov.nTilesRect, 2), dtype=np.double)
                    for k,j in zip(range(cmfov.niTiles), cmfov.iring['hex_to_rect']):
                        decay_params_rect[j,:] = decay_params[k,:]

            tmp, _, _ = zimages.montage(cmfov.mfov_images, coords, scale=scale, get_histos_only=True, adjust=offset,
                    img_scale_adjusts=mean_mfovs_mode_ratio_rect, img_decay_params=decay_params_rect)
            ihistos_all_sat[cnt] = tmp.copy()
            # remove saturated pixels at ends.
            tmp[:,:histo_nsat[0]] = tmp[:,histo_nsat[0]][:,None]
            tmp[:,-histo_nsat[1]:] = tmp[:,-histo_nsat[1]-1][:,None]
            sel = (tmp >= 0).all(axis=1)
            summed_histos += tmp[sel,:].sum(axis=0); ihistos_all[cnt] = tmp
            # also save summed histos for each mfov image
            summed_imgs_histos[sel,:] += tmp[sel,:]
        #for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
        if self.region_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
        assert( summed_histos.sum() > 0 ) # zero region, remove it or ask for reimage

        # heuristic mode detection that helps deal with low amount of tissue and/or artifacts.
        summed_histo_mode, smoothed_histo = find_histo_mode(summed_histos, mode_limits, mode_rel, histo_smooth_size)
        terrible_slice = (summed_histo_mode < 0)

        if terrible_slice:
            print('WARNING: really ugly region, mode {}'.format(np.argmax(summed_histos)))
            print('disabling heuristics and adjusts for terrible slice')
            histo_rng = [-1, summed_histos.size+1]
        else:
            print('Using slice mode {} for calculating heuristic cutoffs'.format(summed_histo_mode))
            # find some percentage cutoffs on either side of the mode
            maxcnt = summed_histos[summed_histo_mode]
            lomincnt = summed_histos[summed_histo_mode::-1].min()
            himincnt = summed_histos[summed_histo_mode:].min()

            # allow the lo/hi mins to be set where the curve begins increasing (if multiple peaks).
            # this only works is the curve is very smooth, but it should be for full slice histograms.
            if absolute_rng is not None and (absolute_rng[0] < 0 or absolute_rng[1] < 0):
                dsummed_histos = np.diff(smoothed_histo)
                if absolute_rng[0] < 0:
                    absolute_rng[0] = -absolute_rng[0]
                    dlomincnt = np.nonzero(dsummed_histos[summed_histo_mode-1::-1] < 0)[0]
                    if dlomincnt.size > 0:
                        dlomincnt = summed_histos[summed_histo_mode - dlomincnt[0]]
                        if dlomincnt > lomincnt: lomincnt = dlomincnt
                if absolute_rng[1] < 0:
                    absolute_rng[1] = -absolute_rng[1]
                    dhimincnt = np.nonzero(dsummed_histos[summed_histo_mode:] > 0)[0]
                    if dhimincnt.size > 0:
                        dhimincnt = summed_histos[summed_histo_mode + dhimincnt[0]]
                        if dhimincnt > himincnt: himincnt = dhimincnt

            lorng = (maxcnt - lomincnt); hirng = (maxcnt - himincnt)
            # another heuristic here to try to capture some majority percentile of the range,
            #   but also avoid any weird multimodal histograms by rejecting cutoffs that are
            #   outside of some absolutely specified range.
            # because this is a whole-slice histogram, weird histograms should be relatively rare,
            #   and typically for slices that will likely not be included anyways.
            if absolute_rng is None: absolute_rng = [0, imgmax+1] # to always use last percentage
            for x in [20,12.5,10,6,3]:
                cutlo = np.nonzero(summed_histos[summed_histo_mode::-1] <= (lorng/x + lomincnt))[0]
                cutlo = 0 if cutlo.size == 0 else (summed_histo_mode-cutlo[0])
                if cutlo >= absolute_rng[0]: break
            for x in [20,12.5,10,6,3]:
                cuthi = np.nonzero(summed_histos[summed_histo_mode:] <= (hirng/x + himincnt))[0]
                cuthi = imgmax if cuthi.size == 0 else (summed_histo_mode+cuthi[0])
                if cuthi <= absolute_rng[1]: break
            histo_rng = [cutlo, cuthi]
            if self.region_verbose:
                print('Pre slice_histo_max_rng: slice heuristic cutoffs {} {}'.format(histo_rng[0], histo_rng[1]))
            if slice_histo_max_rng[0] > 0 and cutlo < summed_histo_mode - slice_histo_max_rng[0]:
                histo_rng[0] = summed_histo_mode - slice_histo_max_rng[0]
            if slice_histo_max_rng[1] > 0 and cuthi > summed_histo_mode + slice_histo_max_rng[1]:
                histo_rng[1] = summed_histo_mode + slice_histo_max_rng[1]
            if self.region_verbose:
                print('Using slice heuristic cutoffs {} {}'.format(histo_rng[0], histo_rng[1]))

            #if doplots:
            if doplots:
                plt.figure(1); plt.gcf().clf()
                plt.plot(summed_histos)
                plt.plot([histo_rng[0],histo_rng[0]], [0,summed_histos.max()], 'r')
                plt.plot([histo_rng[1],histo_rng[1]], [0,summed_histos.max()], 'r')
                #plt.title('slice')
                plt.show()
        #else: # if terrible_slice:

        # this sets the img selects globally in case of really terrible slices.
        terrible_slice = (histo_rng[0] < 0 or histo_rng[1] > imgmax)
        if terrible_slice:
            print('WARNING: mean mfov adjust and decay disabled, check if ugly slice')
            # really, really bad region, do not include anything in mean.
            # this basically disables all the correction methods.
            mfovs_img_sel = [np.zeros(self.niTilesRect, dtype=bool)]*self.nmfov_ids
        else:
            mfovs_img_sel = [np.ones(self.niTilesRect, dtype=bool)]*self.nmfov_ids

        if use_heuristics and not terrible_slice:
            # this is another heuristic on top of connected components that dissallows large
            #   sprawling background connected components that are typicaly of membranes.
            bwdist_cutoff = bwdist_heuristic_um / self.scale_nm / self.dsstep * 1000

            # to support yet another heuristic. put a minimum value on included tiles.
            mfovs_img_sel_area = [None]*self.nmfov_ids

            if self.region_verbose:
                print('Iterating mfovs for exclusion heuristics, bwdist cutoff {} pix'.format(bwdist_cutoff))
                t = time.time()
            for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
                D_cutoff = [[-x for x in self.D_cutoff[y]] for y in range(2)]
                cmfov = mfov(self.experiment_folder, self.region_str, self.region_ind, mfov_id+1,
                    dsstep=self.dsstep, overlap_radius=0, use_thumbnails_ds=self.use_thumbnails_ds,
                    overlap_correction_borders=self.overlap_correction_borders, region_coords=self.region_coords,
                    thumbnail_folders=self.thumbnail_folder, region_filenames=self.region_filenames,
                    mfov_tri=self.mfov_tri, D_cutoff=D_cutoff, V_cutoff=self.V_cutoff, W_default=self.W_default,
                    legacy_zen_format=self.legacy_zen_format, scale_nm=self.native_scale_nm,
                    C_cutoff_soft_nGMM=self.C_cutoff_soft_nGMM, nimages_per_mfov=self.nimages_per_mfov, verbose=False)
                cmfov.align_and_stitch(init_only=True)

                # decide which image mode to use for removing images with out of range modes.
                # the mode either comes from the histogram with the saturated points removed or without
                #   based on a heuristic using a cutoff on the ratio of the sat pixels to nonsat pixels.
                #ratio_cutoff = np.inf # to disable, always use histo with sat removed to get modes
                ratio_cutoff = 1
                cmps_nosat = ihistos_all[cnt].sum(axis=1) # conservative, use the sum of the nonsat pixels
                #cmps_nosat = ihistos_all[cnt].max(axis=1) # liberal, use the peak of the nonsat pixels
                peaks_sat = ihistos_all_sat[cnt].max(axis=1) # always compare to peak of sat pixels
                modes_nosat = np.argmax(ihistos_all[cnt], axis=1)
                modes_sat = np.argmax(ihistos_all_sat[cnt], axis=1)
                ratio = np.empty(peaks_sat.shape, dtype=np.double); ratio.fill(ratio_cutoff)
                sel = (cmps_nosat > 0); ratio[sel] = peaks_sat[sel] / cmps_nosat[sel]
                sel = np.logical_and(modes_sat != modes_nosat, ratio > ratio_cutoff)
                modes = modes_nosat; modes[sel] = modes_sat[sel]

                # compute area of largest component below and above thresholds for each image.
                # if the connected area exceeds some threshold, remove.
                area_cutoff = 0.1 # xxx - parameterize?
                mfovs_img_sel[cnt] = np.ones(cmfov.nTilesRect, dtype=bool)
                mfovs_img_sel_area[cnt] = -np.ones(cmfov.nTilesRect)
                for i in range(cmfov.nTilesRect):
                    cdoplots = doplots
                    #cdoplots = doplots and cmfov.ring['rect_to_hex'][i] == 47 # for debug

                    if cmfov.mfov_images[i] is None:
                        mfovs_img_sel[cnt][i] = 0
                        continue

                    # apply the mode ratio, if it was specified
                    if mean_mfovs_mode_ratio_rect is not None:
                        cmfov.mfov_images[i] = \
                            np.clip(np.round(cmfov.mfov_images[i] / mean_mfovs_mode_ratio_rect[i,:,:]),
                                0, imgmax).astype(image_dtype)
                    # apply the decay, if it was specified
                    if decay_params_rect is not None and decay_params_rect[i,0] > 0:
                        # decay is modeled with 1/f, matches well to data
                        decay = 1./(decay_params_rect[i,0]*np.arange(image_shape[0])*res_nm + decay_params_rect[i,1])
                        idecay = np.ones(image_shape, dtype=np.double)*(decay[:,None] + 1)
                        cmfov.mfov_images[i] = \
                            np.clip(np.round(cmfov.mfov_images[i] / idecay), 0, imgmax).astype(image_dtype)

                    if cdoplots:
                        plt.figure(1); plt.gcf().clf()
                        plt.subplot(1,2,2); plt.imshow(cmfov.mfov_images[i], cmap='gray')
                    # do not include this image if its mode is outside of the per mfov image histo range
                    if modes[i] < histo_rng[0] or modes[i] > histo_rng[1]:
                        mfovs_img_sel[cnt][i] = 0
                        if cdoplots:
                            print(modes[i]); plt.title('skip'); plt.show()
                        continue

                    bw = (cmfov.mfov_images[i] < histo_rng[0])
                    if bwdist_cutoff > 0: bw = (nd.distance_transform_cdt(bw) > bwdist_cutoff)
                    labels, nlbls = nd.label(bw)
                    if nlbls > 0:
                        sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                        topn = np.argsort(sizes)[-20:]
                        tmp = sizes[topn].sum()/image_size
                        mfovs_img_sel_area[cnt][i] = max([tmp, mfovs_img_sel_area[cnt][i]])
                        if tmp > area_cutoff:
                            if cdoplots:
                                tmp = np.zeros(labels.shape, dtype=bool); tmp[np.isin(labels,topn+1)] = 1
                                plt.subplot(1,2,1); plt.imshow(tmp); plt.title('lo discard'); plt.show()
                            mfovs_img_sel[cnt][i] = 0
                    bw = (cmfov.mfov_images[i] > histo_rng[1])
                    if bwdist_cutoff > 0: bw = (nd.distance_transform_cdt(bw) > bwdist_cutoff)
                    labels, nlbls = nd.label(bw)
                    if nlbls > 0:
                        sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                        topn = np.argsort(sizes)[-20:]
                        tmp = sizes[topn].sum()/image_size
                        mfovs_img_sel_area[cnt][i] = max([tmp, mfovs_img_sel_area[cnt][i]])
                        if tmp > area_cutoff:
                            if cdoplots:
                                tmp = np.zeros(labels.shape, dtype=bool); tmp[np.isin(labels,topn+1)] = 1
                                plt.subplot(1,2,1); plt.imshow(labels); plt.title('hi discard'); plt.show()
                            mfovs_img_sel[cnt][i] = 0
                    if mfovs_img_sel[cnt][i] and cdoplots: plt.title('keep'); plt.show()
                #for i in range(cmfov.nTilesRect):
            #for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):

            # to support yet another heuristic. put a minimum value on included tiles.
            min_included = 2 # xxx - yet again, expose or not?
            if min_included > 0:
                for i in range(cmfov.nTilesRect):
                    if cmfov.mfov_images[i] is None:
                        continue

                    # add back tiles with the least area outside of histo range
                    cnt = sum([x[i] for x in mfovs_img_sel])
                    if cnt < min_included:
                        areas = np.array([x[i] for x in mfovs_img_sel_area])
                        areas_removed = (areas > area_cutoff)
                        if areas_removed.sum() > 0:
                            iareas = np.argsort(areas[areas_removed])
                            nadd = min([min_included - cnt, iareas.size])
                            tmp = np.zeros(iareas.size, dtype=bool); tmp[iareas[:nadd]] = 1
                            sadd = np.zeros(areas_removed.size, dtype=bool); sadd[areas_removed] = tmp
                            inds = np.nonzero(sadd)[0]

                            for cnt in inds: mfovs_img_sel[cnt][i] = 1
                #for i in range(cmfov.nTilesRect): # "add back" heuristic
            #if min_included > 0:

            if self.region_verbose:
                print('\tdone in %.4f s' % (time.time() - t, ))
        #if use_heuristics:

        if self.region_verbose:
            print('Iterating mfovs for mean mfov and counts and mean overall image'); t = time.time()
        doinit = True; ntotal_imgs = 0
        for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
            D_cutoff = [[-x for x in self.D_cutoff[y]] for y in range(2)]
            cmfov = mfov(self.experiment_folder, self.region_str, self.region_ind, mfov_id+1,
                dsstep=self.dsstep, overlap_radius=0, use_thumbnails_ds=self.use_thumbnails_ds,
                overlap_correction_borders=self.overlap_correction_borders, thumbnail_folders=self.thumbnail_folder,
                region_coords=self.region_coords, region_filenames=self.region_filenames, mfov_tri=self.mfov_tri,
                D_cutoff=D_cutoff, V_cutoff=self.V_cutoff, W_default=self.W_default, scale_nm=self.native_scale_nm,
                legacy_zen_format=self.legacy_zen_format, nimages_per_mfov=self.nimages_per_mfov,
                C_cutoff_soft_nGMM=self.C_cutoff_soft_nGMM, verbose=False)
            cmfov.align_and_stitch(init_only=True)

            if doinit:
                blank_image = np.zeros(image_shape, dtype=np.int64)
                ones_image = np.ones(image_shape, dtype=np.int64)
                # each mfov tile stored seperately
                mfovs_imgs = [None]*cmfov.nTilesRect
                mfovs_cnts = np.zeros(cmfov.nTilesRect, dtype=np.int64)
                for i in range(cmfov.nTilesRect):
                    mfovs_imgs[i] = None if cmfov.mfov_images[i] is None else blank_image.copy()
                summed_image = np.ones(image_shape, dtype=np.int64)
                mfovs_filenames_all = [[None]*cmfov.nTilesRect for x in range(self.nmfov_ids)]

            # preprocess the mfov image tiles before computing means
            cmfov_imgs = [None]*cmfov.nTilesRect
            for i in range(cmfov.nTilesRect):
                #cimg = cmfov.mfov_images[i].copy() # why?
                cimg = cmfov.mfov_images[i]

                if cimg is not None and not terrible_slice:
                    # apply the mode ratio, if it was specified
                    if mean_mfovs_mode_ratio_rect is not None:
                        cimg = np.clip(np.round(cimg / mean_mfovs_mode_ratio_rect[i,:,:]),
                                0, imgmax).astype(image_dtype)
                    # apply the decay, if it was specified
                    if decay_params_rect is not None and decay_params_rect[i,0] > 0:
                        # decay is modeled with 1/f, matches well to data
                        decay = 1./(decay_params_rect[i,0]*np.arange(image_shape[0])*res_nm + decay_params_rect[i,1])
                        idecay = np.ones(image_shape, dtype=np.double)*(decay[:,None] + 1)
                        cimg = np.clip(np.round(cimg / idecay), 0, imgmax).astype(image_dtype)
                # if cimg is not None and not terrible_slice:
                cmfov_imgs[i] = cimg
            #for i in range(cmfov.nTilesRect):

            # can not set images to None here, because if the image is at the edge it will change the montage size.
            images = [x if (x is None or mfovs_img_sel[cnt][y]) else blank_image \
                    for x,y in zip(cmfov_imgs, range(cmfov.nTilesRect))]
            image_tiled, _, _ = zimages.montage(images, coords, scale=scale, bg=background)
            # xxx - hacky way to get counts but keep mfovs the same size
            images = [x if x is None else (ones_image if mfovs_img_sel[cnt][y] else blank_image) \
                    for x,y in zip(cmfov_imgs, range(cmfov.nTilesRect))]
            image_count, _, _ = zimages.montage(images, coords, scale=scale)

            if doinit:
                summed_mfov = np.zeros(image_tiled.shape, dtype=np.double)
                counts_mfov = np.zeros(image_count.shape, dtype=np.double)
                doinit = False
            summed_mfov = summed_mfov + image_tiled
            counts_mfov = counts_mfov + image_count
            for i in range(cmfov.nTilesRect):
                mfovs_filenames_all[cnt][i] = cmfov.mfov_filenames[i]
                if cmfov_imgs[i] is None or not mfovs_img_sel[cnt][i]: continue
                # mean images per mfov tile (beam)
                mfovs_imgs[i] = mfovs_imgs[i] + cmfov_imgs[i]
                mfovs_cnts[i] += 1
                # overall mean image over all mfovs tiles (beams)
                summed_image = summed_image + cmfov_imgs[i]
                ntotal_imgs += 1
        #for mfov_id,cnt in zip(self.mfov_ids, range(self.nmfov_ids)):
        if self.region_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        if self.region_verbose:
            print('Getting averages and fit parameters'); t = time.time()

        # mean over all image in region
        if ntotal_imgs > 0:
            mean_image = np.round(summed_image / ntotal_imgs).astype(image_dtype)
        else:
            mean_image = np.zeros(summed_image.shape, dtype=image_dtype)
        mean_image_mode, _ = stats.mode(mean_image, axis=None, keepdims=False)

        # ratio of the image mean vs its mode
        if mean_image_mode > 0:
            mean_image_mode_ratio = (mean_image / mean_image_mode)
        else:
            mean_image_mode_ratio = np.ones(image_shape, dtype=np.double)

        # mean images at tiled mfov locations (montaged)
        sel = (counts_mfov > 0)
        mean_mfov = np.zeros(summed_mfov.shape, dtype=np.double)
        mean_mfov[sel] = summed_mfov[sel] / counts_mfov[sel]
        mean_mfov = np.round(mean_mfov).astype(image_dtype)

        # individual mean images at mfov locations
        # most successfully method so far is to use the mean image modes.
        # get adjustments based on difference from the overall mode.
        mfovs_imgs_modes = [None]*cmfov.nTilesRect
        mfovs_imgs_mode_ratio = [None]*cmfov.nTilesRect
        mfovs_imgs_adjusts = np.zeros(cmfov.nTilesRect, dtype=np.double)
        fit_decay_params = (fit_decay and decay_params is None)
        if fit_decay_params:
            print('Fitting decay parameters for each tile')
            fitted_decay_params = np.zeros((cmfov.nTilesRect, 2), dtype=np.double)
        if mode_ratio_block_shape_um is not None:
            block_shape = np.array(mode_ratio_block_shape_um) / self.scale_nm / self.dsstep * 1000
            block_shape = np.round(block_shape).astype(np.int64)
            print('Box filter on mode ratio of size {}x{} pix'.format(block_shape[0],block_shape[1]))
        else:
            block_shape = None
        for i in range(cmfov.nTilesRect):
            valid_mode_ratio = False
            if mfovs_cnts[i] > 0:
                mfovs_imgs[i] = np.round(mfovs_imgs[i] / mfovs_cnts[i]).astype(image_dtype)
                mode, _ = stats.mode(mfovs_imgs[i], axis=None, keepdims=False)
                # this is only for one optional type of returned montaged image,
                #   the actual mean mfov tile (beam) image modes.
                if mean_mfov_return_type=='mode':
                    mfovs_imgs_modes[i] = (ones_image*mode).astype(image_dtype)
                # mfovs_imgs_adjusts is the difference between average mfov tile (beam) image modes,
                #   and the mode of the overall mean image.
                mfovs_imgs_adjusts[i] = float(mean_image_mode) - mode

                # mfovs_imgs_mode_ratio is a scaling factor for each average mfov tile (beam) image,
                #   relative to it's own mode. can use this to try and correct any artifacts that are
                #   consistent for all the images acquired by a single beam.
                if mode > 0:
                    res = mfovs_imgs[i] / mode # get scaling relative to mode

                    # box filter better than gaussian for retaining artifacts.
                    if block_shape is not None:
                        res = nd.uniform_filter(res, size=block_shape, mode='reflect')

                    mfovs_imgs_mode_ratio[i] = res
                    valid_mode_ratio = True
                else:
                    mfovs_imgs_mode_ratio[i] = ones_image
            else:
                if mean_mfov_return_type=='mode':
                    mfovs_imgs_modes[i] = blank_image
                mfovs_imgs_mode_ratio[i] = ones_image

            # this is to try and fit the brightness decay from the top of the image, another brightness artifact.
            if fit_decay_params and valid_mode_ratio:
                mean_image_decay = (mfovs_imgs[i] / mode)
                # fit the mean decay in the y direction with an exponential
                mean_image_decay_y = mean_image_decay.mean(axis=1)
                # this method breaks if any values to fit are less than some small positive value (near zero).
                # xxx - how to define min value for this fit and still have this method work?
                # heuristically avoid zero and small values and also positive slopes in mean decay.
                #   positive slope in the mean decay corresponds to a negative 1/f fit.
                # but do not fit more than some default number of samples.
                nsamples_default = mean_image_decay_y.size//32
                # avoid negative values entirely.
                mean_image_decay_y = mean_image_decay_y - mean_image_decay_y[:4*nsamples_default].min()
                nsamples_min = 4
                nsamples_small = np.nonzero(mean_image_decay_y < 1e-4)[0][0]
                nsamples_pos = np.nonzero(np.diff(mean_image_decay_y) > 0)[0]
                if nsamples_pos.size > 0 and nsamples_pos[0] >= nsamples_min:
                    nsamples_pos = nsamples_pos[0]
                else:
                    nsamples_pos = nsamples_default
                nsamples = min([nsamples_default, nsamples_small, nsamples_pos])
                # only fit if the number of samples exceeds a min cutoff.
                valid_fit_decay = (nsamples >= nsamples_min)
                if valid_fit_decay:
                    Yy = mean_image_decay_y[:nsamples].reshape(-1,1)
                    Yy = 1./Yy; Xy = np.arange(nsamples)[:,None]*res_nm

                    # # normal fit
                    # clf = linear_model.LinearRegression(fit_intercept=True, copy_X=False, n_jobs=self.nthreads)
                    # clf.fit(Xy, Yy); Yyfit = clf.predict(Xy)
                    # force the fit to pass through the first point
                    clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)
                    clf.fit(Xy, Yy - Yy[0,0]); clf.intercept_ = np.array([Yy[0,0]]); Yyfit = clf.predict(Xy)

                    fit_mean_image_decay_y = \
                        1. / (clf.coef_[0] * np.arange(mean_image_decay_y.size)*res_nm + clf.intercept_)
                    # do not take fits with negative or very shallow slopes nor those with negative intercepts
                    if clf.coef_[0][0] > 0.005 and clf.intercept_[0] >= 0:
                        fitted_decay_params[i,:] = [clf.coef_[0][0], clf.intercept_[0]]
                    else:
                        valid_fit_decay = False
                if doplots:
                    print([nsamples_default, nsamples_small, nsamples_pos])
                    plt.figure(1); plt.gcf().clf()
                    plt.subplot(1,2,1)
                    plt.plot(mean_image_decay_y)
                    if valid_fit_decay:
                        plt.plot(fit_mean_image_decay_y, 'r')
                        plt.subplot(1,2,2)
                        plt.plot(Yy); plt.plot(Yyfit, 'r')
                        plt.title('tile {}, slope {}, intercept {}'.format(i,
                            fitted_decay_params[i,0], fitted_decay_params[i,1]))
                    #plt.figure(2); plt.gcf().clf()
                    #plt.imshow(mean_image_decay, cmap='gray')
                    plt.show()
            # if fit_decay_params and valid_mode_ratio:
        # for i in range(cmfov.nTilesRect):

        # this deterimines what to return for mean_mfov which is a montage of the
        #   mfov tile (beam) images and is only used for display, not to do any corrections.
        o = 100 # offset from the corner to put some form of the overall mean image.
        if mean_mfov_return_type=='mean':
            # mean_mfov already contains the montage of the mean mfov tile (beam) images.
            # put the overall mean image in the corner of the montage.
            mean_mfov[o:o+mean_image.shape[0],o:o+mean_image.shape[1]] = mean_image
        elif mean_mfov_return_type=='mode':
            tmp = [mfovs_imgs_modes[i] if mfov_valid_sel[i] else None for i in range(cmfov.nTilesRect)]
            mean_mfov, _, _ = zimages.montage(tmp, coords, scale=scale)
            # put the overall mean image mode in the corner of the montage.
            mean_mfov[o:o+mean_image.shape[0],o:o+mean_image.shape[1]] = mean_image_mode
        elif mean_mfov_return_type=='mode-ratio':
            tmp = [mfovs_imgs_mode_ratio[i] if mfov_valid_sel[i] else None for i in range(cmfov.nTilesRect)]
            mean_mfov, _, _ = zimages.montage(tmp, coords, scale=scale); mean_mfov = mean_mfov.astype(np.single)
            # put the overall mean image mode ratio in the corner of the montage.
            mean_mfov[o:o+mean_image.shape[0],o:o+mean_image.shape[1]] = mean_image_mode_ratio
        elif mean_mfov_return_type=='counts':
            mean_mfov = counts_mfov.astype(np.single)
            # put the total count in the corner of the montage.
            mean_mfov[o:o+mean_image.shape[0],o:o+mean_image.shape[1]] = ntotal_imgs
        else:
            mean_mfov = None

        # return the mfov_adjusts duplicated over all mfovs and with unrolled mfov image filenames.
        mfovs_filenames_all = [item for sublist in mfovs_filenames_all for item in sublist]
        mfovs_imgs_adjusts = np.tile(mfovs_imgs_adjusts, self.nmfov_ids)[:,None]

        # convert the mode ratios / decay params back to the hex ordering for better storage.
        # this is not needed for the adjusts, since these are written to the coordinates-style files.
        mfovs_imgs_mode_ratio_rect = mfovs_imgs_mode_ratio
        mfovs_imgs_mode_ratio = [None]*cmfov.nTilesRect
        for i,j in zip(range(cmfov.nTilesRect), cmfov.ring['rect_to_hex']):
            mfovs_imgs_mode_ratio[j] = mfovs_imgs_mode_ratio_rect[i]
        mfovs_imgs_mode_ratio = mfovs_imgs_mode_ratio[:cmfov.nTiles]

        if fit_decay_params:
            decay_params_rect = fitted_decay_params
            decay_params = np.zeros((cmfov.nTilesRect, 2), dtype=np.double)
            for i,j in zip(range(cmfov.nTilesRect), cmfov.ring['rect_to_hex']):
                decay_params[j,:] = decay_params_rect[i,:]
            decay_params = decay_params[:cmfov.nTiles,:]

        if self.region_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        return mean_mfov, mfovs_imgs_adjusts, mfovs_filenames_all, histo_rng, decay_params, mfovs_imgs_mode_ratio
    # def average_region_mfovs

    # this is to create a single giant sparse adjacency matrix for the whole region used by mfov_stitch
    def _add_mfov_to_region_adj(self, subs, region_tile_ids_rect, xy_fixed, weights, cmp):
        # new subscripts in the "global" adjacency matrix
        x = region_tile_ids_rect[subs[:,0]]; y = region_tile_ids_rect[subs[:,1]]
        # re-create deltas from solved mfov coordinates
        Dxinds = xy_fixed[subs[:,0],0] - xy_fixed[subs[:,1],0]
        Dyinds = xy_fixed[subs[:,0],1] - xy_fixed[subs[:,1],1]
        # corresponding adjacencies in current "global" adjacency matrix
        adj = self.region_adj_tiles[x,y].todense().A1
        inds = adj.nonzero()[0]

        # for existing adjacencies in the global matrix, only copy the deltas if
        #   the comparison value (correlation or -||residual||) is better.
        sel = (cmp[inds] > self.region_adj_cmp[x[inds],y[inds]].todense().A1); z = inds[sel]
        self.region_adj_cmp[x[z], y[z]] = cmp[z]
        self.region_adj_Dx[x[z], y[z]] = Dxinds[z]
        self.region_adj_Dy[x[z], y[z]] = Dyinds[z]
        self.region_weights[x[z], y[z]] = weights[z]

        # add any adjacencies not currently in the "global" adjacency matrix.
        sel = np.logical_not(adj)
        self.region_adj_tiles[x[sel],y[sel]] = 1
        self.region_adj_cmp[x[sel], y[sel]] = cmp[sel]
        self.region_adj_Dx[x[sel], y[sel]] = Dxinds[sel]
        self.region_adj_Dy[x[sel], y[sel]] = Dyinds[sel]
        self.region_weights[x[sel], y[sel]] = weights[sel]

    def _alloc_mfov_to_region_adj(self):
        n = self.region_ntiles
        self.region_adj_tiles = sp.dok_matrix((n,n), dtype=bool)
        self.region_adj_cmp = sp.dok_matrix((n,n), dtype=np.double)
        self.region_adj_Dx = sp.dok_matrix((n,n), dtype=np.double)
        self.region_adj_Dy = sp.dok_matrix((n,n), dtype=np.double)
        self.region_weights = sp.dok_matrix((n,n), dtype=np.double)

    ## for debug, do not delete, you'll want them again - xxxalgnbypass
    #def _save_mfov_to_region_adj(self):
    #    import dill
    #    d = {
    #        'region_adj_tiles':self.region_adj_tiles,
    #        'region_adj_cmp':self.region_adj_cmp,
    #        'region_adj_Dx':self.region_adj_Dx,
    #        'region_adj_Dy':self.region_adj_Dy,}
    #    with open('tmp.dill', 'wb') as f: dill.dump(d, f)
    #def _load_mfov_to_region_adj(self):
    #    import dill
    #    with open('tmp.dill', 'rb') as f: d = dill.load(f)
    #    self.region_adj_tiles = d['region_adj_tiles']
    #    self.region_adj_cmp = d['region_adj_cmp']
    #    self.region_adj_Dx = d['region_adj_Dx']
    #    self.region_adj_Dy = d['region_adj_Dy']

    def mfov_stitch(self, get_residuals=False):
        ## for debug to bypass the alignment, do not delete, you'll want it again - xxxalgnbypass
        #self._load_mfov_to_region_adj()

        # select out only the tiles being aligned (based on mfov select)
        sel_tiles = (self.region_adj_tiles.sum(0).A1 > 0)

        if self.region_verbose:
            print('Stitching %d of %d tiles (from %d mfovs) in region %d' % (sel_tiles.sum(), self.region_ntiles,
                self.nmfov_ids, self.region_ind,)); t = time.time()

        # this is only used for plotting the residuals of the region stitching.
        # NOTE: this is a second level of residuals, the current deltas were produced from the
        #   the mfovs first being individually stitched.
        if get_residuals:
            region_xy, _, _, subs, delta_coords, _ = mfov.solve_stitching(self.region_adj_tiles, self.region_adj_Dx,
                    Dy=self.region_adj_Dy, W=self.region_weights, center_sel=sel_tiles, return_inds_subs=True,
                    return_deltas=True)
            # solve_stitching always interprets adj matrix as row - col when creating diff matrix
            residuals = (region_xy[subs[:,0],:] - region_xy[subs[:,1],:]) - delta_coords
            residuals_xy = (region_xy[subs[:,0],:] + region_xy[subs[:,1],:])/2
            residuals_triu = (subs[:,0] < subs[:,1])
        else:
            region_xy, _ = mfov.solve_stitching(self.region_adj_tiles, self.region_adj_Dx, Dy=self.region_adj_Dy,
                    W=self.region_weights, center_sel=sel_tiles)

        # NOTE: this is dependent on how the "tile identifiers" were created / un-rolled in zimages,
        #  i.e., the tiles are the most quickly changing dimension.
        region_xy = region_xy.reshape((self.nmfovs, self.niTiles, 2))

        self.create_region_coords_from_inner_rect_coords(region_xy)

        if self.region_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        if get_residuals: return residuals, residuals_xy, residuals_triu

    # xxx - multiple layers of coords has gotten confusing, need some comments here to clarify what is what.
    def create_region_coords_from_inner_rect_coords(self, ihex_xy, load_neighbors=False):
        self._create_mfov_coords_from_inner_rect_coords(ihex_xy, load_neighbors=load_neighbors)
        self._create_region_coords_from_mfov_coords()

    def _get_mfov_ids_with_neighbors(self):
        if self.nmfov_ids < self.nmfovs and self.overlap_radius > 0:
            # hook to include neighbors if subset of mfovs was specified in init
            mfov_ids = np.array(self.mfov_ids)
            for self.mfov_id,i in zip(self.mfov_ids, range(self.nmfov_ids)):
                self.get_mfov_neighbors()
                mfov_ids = np.concatenate((mfov_ids, self.mfov_neighbors))
            mfov_ids = np.unique(mfov_ids)
            self.mfov_id = self.mfov_ids[0]
        else:
            mfov_ids = self.mfov_ids
        self.mfov_ids_with_neighbors = mfov_ids

        if self.false_color_montage:
            # do a graph coloring on the mfov and neighbors for mfovs being assembled.
            # this allows the overlapping colors plot to be made without any neighboring tiles
            #   using the same color (3-coloring possible with hexagonal tiling).
            mfov_id_to_ind = -np.ones((self.nmfovs,), dtype=np.int64)
            mfov_id_to_ind[mfov_ids] = np.arange(len(mfov_ids))
            mfov_graph_edges = np.zeros((0,2), dtype=np.int64)
            for self.mfov_id,i in zip(mfov_ids, range(len(mfov_ids))):
                self.get_mfov_neighbors()
                clist = self.mfov_neighbors[mfov_id_to_ind[self.mfov_neighbors] >= 0]
                mfov_graph_edges = np.concatenate((mfov_graph_edges,
                        np.concatenate((i*np.ones((len(clist),1), dtype=np.int64),
                            mfov_id_to_ind[clist][:,None]),axis=1)), axis=0)
            self.mfov_id = self.mfov_ids[0]
            if self.overlap_radius > 0:
                self.mfov_graph_colors = color_from_edgelist(mfov_graph_edges, chromatic=3)
            else:
                self.mfov_graph_colors = [0]

    def _create_mfov_coords_from_inner_rect_coords(self, ihex_xy, load_neighbors=False):
        assert( all([x == y for x,y in zip(ihex_xy.shape, (self.nmfovs, self.niTiles, 2))]) )
        self.mfov_coords_independent = self.mfov_coords
        self.mfov_coords, self.mfov_adjust_rect = [None]*self.nmfovs, [None]*self.nmfovs

        # return back to the rectangular ordering to use a common final coordinates ordering method,
        #   that is shared with the "normal mfov-only" stitching.
        m = self.omfov_diameter; r = self.overlap_radius; n = self.imfov_diameter; er = m - r
        #for mfov_id in self.mfov_ids:
        for mfov_id in self.mfov_ids_with_neighbors:
            # select out the inner mfov coords only and re-order into "rect" order typically solved in mfov.
            tmp = np.empty((self.niTilesRect,2), dtype=np.double); tmp.fill(np.nan)
            tmp[self.iring['hex_to_rect'],:] = ihex_xy[mfov_id, :, :]

            # assign back into mfov coords containing full mfov alignments (with overlapping tiles not assigned).
            self.mfov_coords[mfov_id] = np.empty((m,m,2), dtype=np.double); self.mfov_coords[mfov_id].fill(np.nan)
            self.mfov_coords[mfov_id][r:er,r:er,:] = tmp.reshape((n,n,2))
            self.mfov_coords[mfov_id] = self.mfov_coords[mfov_id].reshape((self.nTilesRect,2))

            if load_neighbors:
                # xxx - not sure if this will work with more than one mfov specified for the region
                for nbr,i in zip(self.mfov_neighbors, range(len(self.mfov_neighbors))):
                    # iterate each outer ring to load
                    for rad in range(len(self.ring['to_neighbor'])):
                        ctiles = self.ring['to_neighbor'][rad][self.mfov_neighbors_edge[i],:]
                        cntiles = self.ring['from_neighbor'][rad][self.mfov_neighbors_edge[i],:]
                        self.mfov_coords[mfov_id][self.ring['hex_to_rect'][ctiles],:] = ihex_xy[nbr, cntiles, :]

            if self.brightness_balancing and hasattr(self, 'mfov_adjust'):
                # repeat same procedure for the brightness balancing adjustments
                nd = self.mfov_adjust.shape[2]
                tmp = np.empty((self.niTilesRect,nd), dtype=np.double); tmp.fill(np.nan)
                tmp[self.iring['hex_to_rect'],:] = self.mfov_adjust[mfov_id,:,:]
                self.mfov_adjust_rect[mfov_id] = np.empty((m,m,nd), dtype=np.double)
                self.mfov_adjust_rect[mfov_id].fill(np.nan)
                self.mfov_adjust_rect[mfov_id][r:er,r:er,:] = tmp.reshape((n,n,nd))
                self.mfov_adjust_rect[mfov_id] = self.mfov_adjust_rect[mfov_id].reshape((self.nTilesRect,nd))

        # do not repeat this over mfovs, big waste of memory.
        #self.mfov_scale_adjust_rect = [None]*self.nmfovs
        if self.brightness_balancing and hasattr(self, 'mfov_scale_adjust'):
            # repeat same procedure for the brightness balancing adjustments
            s = self.mfov_scale_adjust.shape[-2:]; dtype = self.mfov_scale_adjust.dtype
            tmp = np.empty((self.niTilesRect,s[0],s[1]), dtype=dtype); tmp.fill(np.nan)
            tmp[self.iring['hex_to_rect'],:,:] = self.mfov_scale_adjust
            self.mfov_scale_adjust_rect = np.empty((m,m,s[0],s[1]), dtype=dtype)
            self.mfov_scale_adjust_rect.fill(np.nan)
            self.mfov_scale_adjust_rect[r:er,r:er,:] = tmp.reshape((n,n,s[0],s[1]))
            self.mfov_scale_adjust_rect = self.mfov_scale_adjust_rect.reshape((self.nTilesRect,s[0],s[1]))

        if self.brightness_balancing and hasattr(self, 'mfov_decay_params'):
            # repeat same procedure for the brightness balancing adjustments
            dtype = self.mfov_decay_params.dtype
            tmp = np.empty((self.niTilesRect,2), dtype=dtype); tmp.fill(np.nan)
            tmp[self.iring['hex_to_rect'],:] = self.mfov_decay_params
            self.mfov_decay_params_rect = np.empty((m,m,2), dtype=dtype)
            self.mfov_decay_params_rect.fill(np.nan)
            self.mfov_decay_params_rect[r:er,r:er,:] = tmp.reshape((n,n,2))
            self.mfov_decay_params_rect = self.mfov_decay_params_rect.reshape((self.nTilesRect,2))

        # xxx - written to hold all the max deltas, do not remember why, should be static over all mfovs, or?
        #   the value is the max difference between image coorindates within an mfov.
        #   just set it to the value for the first mfov that is loaded with single mfov init in region init.
        self.mfov_max_delta_zeiss = np.zeros((self.nmfovs,2), dtype=np.double)
        self.mfov_max_delta_zeiss[:,:] = self.max_delta_zeiss

    def _create_region_coords_from_mfov_coords(self):
        # create unrolled lists of all images and all coordinates in the entire region for montage.
        # change rectangular ordering from F-order to C-order to try to minimize overlap "banding".
        m = self.omfov_diameter; r = self.overlap_radius; er = m - r
        n = self.niTilesRect; d = self.imfov_diameter; inds = np.arange(n).reshape((d,d)).T.reshape(n)
        nd = self.mfov_adjust.shape[2] if self.brightness_balancing and hasattr(self, 'mfov_adjust') else 1
        #n = self.niTilesRect; inds = np.arange(n) # keep current rectangular ordering
        #self.nTotalTiles = self.nmfov_ids*n
        self.nTotalTiles = len(self.mfov_ids_with_neighbors)*n
        self.stitched_region_fns = [None]*self.nTotalTiles
        self.stitched_region_coords = np.zeros((self.nTotalTiles,2), dtype=np.double)
        self.flat_zeiss_region_coords = np.zeros((self.nTotalTiles,2), dtype=np.double)
        if self.false_color_montage:
            self.stitched_hex_coords = np.zeros((self.nTotalTiles,2), dtype=np.double)
        else:
            self.stitched_hex_coords = None
        self.stitched_region_adjusts = np.zeros((self.nTotalTiles,nd), dtype=np.double)
        self.stitched_region_max_delta_zeiss = np.zeros((self.nTotalTiles,2), dtype=np.double)
        self.stitched_region_scale_adjusts = self.stitched_region_decay_params = None
        # iterate in reverse mfov order to try to reduce the overlap shading between mfovs
        #for i,cnt in zip(self.mfov_ids[::-1], range(self.nmfov_ids)):
        for i,cnt in zip(self.mfov_ids_with_neighbors[::-1], range(len(self.mfov_ids_with_neighbors)-1,-1,-1)):
            slc = np.s_[cnt*n:(cnt+1)*n]

            # convert filenames to rect order, re-order for montage
            fns_rect = [None]*n
            for k,j in zip(range(self.niTiles), self.iring['hex_to_rect']):
                fns_rect[j] = self.region_filenames[i][k]
            fns_rect = [fns_rect[x] for x in inds]
            self.stitched_region_fns[slc] = fns_rect

            # select out the inner rectangular coords for this mfov and re-order for montage
            self.stitched_region_coords[slc,:] = self.mfov_coords[i].copy().reshape((m,m,2))[r:er,r:er,:].\
                    reshape((n,2))[inds,:] * self.dsstep
            if self.false_color_montage:
                self.stitched_hex_coords[slc,:] = self.mfov_hex_coords.reshape((m,m,2))[r:er,r:er,:].\
                        reshape((n,2))[inds,:] * self.dsstep
                # second dim of color_coords for false montage is an rgb offset
                self.stitched_hex_coords[slc,:][:,1] = self.mfov_graph_colors[cnt]

            # create compatible stitched coordinates using zeiss coordinates
            coords_imgs_rect = np.zeros((self.nTilesRect,2), dtype=np.double)
            crds = self.region_coords if not hasattr(self,'zeiss_region_coords') else self.zeiss_region_coords
            coords_imgs_rect[self.iring['hex_to_rect'],0] = crds[i,:,0]
            coords_imgs_rect[self.iring['hex_to_rect'],1] = crds[i,:,1]
            self.flat_zeiss_region_coords[slc,:] = coords_imgs_rect[inds,:] # re-order for montage

            if self.brightness_balancing and hasattr(self, 'mfov_adjust'):
                # select out the inner rectangular coords for this mfov and re-order for montage
                self.stitched_region_adjusts[slc,:] = \
                    self.mfov_adjust_rect[i].reshape((m,m,nd))[r:er,r:er,:].reshape((n,nd))[inds,:]

            # currently this is just one xy max per mfov, so no fancy re-ordering required
            self.stitched_region_max_delta_zeiss[slc,:] = self.mfov_max_delta_zeiss[i]

        if self.brightness_balancing and hasattr(self, 'mfov_scale_adjust'):
            s = self.mfov_scale_adjust.shape[-2:]
            # select out the inner rectangular coords for this mfov and re-order for montage
            self.stitched_region_scale_adjusts = self.mfov_scale_adjust_rect.\
                reshape((m,m,s[0],s[1]))[r:er,r:er,:].reshape((n,s[0],s[1]))[inds,:]

        if self.brightness_balancing and hasattr(self, 'mfov_decay_params'):
            # select out the inner rectangular coords for this mfov and re-order for montage
            self.stitched_region_decay_params = self.mfov_decay_params_rect.\
                reshape((m,m,2))[r:er,r:er,:].reshape((n,2))[inds,:]

    def load_stitched_region_image_brightness_adjustments(self, fn, rmv_thb=False, add_thb=False, param_dict={}):
        self.mfov_adjust, fns, param_dict = zimages.read_all_image_coords(fn, self.niTiles, ndims=-1,
                nmfovs=self.nmfovs, param_dict=param_dict, cache_dn=self.cache_dir)
        if rmv_thb:
            fns = [[None if x is None else os.path.join(os.path.dirname(x), re.sub('^thumbnail_','',
                    os.path.basename(x))) for x in y] for y in fns]
        if add_thb:
            fns = [[None if x is None else os.path.join(os.path.dirname(x),
                    'thumbnail_' + os.path.basename(x)) for x in y] for y in fns]
        return param_dict

    def tile_brightness_balancing(self, ntop=30, nspan=0, ntiles_per_img=[1,1], nchunks=1, chunksize=None, degree=1,
            dsstep=None, list_ihistos=None, histos_only=False, maxlag=32, offset=0., histo_roi_polygon=None,
            xcorrs_nworkers=1, decay=None, scale_adjust=None, L2_norm=0., histo_rng=None, res_nm=1., doplots=False):
        # these are previous balancings to run on top of, either single scalar or array of adjusts,
        #  one for each image. running on top of a previous tiled-brightness balancing is not supported.
        offset = np.array(offset).reshape(-1,1)
        if list_ihistos is not None:
            # xxx - list_ihistos is a throwback for cross-slice tile balancing, one element for each slice.
            #   maybe delete this, decided to keep for now as it's a way to have the histos computed beforehand.
            # the current slice to calculate the balancing for should come first.
            # this is to the support tile-based brightness balancing but across multiple slices
            #   in the solved order. ideally this allows for better balancing.
            nslices = len(list_ihistos)
        else:
            nslices = 1

        if self.region_verbose:
            LOGGER.info(('Tile brightness balancing with nslices=%d, ntop=%d, ntiles=%dx%d, degree=%d, nspan=%d, ' + \
                'L2_norm=%g') % (nslices, ntop, ntiles_per_img[0], ntiles_per_img[1], degree, nspan, L2_norm))

        # tiling is a special feature to also fit a surface to adjusted tiles within each image.
        # if ntiles_per_img == [1,1] this is disabled and the brightness adjust is only a constant for each image.
        ntiles = np.array(ntiles_per_img)
        nttiles = np.prod(ntiles)

        # first calculate the center points of each tile. move the edge tile points from the center to
        #   the edge of the respective dimension edge. move the corners all the way to the image corners.
        # this brute force loop was easier to read and the number of total tiles will never be very large.
        #   i.e., likely only [2,2], [3,3], [2,3] .... etc
        # this loop is similar to that in zimages. could not move this there, as the brightness matching needs
        #   to be computed before the tiles brightnesses can be fit.
        csz = np.array(self.images[0].shape)*res_nm; shp = csz // ntiles
        tile_pts = np.zeros(np.concatenate((ntiles, [2])), dtype=np.double)
        for x in range(ntiles[0]):
            for y in range(ntiles[1]):
                xy = np.array([x,y]); beg = xy*shp; end = (xy+1)*shp
                sel = np.array([x==ntiles[0]-1, y==ntiles[1]-1])
                end[sel] = csz[sel]; ctr = (end - beg)/2 + beg
                if x > 0:
                    tile_pts[x,y,0] = end[0] if x == ntiles[0]-1 else ctr[0]
                if y > 0:
                    tile_pts[x,y,1] = end[1] if y == ntiles[1]-1 else ctr[1]
        tile_pts = tile_pts.reshape((-1,2))

        if list_ihistos is None:
            if self.region_verbose:
                LOGGER.info('Running montage to get image histograms'); t = time.time()
            ihistos, _, _ = self.montage(dsstep=dsstep, get_histos_only=True, adjust=offset, decay=decay,
                    scale_adjust=scale_adjust, ntiles_per_img=ntiles_per_img, histo_roi_polygon=histo_roi_polygon)
            if self.region_verbose:
                LOGGER.info('\tdone in %.4f s' % (time.time() - t, ))
            if histos_only: return ihistos
        else:
            ihistos = np.concatenate(list_ihistos, axis=0)
        ngrayscales = ihistos.shape[-1]
        flatihistos = ihistos.reshape((-1,ngrayscales))
        if list_ihistos is None:
            imgtiles_shape = ihistos.shape[:-1]
            nimgtiles = flatihistos.shape[0]
        else:
            imgtiles_shape = list_ihistos[0].shape[:-1]
            nimgtiles = np.array(list_ihistos[0].shape[:-1]).prod()

        # xxx - some debug stuff for finding specific tile, maybe delete?
        #tmp = [('_000005_035' in x) if x is not None else False for x in self.stitched_region_fns]
        ##tmp = [('_000001_087' in x) if x is not None else False for x in self.stitched_region_fns]
        #print(tmp.index(True))
        #print(len(self.stitched_region_fns))
        #print(flatihistos.shape)

        # calcalate the brightness balancing using all of the available histograms.
        #   this is either a single brightness for each image, or one for each tile of each image.
        # the balancing can also optinoally be broken into randomized chunks (for memory constraints).
        if nchunks==1 and chunksize is None:
            # a slight speed and readability optimization, the loop below should also work with nchunks==1
            adjusts = region.img_brightness_balancing(flatihistos, ntop=ntop, nspan=nspan, maxlag=maxlag,
                    nworkers=xcorrs_nworkers, L2_norm=L2_norm, histo_rng=histo_rng, verbose=self.region_verbose)
        else:
            # optionally create randomly permuted chunks. need this if this matrix becomes quite large.
            # typically for a normal sized slice and ntiles_per_img more than 2x2, this is required.
            ntotalimgtiles = flatihistos.shape[0]
            nvalid_imgs = (flatihistos >= 0).all(1).sum()
            if chunksize is not None:
                # calculate nchunks based on desired chunk size.
                # there can be invalid images or tiles in the histograms:
                #   (1) to maintain valid rectangular mfov indexing
                #   (1) tiles ignore when histograms were calculated because they are outside the roi polygon
                nchunks = max([np.round(nvalid_imgs/chunksize).astype(np.int64), 1])
            LOGGER.info('\tchunking balancing with nchunks=%d' % (nchunks,))

            ichunks = np.array_split(np.arange(ntotalimgtiles), nchunks)
            inds = np.random.permutation(ntotalimgtiles)
            adjusts = np.zeros((ntotalimgtiles,), dtype=np.double)
            for i in range(nchunks):
                beg = ichunks[i][0]; end = ichunks[i][-1]+1
                # calculate least squares brightness balancing for all tiles in current chunk.
                adjusts[inds[beg:end]] = region.img_brightness_balancing(flatihistos[inds[beg:end],:], ntop=ntop,
                        nspan=nspan, maxlag=maxlag, nworkers=xcorrs_nworkers, L2_norm=L2_norm, histo_rng=histo_rng,
                        verbose=self.region_verbose)

        # if this is cross-slice balancing, only take the adjusts for the current slice
        adjusts = adjusts[:nimgtiles]

        if nttiles > 1:
            if self.region_verbose:
                LOGGER.info('Fitting image tiled adjustments'); t = time.time()
            adjusts = adjusts.reshape(imgtiles_shape)
            poly = preprocessing.PolynomialFeatures(degree=degree)
            clf = linear_model.LinearRegression(fit_intercept=False, n_jobs=self.nthreads)
            # error here something like
            # ValueError: Found input variables with inconsistent numbers of samples: [9, 4]
            # means that you need to recompute the tile histograms
            tile_pts_fit = poly.fit_transform(tile_pts)
            nimgs = imgtiles_shape[0]
            self.stitched_region_adjusts = np.zeros((nimgs,tile_pts_fit.shape[1]), dtype=np.double)
            # add back the offset, as the histograms were shifted above
            adjusts_offset = np.repeat(offset, nimgs) if offset.size == 1 else offset
            for i in range(nimgs):
                clf.fit(tile_pts_fit, (adjusts[i,:,:] + adjusts_offset[i]).flat[:][:,None])
                self.stitched_region_adjusts[i,:] = clf.coef_.copy()
            if self.region_verbose:
                LOGGER.info('%s', '\tdone in %.4f s' % (time.time() - t, ))
        else:
            # add back the offset, as the histograms were shifted above
            adjusts = adjusts + offset.reshape(-1)
            self.stitched_region_adjusts = adjusts[:,None]

        return ihistos

    @staticmethod
    def plot_histo_xcorr(histo1, histo2, Dh, D, X, midpt, maxlag):
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(histo1); plt.plot(histo2)
        apt = np.argmax(histo1); pt = np.max(histo2)
        plt.plot([apt, apt+Dh], [pt+0.01, pt+0.01], 'r', linewidth=2)
        plt.title('%d %.3f' % (Dh, D))
        plt.subplot(1,2,2)
        plt.plot(X)
        plt.plot([np.argmax(X)], [0], 'rx')
        plt.plot([midpt-maxlag, midpt-maxlag], [0, 2], 'r', linewidth=2)
        plt.plot([midpt+maxlag, midpt+maxlag], [0, 2], 'r', linewidth=2)
        plt.show()

    @staticmethod
    def img_brightness_balancing(ihistos, ntop=30, nspan=0, maxlag=32, adj_matrix=None, label_adj_min=-1,
            nworkers=1, L2_norm=0., histo_rng=None, absolute_rng=[32,252], regr_bias=False, verbose=False):
        nimgs = ihistos.shape[0]
        assert(ihistos.shape[1] == 256) # function is parameterized for 8bit grayscale data only
        # these are to specify "missing" images (or those not not be included).
        img_nsel = (ihistos < 0).all(1)

        # originally this was just for removing saturated pixels.
        # decided to change to removing parts of the histogram outside of an optionally specified range,
        #   but further bounded by an absolute range.
        use_histo_rng = list(histo_rng) if histo_rng is not None else list(absolute_rng)
        if use_histo_rng[0] < absolute_rng[0]: use_histo_rng[0] = absolute_rng[0]
        if use_histo_rng[1] > absolute_rng[1]: use_histo_rng[1] = absolute_rng[1]

        # replace with the mean value so these pixels have no effect on the xcorrs
        dhistos = ihistos.astype(np.double) # do xcorrs with floats, need to normalize
        m = dhistos[:,use_histo_rng[0]:use_histo_rng[1]].mean(1)[:,None]
        dhistos[:,:use_histo_rng[0]] = m; dhistos[:,use_histo_rng[1]:] = m

        # do not utilize or adjust any completely uniform tiles.
        std = np.std(dhistos, axis=1); ssel = (std > 0)
        img_bad_sel_std = np.logical_and(np.logical_not(img_nsel), np.logical_not(ssel))
        if verbose:
            nbad = img_bad_sel_std.sum()
            if nbad > 0: print('WARNING: %d completely uniform image tiles' % (nbad,))
        img_nsel = np.logical_or(img_nsel, img_bad_sel_std)
        # normalize the histos for xcorrs so we can compute a normalized cross-correlation at multiple lags
        # SO - why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize
        # histogram lengths are 256 for 8bit grayscale, sqrt(256) is 16
        dhistos[ssel,:] = (dhistos[ssel,:] - np.mean(dhistos, axis=1)[ssel,None]) / (std[ssel,None] * 16)

        # local parameters
        midpt = 128 # for 8bit grayscale images
        maxD = 2. # maximum distance value, only used if adj matrix is not provided

        img_sel = np.logical_not(img_nsel)
        if verbose:
            print('Getting pairwise bounded xcorr distances/lags of %d valid histograms' % (img_sel.sum()))
            t = time.time()
        adj_matrix_provided = (adj_matrix is not None)
        if not adj_matrix_provided:
            # do all the pairwise comparisons
            D, Dh = region._helper_run_par_histo_full_xcorrs(dhistos, midpt, nworkers, img_nsel, maxlag,
                    verbose, doplot=False)
        else: # if not adj_matrix_provided
            D, Dh = region._helper_run_par_histo_sparse_xcorrs(dhistos, adj_matrix, midpt, nworkers, img_nsel,
                    maxlag, verbose, doplot=False)
        # if not adj_matrix_provided
        if verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        if not adj_matrix_provided:
            # replace the missing images and the diagonal with max distances
            #   whilst ignoring the "missing" rect images.
            D[img_nsel,:] = maxD; D[:,img_nsel] = maxD; np.fill_diagonal(D, maxD)

            if verbose:
                print('Getting adjacency matrix based on top %d cross-correlations' % (ntop,))

            # for each image get the top n-closest images in terms of similar histograms.
            # create an adjacency matrix based on the top-n most similar histograms.
            amin = np.argsort(D, 1)
            indsx = np.concatenate([np.arange(nimgs, dtype=np.int32)[:,None] \
                for x in range(ntop)], axis=1).flat[:]
            indsy = amin[:,:ntop].flat[:]
            adj_matrix = np.zeros((nimgs,nimgs), dtype=bool)
            adj_matrix[indsx,indsy] = 1; adj_matrix[indsy,indsx] = 1

            if nspan > 0:
                # the idea here is to force a single connected component for the adjacency matrix.
                # depending on the use case, this might be more ideal.
                if verbose:
                    print('Append adjacencies for %d minimum (xcorr distance) spanning trees' % (nspan,))
                spanD = D.copy()
                for i in range(nspan):
                    cD = sp.csgraph.minimum_spanning_tree(spanD)
                    inds = np.transpose(cD.nonzero())
                    adj_matrix[inds[:,0],inds[:,1]] = 1
                    # remove the current MST for the next iteration
                    spanD[inds[:,0],inds[:,1]] = maxD

            # be certain that missing / invalid images and diagonal are not selected
            adj_matrix[img_nsel,:] = 0; adj_matrix[:,img_nsel] = 0; np.fill_diagonal(adj_matrix, 0)
        # if not adj_matrix_provided:

        # get least squares solution for best matching histogram statistic,
        #   depending on what is calculated above for the histograme statistic to use.
        if verbose:
            print('Calculating least squares brightness balancing for region, nadjs=%d' % (adj_matrix.sum(),))
            t = time.time()
        if label_adj_min >= 0:
            if verbose:
                print('Using connected compents centering with min component size {}'.format(label_adj_min))
            center_sel = None
        else:
            if verbose:
                print('Using valid image select for centering')
            center_sel = img_sel

        try:
            brightness_adjusts, _ = mfov.solve_stitching(adj_matrix, Dh, center_sel=center_sel, regr_bias=regr_bias,
                label_adj_min=label_adj_min, l2_alpha=L2_norm)
            brightness_adjusts = brightness_adjusts.reshape(-1)
        except np.linalg.LinAlgError:
            print('WARNING: brightness solver did not converge, setting adjusts to zero')
            brightness_adjusts = np.zeros((nimgs,), dtype=np.double)
        if verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

        # set to zero anything that comes back much larger than expected.
        # this can happen in the case of multiple connected components in the adj matrix.
        # specify label_adj_min to account for this within solve_stitching.
        brightness_adjusts[np.abs(brightness_adjusts) > 2*maxlag] = 0

        return brightness_adjusts

    @staticmethod
    def _helper_run_par_histo_full_xcorrs(dhistos, midpt, nworkers, img_nsel, maxlag, verbose, doplot=False):
        nimgs = dhistos.shape[0]

        workers = [None]*nworkers
        result_queue = mp.Queue(nimgs)
        if doplot:
            nworkers = 1 # force single thread for the plotting validation / debug
            inds = np.array_split(np.arange(nimgs), nworkers)
        else:
            # permutation for better balancing.
            # NOTE: do NOT use this if we go to multiple procs
            inds = np.array_split(np.random.permutation(nimgs), nworkers)
        if verbose: print('starting %d worker jobs' % (nworkers,))
        for i in range(nworkers):
            workers[i] = mp.Process(target=compute_histos_xcorrs_full_job, daemon=True,
                    args=(i, inds[i], img_nsel, dhistos, midpt, maxlag, result_queue, verbose, doplot))
            workers[i].start()
        # NOTE: only call join after queue is emptied

        D = np.zeros((nimgs,nimgs), dtype=np.double)
        Dh = np.zeros((nimgs,nimgs), dtype=np.double)
        dt = time.time()
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        nprint = int(5e8 / nimgs)
        # results are returned per row because returning entire matrix can run into a 32-bit size limitation error
        #   with the matrices for large nimgs and returning per element is horrendously inefficient.
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for i in range(nimgs):
        i = 0
        while i < nimgs:
            if verbose and i>0 and i%nprint==0:
                print('{} through q in {:.2f} s, worker_cnts:'.format(nprint,time.time()-dt,)); dt = time.time()
                print(worker_cnts)

            try:
                res = result_queue.get(block=True, timeout=zimages.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != inds[x].size:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / {}'.format(x,worker_cnts[x],inds[x].size))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            if res['D'] is not None:
                D[res['i'],:] = res['D']
                Dh[res['i'],:] = res['Dh']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        return D, Dh

    @staticmethod
    def _helper_run_par_histo_sparse_xcorrs(dhistos, adj_matrix, midpt, nworkers, img_nsel, maxlag, verbose,
            doplot=False):
        nimgs = dhistos.shape[0]
        progress_cnt = 5e6

        # adj_matrix passed in from outside
        if verbose: print('Using provided adj matrix')
        assert( all([adj_matrix.shape[x] == nimgs for x in [0,1]]) )
        adj_matrix[:,img_nsel] = 0; adj_matrix[img_nsel,:] = 0 # remove missing or invalid images
        adj_matrix.setdiag(0) # remove diagonal

        # get the nonzero subscripts and unraveled indices
        adj_subs = np.transpose(adj_matrix.nonzero())

        print('Computing %d total histogram xcorrs (~%d pairwise)' % \
            (adj_subs.shape[0], int(np.sqrt(adj_subs.shape[0]))))
        print('starting %d worker jobs' % (nworkers,))
        workers = [None]*nworkers
        result_queue = mp.Queue(nimgs)
        subs = np.array_split(adj_subs, nworkers)
        for i in range(nworkers):
            workers[i] = mp.Process(target=compute_histos_xcorrs_sparse_job, daemon=True,
                    args=(i, subs[i], nimgs, dhistos, midpt, maxlag, progress_cnt, result_queue, verbose))
            workers[i].start()
        # NOTE: only call join after queue is emptied

        # because these matrices are sparse, did not run into any MP queue pickling issue like for pairwise.
        # so the entire matrices for each woker are returned at once.
        D = sp.lil_matrix((nimgs,nimgs), dtype=np.double)
        Dh = sp.lil_matrix((nimgs,nimgs), dtype=np.double)
        worker_cnts = np.zeros((nworkers,), dtype=np.int64)
        dead_workers = np.zeros((nworkers,), dtype=bool)
        #for i in range(nworkers):
        i = 0
        while i < nworkers:
            try:
                res = result_queue.get(block=True, timeout=zimages.queue_timeout)
            except queue.Empty:
                for x in range(nworkers):
                    if not workers[x].is_alive() and worker_cnts[x] != 1:
                        if dead_workers[x]:
                            print('worker {} is dead and worker cnt is {} / 1'.format(x,worker_cnts[x]))
                            assert(False) # a worker exitted with an error or was killed without finishing
                        else:
                            # to make sure this is not a race condition, try the queue again before error exit
                            dead_workers[x] = 1
                continue

            # the indices are not overlapping for the sparse matrices, so we can just add them
            D = D + res['D']
            Dh = Dh + res['Dh']
            worker_cnts[res['iworker']] += 1
            i += 1
        assert(result_queue.empty())
        [x.join() for x in workers]
        [x.close() for x in workers]

        return D, Dh

    def montage(self, zeiss=False, dsstep=None, get_histos_only=False, blending_mode='None', crop_size=[0,0],
                ntiles_per_img=[1,1], histo_roi_polygon=None, adjust=None, res_nm=1., decay=None,
                scale_adjust=None, nblks=[1,1], iblk=[0,0], novlp_pix=[0,0], get_overlap_sum=False):
        assert( not get_histos_only or blending_mode == 'None' ) # not-compatible options
        if dsstep is None: dsstep=self.dsstep
        if self.region_verbose:
            msg = ('Loading and montaging region %d with %d total images' % (self.region_ind+1,self.nTotalTiles))
            LOGGER.info("%s",msg)
            t = time.time()
        if adjust is None:
            adjust = self.stitched_region_adjusts if self.brightness_balancing else None
        img_scale_adjusts = self.stitched_region_scale_adjusts if self.brightness_balancing else None
        img_decay_params = self.stitched_region_decay_params if self.brightness_balancing else None
        border_correct = None if not self.overlap_correction and not zeiss else self.overlap_correction_borders
        coords = self.stitched_region_coords if not zeiss else self.flat_zeiss_region_coords

        overlap_sum=None
        if blending_mode == "feathering" or get_overlap_sum:
            LOGGER.info('Need to montage twice, first time just to get overlap counts')
            overlap_sum, corners, _ = zimages.montage(self.stitched_region_fns, coords / dsstep,
                image_load={'folder':self.images_load_folder, 'ovlp_sel':None, 'crop':self.border_crop,
                            'max_delta':self.stitched_region_max_delta_zeiss, 'invert':self.invert_images,
                            'decay':None, 'scale_adjust':None, 'dsstep':dsstep, 'reduce':self.blkrdc_func, },
                verbose_mod=10000 if self.region_verbose else None, get_overlap_sum_only=True, crop_size=crop_size,
                nblks=nblks, iblk=iblk, novlp_pix=novlp_pix, cache_dn=self.cache_dir)

        image, corners, crop_info = zimages.montage(self.stitched_region_fns, coords / dsstep,
                image_load={'folder':self.images_load_folder, 'ovlp_sel':border_correct, 'crop':self.border_crop,
                        'max_delta':self.stitched_region_max_delta_zeiss, 'invert':self.invert_images,
                        'dsstep':dsstep, 'decay':decay, 'scale_adjust':scale_adjust, 'reduce':self.blkrdc_func, },
                adjust=adjust, verbose_mod=500 if self.region_verbose else None, blending_mode=blending_mode,
                blending_mode_feathering_dist=self.blending_mode_feathering_dist_pix, overlap_sum=overlap_sum,
                blending_mode_feathering_min_overlap_dist=self.blending_mode_feathering_min_overlap_dist_pix,
                get_histos_only=get_histos_only, crop_size=crop_size, histo_ntiles=ntiles_per_img,
                histo_roi_polygon=histo_roi_polygon, img_scale_adjusts=img_scale_adjusts, unique_str=self.region_str,
                cache_dn=self.cache_dir, nblks=nblks, iblk=iblk, novlp_pix=novlp_pix, res_nm=res_nm,
                color_coords=self.stitched_hex_coords, img_decay_params=img_decay_params)
        if self.region_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
        if get_overlap_sum:
            return image, corners, crop_info, overlap_sum
        else:
            return image, corners, crop_info
