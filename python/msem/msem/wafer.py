"""wafer.py

Class representation for Zeiss multi-SEM "wafers" which is a collection of
  sections. Currently assumes that each section contains mFOVs that form a
  contiguous region.
Wafers contain multiple sections, each of which is a single z-slice out of
  the original tissue block.

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

# xxx - "imaged order" is a misnomer again, as this now actually means "indexed order"
#   meaning the index is an index into the section lists that define each wafer.
#   (see creation / export in def_common_params)

import numpy as np
import os
import time
#import glob
import configparser
import traceback

import scipy
import scipy.ndimage as nd
import scipy.interpolate as interp
import scipy.linalg as lin

#try:
#    import mkl
#    mkl_imported=True
#except:
#    print('WARNING: mkl module unavailable, can not mkl.set_num_threads')
#    mkl_imported=False

# try:
#     import cupy as cp
# except:
#     print('WARNING: cupy unavailable, needed for one method for template matching with cross-correlations')

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import linear_model, preprocessing
from sklearn.neighbors import NearestNeighbors

import cv2
import skimage.measure as measure
import tifffile
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

from aicspylibczimsem import CziMsem

from .region import region
from ._template_match import template_match_rotate_images, template_match_preproc
from ._template_match import points_match_trans, pyfftw_alloc_like
from .procrustes import RigidRegression
from .zimages import msem_input_data_types

from .utils import constrainAngle, PolyCentroid, tile_nblks_to_ranges, block_construct
from .utils import label_fill_image_background, fill_outside_polygon, cached_image_load
#from .utils import mls_rigid_transform, delta_interp_methods
#from .utils import make_delta_plot #, make_grid_plot
from .utils import big_img_load, big_img_save, big_img_info, big_img_init, gpfs_file_unlock

class wafer(region):
    """Zeiss mSEM region object.

    For z-alignment of multiple zeiss regions (sections) into 3D image volumes.

    .. note::


    """

    ### fixed parameters not exposed

    # interpolation mode to use for PIL
    # NEAREST, BILINEAR, BICUBIC
    #region_interp_type = Image.NEAREST
    region_interp_type = Image.BILINEAR

    # interpolation mode to use for ndimage map_coordinates (for fine alignment warping).
    warping_spline_order_nd = 1 # BILINEAR

    # cutoff for preprocessing the whole region image instead of each grid crop
    #ngrid_cutoff = 50
    ngrid_cutoff = 0 # just disable this, always preprocess whole region (or block)

    # whether or not to copy out the crops from the images and processed images.
    # will improve pyfftw speed, but at a significant memory usage cost.
    image_crops_copy = False

    coords_file_suffix = '_coords.txt'

    # set True to fill background areas (typically with noise).
    # this can help prevent spurious template matches in background / padded areas of region.
    # xxx - since the feature to not consider grid points outside the roi polygon was added,
    #   and outlier detection for spurious deltas was improved, this is no longer necessary.
    region_bg_noise_fill = False

    # if translate_roi_center is True, whether to center on roi polygonal center or bbox center.
    # centering is more appealing on center of bounding box than on polygonal center.
    translate_roi_center_bbox = True

    # The data type of the coordinates (must be prevented from type-casting to default double) has major
    #   memory implications. Because the coordinate transforms work on pixel values, the precision is not
    #   that important. float16 however, would not allow the range required for large slices.
    coordindate_transformations_dtype = np.float32


    def __init__(self, experiment_folders, protocol_folders, alignment_folders, region_strs, wafer_ids=[0],
                 region_inds=None, dsstep=1, crop_um=[50,50], delta_rotation_range=[0.,0.], delta_rotation_step=1.,
                 template_crop_um=[10,10], grid_locations=[0,0], rough_bounding_box=None, region_suffix='.tiff',
                 exclude_regions=[], solved_order=None, load_stitched_coords=None, translate_roi_center=False,
                 use_thumbnails_ds=0, thumbnail_folders=[], roi_polygon_scale=0., wafer_format_str='wafer{:02d}_',
                 backload_roi_polys=None, nblocks=[1,1], iblock=[0,0], block_overlap_um=[0,0], scale_nm=None,
                 region_ext='.h5', region_interp_type_deltas='cubic', deformation_points_boundary_um=0.,
                 load_rough_xformed_img=False, load_img_subfolder='', griddist_um=0., init_regions=True,
                 legacy_zen_format=False, nimages_per_mfov=None, init_region_coords=True, region_manifest_cnts=None,
                 use_tissue_masks=False, tissue_mask_path=None, tissue_mask_ds=1, tissue_mask_fn_str=None,
                 tissue_mask_min_edge_um=0., tissue_mask_min_hole_edge_um=0., tissue_mask_bwdist_um=0., zorder=None,
                 use_coordinate_based_xforms=False, block_overlap_grid_um=[0,0], region_include_cnts=None,
                 verbose=False):
        self.wafer_verbose = verbose
        self.experiment_folders = experiment_folders

        # region counts for all wafers, needed for bookkeeping for some import/exports (e.g. tissue detection)
        if region_manifest_cnts is not None:
            self.wafer_manifest_cnts = np.array(region_manifest_cnts[1:])
            self.cum_wafer_manifest_cnts = np.cumsum(self.wafer_manifest_cnts)
        else:
            self.wafer_manifest_cnts = self.cum_wafer_manifest_cnts = None

        # included region counts for all wafers, these are the manifest counts minus the excluded region counts
        if region_include_cnts is not None:
            self.wafer_include_cnts = np.array(region_include_cnts[1:])
            self.cum_wafer_include_cnts = np.cumsum(self.wafer_include_cnts)
        else:
            self.wafer_include_cnts = self.cum_wafer_include_cnts = None

        # originally this was meant more as a temporary location for the montaged slices.
        # it is now fundamental to the mechanism of the pipeline, i.e., the location where all temporary
        #   saved processing values, image exports and alignment-meta information are saved.
        self.alignment_folders = alignment_folders

        # wafer id is not specifically from the msem software, but each wafer is run as a single "experiment" in a
        #   separate folder. currently some set of on the order of 10 wafers can comprise a single tissue block.
        # setting wafer_id to a list allows regions to be loaded from different wafers so that alignment can
        #   be done between sections in different wafers.
        self.nwafer_ids = len(wafer_ids)
        assert( self.nwafer_ids < 3 ) # only supporting two wafers for "cross-wafer" alignment
        self.is_cross_wafer = (self.nwafer_ids == 2)
        self.wafer_ids = wafer_ids
        self.wafer_strs = [wafer_format_str.format(x,) for x in self.wafer_ids]

        # suffix to use for loading the region input files
        self.region_suffix = region_suffix

        # extension to use for loading the region input files
        self.region_ext = region_ext

        # xxx - this needs to match the downsampling level of the region file inputs...
        #   read this automatically somehow?
        self.dsstep = dsstep

        # degree steps for delta template match
        self.delta_rotation_step = delta_rotation_step/180*np.pi

        # degree range about the starting rotation for delta template match
        self.delta_rotation_range = np.array(delta_rotation_range)/180*np.pi

        # size in microns of the crops to use for template matching
        self.crop_um = crop_um
        self.template_crop_um = template_crop_um

        # grid center locations specified in microns and relative to the center point of each region.
        self.grid_locations = np.array(grid_locations, dtype=np.double)
        if self.grid_locations.ndim == 1: self.grid_locations = self.grid_locations[None,:]

        # "regular" distance between grid points, usually the spacing for hexagonal grids
        self.griddist_um = griddist_um

        # specify rough bounding box for rough alignment.
        # NOTE: for the rough alignemnt step (not the order solving step), it must be centered on origin.
        if rough_bounding_box is None:
            # just so we can instantiate without rough_bounding_box specified.
            # rough_bounding_box is absolutely necessary for alignment however.
            self.rough_bounding_box = [np.zeros((2,), dtype=np.double)]*2
        else:
            self.rough_bounding_box = rough_bounding_box
        # grid points must be inside rough bounding box
        #print(self.grid_locations.min(0), self.grid_locations.max(0))
        #print(self.rough_bounding_box[0], self.rough_bounding_box[1])
        assert( (self.grid_locations.min(0) >= self.rough_bounding_box[0]).all() )
        assert( (self.grid_locations.max(0) <= self.rough_bounding_box[1]).all() )

        # create the region list before super init
        self.indexed_nregions = [sum([len(x) for x in y]) for y in region_strs]
        # region_ind is interpreted as a 1-based index into manifests, whether specified or not.
        if region_inds is None:
            assert(not self.is_cross_wafer) # only supporting specified regions at wafer borders
            region_inds = np.arange(1,self.indexed_nregions[0]+1)
        else:
            region_inds = np.array(region_inds, dtype=np.int32)
            assert( np.unique(region_inds).size == region_inds.size ) # duplicate regions specified
        self.region_inds = region_inds
        self.nregions = len(self.region_inds)
        assert(self.nregions > 0) # bad region list or no regions in experiment folder
        # this is in case the region_stage_coords.csv were not saved during acquisition. (!)
        if backload_roi_polys is None:
            backload_roi_polys = [None]*self.nregions
        else:
            assert( len(backload_roi_polys)==self.nregions ) # has to match number of regions
        valid_experiment_folders = any([x for x in experiment_folders])
        self.load_img_subfolder = load_img_subfolder
        if valid_experiment_folders:
            if legacy_zen_format:
                self.input_data_type = msem_input_data_types(msem_input_data_types.zen_msem_data)
            else:
                self.input_data_type = msem_input_data_types(msem_input_data_types.new_msem_data)
            region.__init__(self, experiment_folders[0], protocol_folders[0], region_strs[0], region_inds[0],
                    dsstep=dsstep, use_thumbnails_ds=use_thumbnails_ds, thumbnail_folders=thumbnail_folders[0],
                    backload_roi_poly_raw=backload_roi_polys[0], legacy_zen_format=legacy_zen_format,
                    nimages_per_mfov=nimages_per_mfov, scale_nm=scale_nm, tissue_mask_ds=tissue_mask_ds,
                    tissue_mask_min_edge_um=tissue_mask_min_edge_um,
                    tissue_mask_min_hole_edge_um=tissue_mask_min_hole_edge_um,
                    tissue_mask_bwdist_um=tissue_mask_bwdist_um, verbose=False)

            self.use_tissue_masks = use_tissue_masks
            # xxx - OLDtissue - remove when fully moved to saving the masks
            self.tissue_mask_path = tissue_mask_path
            self.tissue_mask_fn_str = tissue_mask_fn_str
        else:
            bn, ext = os.path.splitext(self.region_ext)
            if ext == '.h5' or (not ext and bn == '.h5'):
                self.input_data_type = msem_input_data_types(msem_input_data_types.hdf5_stack)
            else:
                self.input_data_type = msem_input_data_types(msem_input_data_types.image_stack)
            self.wafer_strs = ['' for x in range(self.nwafer_ids)]
            assert(scale_nm is not None)
            self.native_scale_nm = scale_nm
            self.scale_nm = self.native_scale_nm * (use_thumbnails_ds if use_thumbnails_ds > 0 else 1)
            # try to set reasonable defaults for the xcorr preprocessing, scale_nm must be set first
            self.set_xcorr_preproc_params()
            self.use_tissue_masks = False
            self.export_xcorr_comps_path = None
            # xxx - OLDtissue - remove when fully moved to saving the masks
            self.tissue_mask_path = None
            self.tissue_mask_ds = tissue_mask_ds # xxx - this stays even if we remove OLDtissue???


        # missing regions allows some regions to be completely ignored, but still maintain the same numbering for
        #   the rest of the regions as the experimental numbering. this was way less painful that maintaining another
        #   mapping (for example for re-ordering based on the czifile mapping or the region ordering mapping).
        # this means missing regions are not actually removed from the region_ind list.
        if exclude_regions is None: exclude_regions = []
        self.missing_regions = np.array(exclude_regions, dtype=np.int32)

        # get the region centers for all the region_inds to be analyzed for this wafer
        self.wafer_coords, self.wafer_filenames = [None]*self.nregions, [None]*self.nregions
        self.wafer_tris, self.wafer_region_strs = [None]*self.nregions, [None]*self.nregions
        self.wafer_zcoords, self.wafer_ztris = [None]*self.nregions, [None]*self.nregions
        self.region_corners = np.zeros((self.nregions,2), dtype=np.double)
        self.region_sizes = np.zeros((self.nregions,2), dtype=np.double)
        self.region_pcenters = np.zeros((self.nregions,2), dtype=np.double)
        #self.region_angles = np.zeros((self.nregions,), dtype=np.double)
        self.region_hulls = [None]*self.nregions
        self.region_s2ics = [None]*self.nregions; self.region_roi_poly = [None]*self.nregions
        self.region_roi_poly_raw = [None]*self.nregions
        if (self.input_data_type == msem_input_data_types.zen_msem_data or \
                self.input_data_type == msem_input_data_types.new_msem_data) and init_regions:
            self._initialize_wafer_regions(experiment_folders, protocol_folders, region_strs,
                    dsstep=dsstep, use_thumbnails_ds=use_thumbnails_ds, thumbnail_folders=thumbnail_folders,
                    backload_roi_polys=backload_roi_polys, init_region_coords=init_region_coords,
                    load_stitched_coords=load_stitched_coords)
        elif self.input_data_type == msem_input_data_types.image_stack or \
                self.input_data_type == msem_input_data_types.hdf5_stack:
            for region_ind, i in zip(self.region_inds, range(self.nregions)):
                if region_ind in self.missing_regions: continue
                wafer_ind = 0 if not self.is_cross_wafer else i
                # xxx - did not see any use case in having image stacks split between multiple folders
                self.wafer_region_strs[i] = region_strs[wafer_ind][0][region_ind-1]

        self.missing_regions = np.unique(self.missing_regions)
        self.sel_missing_regions = np.in1d(self.region_inds, self.missing_regions)
        self.sel_valid_regions = np.logical_not(self.sel_missing_regions)
        self.nmissing_regions = self.sel_missing_regions.sum()
        self.nvalid_regions = self.nregions - self.nmissing_regions
        assert(self.missing_regions.size == self.nmissing_regions) # bad missing regions specified
        if self.wafer_verbose:
            print('\ttotal %d region(s) missing or specified to be excluded' % (self.nmissing_regions, ))
            #print(self.missing_regions)

        # solved_order has to be set to something.
        # use wafer_solver.py for generating a solved slice (region) ordering.
        if solved_order is not None:
            solved_order = np.array(solved_order)
            assert(solved_order.size == self.nvalid_regions) # region order size needs to match number of regions
            assert(solved_order.size == np.unique(solved_order).size) # region order is not a permutation of nregions
            #assert(solved_order.max() == self.nregions-1 and solved_order.min() == 0) # out of bounds regions in order
            self.solved_order = solved_order
        else:
            self.solved_order = np.arange(self.nregions) # default to the Zeiss imaging order (region number order)

        # added after tissue mask support because the mask generation is not really aware of anything
        #   excpet for the final z-orderings for the entire dataset (experiment).
        self.zorder = zorder

        # initialize default rotations and rotation centers to initially apply to regions
        self.set_region_rotations_manual()
        self.set_region_translations_manual()
        # initialize default values for properties set using limi czi overviews
        #self.region_to_limi_roi = -np.ones((self.nregions,), dtype=np.int64)
        self.region_to_limi_roi = np.arange(self.nregions, dtype=np.int64)
        self.region_recon_roi_poly_raw = [None]*self.nregions
        self.limi_to_region_affine = None

        # initialize optional full affine transformation for rough alignment
        self.region_affines = [None]*self.nregions

        # initialize the deformation grids to None so by default no deformations are applied when loading.
        self.deformation_points = None
        self.imaged_order_deformation_vectors = [None]*self.nregions

        # only use these values for computing neighboring regions.
        # they are based on the zeiss coordinates, so will not be exactly correct for msem stitched regions.
        self.region_centers = self.region_corners + self.region_sizes/2
        assert( self.solved_order is not None or self.nregions > 3 ) # must specify order for less than 3 regions

        self.scale_um_to_pix = 1e3/(self.scale_nm*self.dsstep)

        # calculate crop sizes for template matches in pixels
        self.crop_size_pixels = np.array(crop_um, dtype=np.double)*self.scale_um_to_pix
        self.tcrop_size_pixels = np.array(template_crop_um, dtype=np.double)*self.scale_um_to_pix
        self.crop_size_pixels = self.crop_size_pixels.astype(np.int32)
        self.tcrop_size_pixels = self.tcrop_size_pixels.astype(np.int32)

        # calculate the bounding box used to crop each region during load.
        self.rough_bounding_box_pixels = [x*self.scale_um_to_pix for x in self.rough_bounding_box]

        # calculate locations for region crops to be used with registration
        self.ngrid = self.grid_locations.shape[0]
        assert(self.grid_locations.shape[1] == 2)
        self.grid_locations_pixels = np.array(self.grid_locations, dtype=np.double)*self.scale_um_to_pix
        self.griddist = self.griddist_um*self.scale_um_to_pix
        if self.wafer_verbose:
            tmp1 = self.rough_bounding_box[1]-self.rough_bounding_box[0]
            tmp2 = self.rough_bounding_box_pixels[1]-self.rough_bounding_box_pixels[0]
            print('Rough box is %.1f x %.1f (um), %d x %d (pix)' % \
                  (tmp1[0], tmp1[1], tmp2[0], tmp2[1]))
            tmp1 = self.grid_locations.max(0) - self.grid_locations.min(0)
            tmp2 = self.grid_locations_pixels.max(0) - self.grid_locations_pixels.min(0)
            print('Grid is %.1f x %.1f (um), %d x %d (pix), %d points' % \
                  (tmp1[0], tmp1[1], tmp2[0], tmp2[1], self.ngrid))

        # the grid locations are sent in centerd on zero.
        # adjust so they are centered on the rough bounding box.
        ctr = (self.rough_bounding_box_pixels[1]-self.rough_bounding_box_pixels[0])/2
        self.grid_locations_pixels += ctr[None,:]

        # linear fits / applying affines (for constant term)
        self.poly_degree1 = preprocessing.PolynomialFeatures(degree=1)

        # optionally translates the loaded region images so that they are centered on their zeiss roi polygon.
        # normally recommend False for this because it typically does not help much and can greatly increase
        #   the loaded image size. but, can be useful in instances where some portions of the neighboring slice are
        #   imaged in the current slice also. this can really mess up the rough alignment since it might match to the
        #   partially imaged slice (meaning matching to itself) instead of the actual next slice.
        self.translate_roi_center = translate_roi_center

        # this is a method that prevents having to customize grid points for an roi shape, and more particularly
        #   allows different shaped rois for the same experiment.
        # grid points outside the scaled roi polygon are not used to calculate deltas.
        # this value is the amount to scale the inclusion polygon. set to zero to disable (use all grid points).
        self.roi_polygon_scale = float(roi_polygon_scale)

        # method of interpolating the deltas before the remap for deformations (griddata).
        # NOTE: using nearest here is a bad idea, as then the deltas will simply jump
        #   at the grid triangulation lines.
        self.region_interp_type_deltas = region_interp_type_deltas

        # options for blockwise processing if memory limited
        self.nblocks = nblocks
        self.iblock = iblock
        self.block_overlap_pix = np.round(np.array(block_overlap_um,
            dtype=np.double)*self.scale_um_to_pix).astype(np.int64)
        self.block_overlap_grid_pix = np.round(np.array(block_overlap_grid_um,
            dtype=np.double)*self.scale_um_to_pix).astype(np.int64)
        self.single_block = all([x==1 for x in self.nblocks])
        self.first_block = all([x==0 for x in self.iblock])

        # this is a special mode typically used with native (ultrafine processing or fine export)
        #   that skips the initial microscope (angle) and rough affine transformations.
        self.load_rough_xformed_img = load_rough_xformed_img

        # for when tissues masks are stored along with the slices
        self.tissue_mask_bw = [None]*self.nregions
        self.tissue_mask_bw_rel_ds = -1

        # this is to use transforms that fully support block loading, including the region xforms.
        # this is accomplished by transforming the coordinates starting with the output block,
        #   and then doing a single image remap using the inverse transformed coordinates (dest -> src mapping).
        self.use_coordinate_based_xforms = use_coordinate_based_xforms

        # for inverting saved transformed control points
        self.invert_control_points = False

    def _initialize_wafer_regions(self, experiment_folders, protocol_folders, region_strs, load_stitched_coords=None,
            dsstep=1, use_thumbnails_ds=0, thumbnail_folders=[], backload_roi_polys=None, init_region_coords=True):
        config = configparser.ConfigParser()
        if self.wafer_verbose:
            print('Instantiating mfovs and loading coords for %d region(s)' % (self.nregions, ))
            t = time.time()
        for region_ind, i in zip(self.region_inds, range(self.nregions)):
            if region_ind in self.missing_regions: continue
            #print(region_ind)

            wafer_ind = 0 if not self.is_cross_wafer else i
            try:
                # mfov_align_init=(i==0)
                c = region(experiment_folders[wafer_ind], protocol_folders[wafer_ind], region_strs[wafer_ind],
                        region_ind, dsstep=dsstep, mfov_align_init=False, use_thumbnails_ds=use_thumbnails_ds,
                        thumbnail_folders=thumbnail_folders[wafer_ind], backload_roi_poly_raw=backload_roi_polys[i],
                        legacy_zen_format=self.legacy_zen_format, scale_nm=self.native_scale_nm,
                        nimages_per_mfov=self.nimages_per_mfov, init_region_coords=init_region_coords, verbose=False)
            except:
                print('WARNING: error add region %d to missing_regions, might be bad, might be Zeiss $%%#&' % \
                      (region_ind,))
                traceback.print_exc()
                self.missing_regions = np.append(self.missing_regions, [region_ind])
                assert( self.legacy_zen_format ) # region exception should not happen with new acquisition format
                continue

            # had to deal with possibility of empty regions on the experimental side.
            if c.imfov_diameter == 0:
                self.missing_regions = np.append(self.missing_regions, [region_ind])
                assert( self.legacy_zen_format ) # bad mfov should not happen with new acquisition format
                continue

            # save the original zeiss coordinates.
            self.wafer_zcoords[i], self.wafer_ztris[i] = c.region_coords, c.mfov_tri

            if init_region_coords:
                # typically use True to load stitched region coords.
                # set to False to use zeiss coords (for example if something is wrong with stitched region coords).
                if load_stitched_coords is not None and load_stitched_coords[wafer_ind]:
                    # reload coords to use the stitched coordinates and not the zeiss coordinates.
                    #fn = os.path.join(self.alignment_folders[wafer_ind],
                    fn = os.path.join(load_stitched_coords[wafer_ind],
                                      self.wafer_strs[wafer_ind] + c.region_str + self.coords_file_suffix)
                    c.load_stitched_region_image_coords(fn)

                # get bounding box on all the mfovs in each region
                self.region_corners[i,:] = c.region_coords.reshape(-1,2).min(0)
                self.region_sizes[i,:] = c.region_coords.reshape(-1,2).max(0) - self.region_corners[i,:]

            # save some of the region info as lists for all regions to be aligned in the wafer.
            self.wafer_coords[i], self.wafer_filenames[i], self.wafer_tris[i], self.wafer_region_strs[i] = \
                c.region_coords, c.region_filenames, c.mfov_tri, c.region_str

            # save the stage to ics zeiss transform and the zeiss polygon for each region.
            self.region_roi_poly_raw[i] = c.roi_poly_raw
            self.region_s2ics[i], self.region_roi_poly[i] = c.s2ics, c.roi_poly

            # adjust the roi polygon for any cropping that may have occurred during region montage (if present)
            bfn = self.wafer_strs[wafer_ind] + c.region_str + '_crop_info.ini'
            fn = os.path.join(self.alignment_folders[wafer_ind], bfn)
            if os.path.exists(fn):
                config.read(fn)
                crop_min = np.fromstring(config['crop_info']['crop_min'][1:-1], dtype=np.int, sep=' ')
                self.region_roi_poly[i] = self.region_roi_poly[i] - crop_min
        if self.wafer_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def _initialize_wafer_regions

    # this is basically a utility function that prints all the summary info about the wafer.
    def print_wafer_stats(self, full_print=False):
        assert(len(self.wafer_ids) == 1) # do not specify multiple wafers for stats
        if full_print: print('Wafer %d regions:' % (self.wafer_ids[0],))
        nmfovs = 0
        for region_ind, i in zip(self.region_inds, range(self.nregions)):
            if self.sel_missing_regions[i]: continue
            cnmfovs = self.wafer_coords[i].shape[0]
            if full_print: print('\tRegion %03d contains %d mfovs' % (region_ind, cnmfovs))
            nmfovs += cnmfovs
        print('For wafer %d, missing or excluded %d region(s):' % (self.wafer_ids[0], self.nmissing_regions,))
        print(np.nonzero(self.sel_missing_regions)[0] + 1)
        print('Summary stats for wafer %d' % (self.wafer_ids[0],))
        print('\tcontains %d valid regions' % (self.nvalid_regions,))
        print('\t%d total mfovs, %.3f mean mfovs per region' % (nmfovs, nmfovs/self.nvalid_regions))
        print('\t%d images per mfov, %d total images' % (self.niTiles, self.niTiles*nmfovs))
        x,y = self.images[0].shape; n = self.niTiles*nmfovs*x*y
        print('\tindividial image size %dx%d (%d) pixels, total %d pixels in wafer' % (x,y,x*y,n))


    # xxx - probably delete, some old throwback I think that I forgot to clean up
    # @staticmethod
    # def _get_all_wafer_regions(folder, coords_file_suffix):
    #     region_coords_files = glob.glob(os.path.join(folder, '*' + coords_file_suffix))
    #     region_inds = [os.path.basename(x).split('_') for x in region_coords_files]
    #     region_inds = [x[1] for x in region_inds if len(x) > 1]
    #     region_inds = [int(x) for x in region_inds if isInt_str(x)]
    #     return region_inds

    def set_region_rotations_czi(self, czifile, scene=1, ribbon=1, rotation=0., czfile=None, use_polygons=False,
                                 use_roi_polygons=False, use_full_affine=True, remove_duplicates=False, doplots=False):
        assert(self.nregions > 5) # point matching is not going to work for only a few regions
        if use_roi_polygons:
            print('WARNING: using roi polys for limi to region mapping, mostly intended as validation')

        # read the roi centers in the order they are defined in the czifile
        oscene = CziMsem(czifile, scene=scene, ribbon=ribbon, verbose=self.wafer_verbose)
        if czfile:
            oscene.load_cz_file_to_polys_or_rois(czfile, load_rois=True)
        else:
            oscene.read_scene_meta()
        assert( oscene.nROIs > 0 or oscene.npolygons > 0 ) # czifile contains no polygons or ROIs
        # if there are no ROIs, then use the polygons (or optionally force polygons with parameter)
        use_polygons = use_polygons or (oscene.nROIs == 0)
        nczi_regions = oscene.npolygons if use_polygons else oscene.nROIs
        limi_roi_pts = [None]*nczi_regions
        limi_roi_ctrs = np.zeros((nczi_regions,2), dtype=np.double)
        for i in range(nczi_regions):
            pts = (oscene.polygons_points[i] if use_polygons else oscene.rois_points[i])
            # thanks Zeiss, remove duplicates
            # https://stackoverflow.com/questions/37839928/remove-following-duplicates-in-a-numpy-array
            pts = pts[np.insert((np.diff(pts,axis=0)!=0).all(1),0,True),:]
            # remove final point if repeat of first point (code here assumes polygon is closed).
            if (pts[0,:] == pts[-1,:]).all(): pts = pts[:-1,:]
            pts = pts*oscene.scale*1e3 # convert to nm
            limi_roi_pts[i] = pts
            limi_roi_ctrs[i,:] = PolyCentroid(pts[:,0], pts[:,1])

        # get the region centers from msem data either using the stored roi polygon,
        #   or by using the center of the region based on the msem coordinates.
        msem_roi_ctrs = np.empty((self.nregions,2), dtype=np.double); msem_roi_ctrs.fill(np.nan)
        for region_ind, i in zip(self.region_inds, range(self.nregions)):
            # had to deal with possibility of empty regions on the experimental side.
            if self.sel_missing_regions[i]: continue
            if use_roi_polygons:
                assert(self.region_roi_poly_raw[i] is not None) # use_roi_polygons==True and coords missing
                # these coordinates should match without any transformation,
                #   as they are also in the stage coordinates (?) space.
                pts = self.region_roi_poly_raw[i]
                msem_roi_ctrs[i,:] = PolyCentroid(pts[:,0], pts[:,1])
            else:
                points = self.wafer_zcoords[i].reshape(-1,2)
                points = points * self.scale_nm
                hull = scipy.spatial.ConvexHull(points); pts = points[hull.vertices,:]
                msem_roi_ctrs[i,:] = PolyCentroid(pts[:,0], pts[:,1])
                # these coords are in msem pixel (ics?) space.
                # use inverse of stage2ics to get to stage coords (?) space.
                msem_roi_ctrs[i,:] = np.dot(lin.inv(self.region_s2ics[i]), msem_roi_ctrs[i,:][None,:].T).T
        # per info from zeiss, the coordinates need to be inverted
        #   when converting between stage and ics coordinates.
        # this does not work here; empirically determined which coords needed to be inverted in each case.
        if use_roi_polygons:
            msem_roi_ctrs[:,0] = -msem_roi_ctrs[:,0] # determined empirically
        else:
            msem_roi_ctrs[:,1] = -msem_roi_ctrs[:,1] # determined empirically
            # put the missing points in the top-left corner (zero if min were to be subtracted)
            msem_roi_ctrs[np.isnan(msem_roi_ctrs).all(1),:] = np.nanmin(msem_roi_ctrs, 0)

        if doplots:
            plt.figure()
            plt.subplot(1,2,1)
            plt.scatter(msem_roi_ctrs[:,0], msem_roi_ctrs[:,1], s=12, edgecolors='b',facecolors='none')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title('msem')
            plt.subplot(1,2,2)
            plt.scatter(limi_roi_ctrs[:,0], limi_roi_ctrs[:,1], s=12, edgecolors='b',facecolors='none')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title('limi')
            plt.show()

        # NOTE: with the addition of missing and exclude regions (#$%#^), advise strongly not to use
        #   the limi_roi_to_region lookup because missing regions will show up in this mapping as duplicates
        #   to the next closest region. better to use the region driven mapping, region_to_limi_roi.
        self.region_to_limi_roi, limi_roi_to_region = points_match_trans(msem_roi_ctrs, limi_roi_ctrs,
                rmvduplB=remove_duplicates, doplots=doplots)
        s = self.sel_valid_regions # do not care about mapping for missing regions
        assert( self.region_to_limi_roi[s].size == np.unique(self.region_to_limi_roi[s]).size ) # bad mapping
        # base rotation specified in degrees, convert to radians
        angs = (oscene.polygons_rotation if use_polygons else oscene.rois_rotation)
        self.region_rotations = constrainAngle(angs[self.region_to_limi_roi] + rotation/180*np.pi)
        #print(self.region_inds, region_to_roi, self.region_rotations/np.pi*180)

        # this is in case the region_stage_coords.csv were not saved during acquisition, another fantastic
        #   [feature] that we have to deal with.
        # use the acquired mapping to calculate an affine xfrom from the limi coorindates to the msem coordinates.
        # then apply this transform to the limi rois to "regenerate" the rois stored in region_stage_coords.csv
        tgt_pts = msem_roi_ctrs[self.sel_valid_regions,:]
        src_pts = limi_roi_ctrs[self.region_to_limi_roi,:][self.sel_valid_regions,:]
        if use_full_affine:
            poly = preprocessing.PolynomialFeatures(degree=1)
            clf = linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=self.nthreads)
            src_pts = poly.fit_transform(src_pts)
        else:
            clf = RigidRegression()
        clf.fit(src_pts,tgt_pts)
        fit_pts = clf.predict(src_pts)
        caffine = clf.coef_
        if use_full_affine:
            # scikit learn puts constant terms on the left, flip and augment
            caffine = np.concatenate( (np.concatenate( (caffine[:,1:], caffine[:,0][:,None]), axis=1 ),
                                       np.zeros((1,3), dtype=caffine.dtype)), axis=0 )
            caffine[2,2] = 1
        # now compute the transformed ("reconstructed") roi polygons using the xform.
        self.limi_to_region_affine = caffine
        self.region_recon_roi_poly_raw = [None]*self.nregions
        for region_ind, i in zip(self.region_inds, range(self.nregions)):
            if self.sel_missing_regions[i]: continue
            rsrc_pts = limi_roi_pts[self.region_to_limi_roi[i]]
            if use_full_affine: rsrc_pts = poly.fit_transform(rsrc_pts)
            rfit_pts = clf.predict(rsrc_pts)
            rfit_pts[:,0] = -rfit_pts[:,0] # determined empirically
            self.region_recon_roi_poly_raw[i] = rfit_pts

        if doplots:
            plt.figure()
            plt.scatter(tgt_pts[:,0], tgt_pts[:,1], s=12, edgecolors='b',facecolors='none')
            plt.scatter(fit_pts[:,0], fit_pts[:,1], s=12, edgecolors='r',facecolors='none')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', 'datalim')
            plt.axis('off')
            plt.show()


    def set_region_rotations_roi(self, roi_points, vaf_cutoff=0.99, diff_warn_deg=np.inf,
            index_roi_points=False, doplots=False):
        vaf = np.zeros((self.nregions,), dtype=np.double)
        rot = np.zeros((self.nregions,), dtype=np.double)
        for region_ind, i in zip(self.region_inds, range(self.nregions)):
            # had to deal with possibility of empty regions on the experimental side.
            if self.sel_missing_regions[i]: continue
            cpts = roi_points[i] if index_roi_points else roi_points
            # xxx - better to use fit/predict exposed sklearn methods here
            R, trans, rot[i] = RigidRegression.rigid_transform(self.region_roi_poly[i], cpts)
            xpts = np.dot(self.region_roi_poly[i], R[:-1,:-1].T) + trans
            mse = ((xpts - cpts)**2).mean()
            var = np.var(cpts - cpts.mean(0))
            vaf[i] = 1 - mse/var
            if doplots:
                plt.figure()
                plt.scatter(self.region_roi_poly[i][:,0], self.region_roi_poly[i][:,1])
                plt.scatter(cpts[:,0], cpts[:,1], color='r')
                plt.scatter(xpts[:,0], xpts[:,1], color='g')
                plt.title('%g' % (vaf[i],))
                plt.show()

        # optionally print out any rotations that are different from existing.
        # typically this would mean different from angles loaded from czifile.
        if np.isfinite(diff_warn_deg):
            diff = np.abs(rot + self.region_rotations); diff[self.sel_missing_regions] = 0
            sel = (diff > diff_warn_deg/180*np.pi)
            if sel.sum() > 0:
                print('WARNING: set_regions_rotations_roi angles are different (from czi) for regions:')
                print(np.nonzero(sel)[0]); print(diff[sel]/np.pi*180)
        sel = np.logical_or(vaf >= vaf_cutoff, self.sel_missing_regions)
        self.region_rotations[sel] = -rot[sel] # image xform is inversion of point transform.
        if self.wafer_verbose and not sel.all():
            print('WARNING: set_regions_rotations_roi vaf cutoff %g does not fit for regions:' % (vaf_cutoff,))
            print(self.region_inds[np.logical_not(sel)])


    def set_region_rotations_manual(self, rotations=None, use_solved_order=False):
        inds = self.solved_order if use_solved_order else np.arange(self.nregions)
        if rotations is None:
            self.region_rotations = np.zeros((self.nregions,), dtype=np.double)
        elif len(rotations) == 1:
            self.region_rotations = np.empty((self.nregions,), dtype=np.double)
            self.region_rotations.fill(constrainAngle(np.array(rotations[0])/180*np.pi))
        else:
            assert(len(rotations) == self.nregions) # need to define rotation for each region
            # rotations are manually specified in degrees
            self.region_rotations[inds] = constrainAngle(np.array(rotations)/180*np.pi)


    def set_region_translations_manual(self, translations=[0.,0.], use_solved_order=False):
        inds = self.solved_order if use_solved_order else np.arange(self.nregions)
        self.region_translations = np.zeros((self.nregions,2), dtype=np.double)
        trans = np.array(translations).reshape(-1,2)
        if trans.shape[0] == 1 or (self.is_cross_wafer and trans.shape[0]==2):
            self.region_translations += trans
        else:
            assert(len(translations) == self.nregions) # need to define translation for each region
            self.region_translations[inds,:] = translations

    # this is a standard way of computing the rough bounding box limits (and correspondingly size),
    #   so that all rough aligned images come out the same size. it is functionized so that it can
    #   be used at different downsamplings (in particular for tissue masks).
    def _center_box_pixels(self, target_ctr, rel_ds):
        rbbox = [x/rel_ds for x in self.rough_bounding_box_pixels]

        # this is so that the rough bbox size matches for all images and in the load_rough_xformed_img pathway.
        #   if the size is calculated on the translated box (below it is translated by target_ctr),
        #     then the rounding causes the image sizes to vary slightly between slices.
        box = [x - rbbox[0] for x in rbbox]
        ibox_size = np.round(box[1]-box[0]).astype(np.int64)

        # crop using rough alignment cropping box, this way we can apply rough affine properly below.
        # bounding box MUST be centered so that the image center does not change. see comment above.
        if target_ctr is not None:
            box = [x + target_ctr/rel_ds for x in rbbox]
        # convert to integer crops so that all cropped output images are exactly the same size.
        # this works because the box remains fixed for all regions.
        igrid_min = np.round(box[0]).astype(np.int64)
        igrid_max = igrid_min + ibox_size

        return igrid_min, igrid_max, ibox_size

    def _region_load_xforms(self, ind, slice_image_fn, bg_fill_type, do_bg_fill, region_size, img, bgsel, tm_bw,
            load_xform_bg, load_xform_tm, roi_points, control_points, rel_ds, verbose, doplots):
        xdtype = self.coordindate_transformations_dtype

        if not self.load_rough_xformed_img:
            # this means we actually apply the microscope and rough xforms,
            #   instead of loading an already xform'ed image.
            # this code path has to load the entire slice to apply the rough xforms,
            #   and if block processing is enabled, then it is done before before the fine warping.
            if verbose: print('Microscope angle rotation'); t = time.time()
            # the "microscope" angle rotation (angle of the ROI as reported by limi slice detection software).
            # do not apply rotation at all if angle is very very small or zero.
            ang_rad = self.region_rotations[ind]; ang_deg = ang_rad/np.pi*180
            do_rotation = (np.abs(ang_deg) > 1e-5)
            if do_rotation:
                # apply rotation with expand on to include all of image in result
                img = img.rotate(ang_deg, expand=True, resample=wafer.region_interp_type)
                if load_xform_bg and bgsel is not None:
                    bgsel = bgsel.rotate(ang_deg, expand=True, resample=Image.NEAREST, fillcolor=1)
                if load_xform_tm and tm_bw is not None:
                    tm_bw = tm_bw.rotate(ang_deg, expand=True, resample=Image.NEAREST)
            if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        # get rotation center and do all xforms for roi points
        roi_points, target_ctr, rot_region_size = self._xform_points(ind, roi_points, region_size)
        if not self.load_rough_xformed_img:
            assert( (rot_region_size == np.array(img.size[:2])).all() ) # rotated size not matching PIL

        # apply transformations to the control points that are used for tear stitching
        if control_points is not None:
            if self.invert_control_points:
                control_points = self._inv_xform_points(ind, control_points, region_size, target_ctr)
            else:
                control_points, _, _ = self._xform_points(ind, control_points, region_size, target_ctr=target_ctr)

        if not self.load_rough_xformed_img:
            if verbose: print('Crop to rough bounding box'); t = time.time()
            igrid_min, igrid_max, _ = self._center_box_pixels(target_ctr, 1)
            if not all([x==y for x,y in zip(img.size,igrid_max-igrid_min)]):
                # xxx - is it always true for stack inputs that
                #   the image sizes should always match the rough bounding box size?
                assert(self.input_data_type == msem_input_data_types.zen_msem_data or \
                    self.input_data_type == msem_input_data_types.new_msem_data)
                img = img.crop((igrid_min[0], igrid_min[1], igrid_max[0], igrid_max[1]))
            if load_xform_bg and bgsel is not None and \
                    (not all([x==y for x,y in zip(bgsel.size,igrid_max-igrid_min)])):
                # https://stackoverflow.com/questions/50527851/background-color-when-cropping-image-with-pil/50725455
                #bgsel = bgsel.crop((igrid_min[0], igrid_min[1], igrid_max[0], igrid_max[1]), fillcolor=1)
                bgnew = Image.new(bgsel.mode, (igrid_max[0]-igrid_min[0], igrid_max[1]-igrid_min[1]), 1)
                bgnew.paste(bgsel, (-igrid_min[0], -igrid_min[1])); bgsel = bgnew
            # if roi_points is not None: roi_points = roi_points - igrid_min
            if load_xform_tm and tm_bw is not None:
                tm_igrid_min, tm_igrid_max, _ = self._center_box_pixels(target_ctr, rel_ds)
                if not all([x==y for x,y in zip(tm_bw.size,tm_igrid_max-tm_igrid_min)]):
                    tm_bwnew = Image.new(tm_bw.mode, (tm_igrid_max[0]-tm_igrid_min[0],
                            tm_igrid_max[1]-tm_igrid_min[1]))
                    tm_bwnew.paste(tm_bw, (-tm_igrid_min[0], -tm_igrid_min[1])); tm_bw = tm_bwnew
            if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

            # apply the rough-alignment affine transformation
            if self.region_affines is not None and self.region_affines[ind] is not None:
                if verbose: print('Apply rough affine xform'); t = time.time()
                #print('applying affine'); print(self.region_affines[ind][:2,:])
                a,b,c,d,e,f = self.region_affines[ind][:2,:].flat[:]
                img = img.transform(img.size, Image.AFFINE, (a,b,c,d,e,f), resample=wafer.region_interp_type)

                if load_xform_bg and bgsel is not None:
                    bgsel = bgsel.transform(bgsel.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.NEAREST,
                            fillcolor=1)
                if load_xform_tm and tm_bw is not None:
                    # adjust the translation for the relative downsampling level of the mask
                    caff = self.region_affines[ind].copy(); caff[:2,2] /= rel_ds; a,b,c,d,e,f = caff[:2,:].flat[:]
                    tm_bw = tm_bw.transform(tm_bw.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.NEAREST)
                if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

            if do_bg_fill:
                aimg = np.asarray(img).copy(); del img # avoid ValueError: assignment destination is read-only
                if bgsel: bgsel = np.asarray(bgsel)
                label_fill_image_background(aimg, bg_fill_type=bg_fill_type, bgsel=bgsel)
                img = Image.fromarray(aimg); del aimg

            # currently this is single block anyways, so update shape after the xforms
            img_shape = img.size[:2][::-1]
        else: #if not self.load_rough_xformed_img:
            img_shape = region_size[:2][::-1]
            # for this pathway, verify that the total image shape matches that of the rough bounding box
            box = [x - self.rough_bounding_box_pixels[0] for x in self.rough_bounding_box_pixels]
            ibox_size = np.round(box[1]-box[0]).astype(np.int64)
            assert(all([x==y for x,y in zip(img_shape[::-1],ibox_size)]))

        # region_load using image transforms only supports block processing for the fine transformation.
        # save min value for the block, used later in many places as a "crop" for the block.
        if not self.single_block:
            #rngs, max_shape, min_shape, rng =
            _, _, _, rng = tile_nblks_to_ranges(img_shape, self.nblocks, self.block_overlap_pix, self.iblock)
            icrop_min = np.array([x[0] for x in rng][::-1])
            if not self.load_rough_xformed_img:
                # xxx - because the microscope/rough xforms do not support block processing,
                #   if it was specified then crop out the block before applying the warping xforms.
                # the block processing is all done off of shape, so swap for image coordinates.
                img = img.crop((rng[1][0], rng[0][0], rng[1][1], rng[0][1]))
                if bgsel is not None:
                    bgsel = bgsel.crop((rng[1][0], rng[0][0], rng[1][1], rng[0][1]))
        else:
            icrop_min = np.zeros((2,),dtype=np.int64)

        # convert bgsel to numpy, if it was not done above
        if bgsel is not None and type(bgsel) is Image.Image:
            # avoid ValueError: assignment destination is read-only
            bgsel = np.asarray(bgsel).copy()
        # convert mask to numpy, if it was not done above
        if tm_bw is not None and type(tm_bw) is Image.Image:
            # avoid ValueError: assignment destination is read-only
            tm_bw = np.asarray(tm_bw).copy()

        # whether to apply the general warping deformation using defined delta vectors.
        # warping is applied using a generalized coordinate remap, e.g. ndimage map_coordinates.
        # apply the deformation only within the bounds of the grid points
        #   in order to avoid extrapolation of the deltas outside their defined area.
        # a pixel-dense method for interpolating that also extrapolates is possible, but
        #   for the methods attempted (MLS, TPS) this is way too compute intensive,
        #   particularly at high resolution (natve @4nm).
        if self.deformation_points is not None:
            if verbose: print('Apply warping deformation'); t = time.time()
            bgsel = None # warping xform is not applied to the background, so do not return it
            assert( not self.use_tissue_masks ) # xxx - did not implement the warping for the tissues masks, ouch
            tm_bw = None # remove the tissues masks, if present, since we did not implement warping them
            # completely convert to numpy here, then convert back before returning.
            sz = np.array(img.size[:2])
            img = np.asarray(img).copy() # avoid ValueError: assignment destination is read-only

            # the points and the vectors that define the warping, use double for interpolating the vector field.
            # with the addition of the rough bounding box, the deformation points are the same for all regions.
            # NOTE: p is modified, so in reality we need this copy.
            #   However, this copy is memory-expensive for dense grids,
            #     and in practive the deformation points are only used once per-process.
            #p = self.deformation_points.astype(np.double, copy=True)
            p = self.deformation_points.astype(np.double)
            v = self.imaged_order_deformation_vectors[ind].astype(np.double)
            assert( np.isfinite(p).all() ) # non-finite points should be handled by wafer_aggregator

            pmin, pmax, cmin, cmax, vx, vy = self._region_load_setup_deformations(p, v,
                    icrop_min, icrop_min, sz, sz, True, verbose, doplots)

            # crop out the grid area plus the context that is needed
            #   for the max/min coordinates (grid plus deltas).
            # NOTE: this code is super confusing, but if you do not do this and only crop to pmin/pmax,
            #   which is the alignment points bounding box, then any image portions that are outside of this
            #   but still inside of the rough alignment bounding box will be zero (i.e. deltas that extend
            #     outside of the alignment points bounding box but remain inside the rough bounding box).
            #   Ultimately maybe this does not matter much, but the workflow is moving to using the coords
            #     xform anyways, which supports warping any of the original region pixels, even if they are
            #     outside of the rough bounding box. So, leave this here for completeness.
            cmin = np.floor(cmin).astype(np.int64); cmax = np.ceil(cmax).astype(np.int64)+1
            # bounding box defined by cmin/cmax should always encompass that defined by pmin/pmax
            cmin = np.minimum(cmin, pmin); cmax = np.maximum(cmax, pmax)
            crng = cmax - cmin
            map_y, map_x = np.indices((crng[1], crng[0]), dtype=xdtype)
            if (cmin < 0).any() or (cmax > sz).any():
                print('WARNING: _region_load_xforms realloc b/c vectors out of bounds of image')
                # handle area plus context crop out of bounds of original image case.
                cimg = np.zeros(crng[::-1], dtype=img.dtype)
                ccmin = np.array([0,0]); sel = (cmin < 0); ccmin[sel] = -cmin[sel]
                ccmax = crng.copy(); sel = (cmax > sz); ccmax[sel] = crng[sel]-(cmax[sel] - sz[sel])
                icmin = cmin.copy(); icmax = cmax.copy()
                icmin[cmin < 0] = 0; sel = (cmax > sz); icmax[sel] = sz[sel]
                cimg[ccmin[1]:ccmax[1], ccmin[0]:ccmax[0]] = img[icmin[1]:icmax[1], icmin[0]:icmax[0]]
            else:
                cimg = img[cmin[1]:cmax[1], cmin[0]:cmax[0]]
            dmin = pmin-cmin; dmax = cmax-pmax
            tmp = map_x[dmin[1]:crng[1]-dmax[1], dmin[0]:crng[0]-dmax[0]]
            map_x[dmin[1]:crng[1]-dmax[1], dmin[0]:crng[0]-dmax[0]] = tmp + vx
            tmp = map_y[dmin[1]:crng[1]-dmax[1], dmin[0]:crng[0]-dmax[0]]
            map_y[dmin[1]:crng[1]-dmax[1], dmin[0]:crng[0]-dmax[0]] = tmp + vy
            del vx,vy,tmp

            # make sure none of the mapping is out-of-bounds.
            assert( (map_x >= 0).all() and (map_x < crng[0]).all() )
            assert( (map_y >= 0).all() and (map_y < crng[1]).all() )
            # warp the cropped image using interpolated deformations, and insert it back into the image.
            # crop is so the result matches the area that is warped (the map_x / map_y crops above).
            # map_coordinates does not flip x/y like cv2 remap does.
            img[pmin[1]:pmax[1], pmin[0]:pmax[0]] = nd.map_coordinates(cimg, [map_y, map_x],
                    order=self.warping_spline_order_nd, mode='constant',
                    prefilter=False)[dmin[1]:crng[1]-dmax[1], dmin[0]:crng[0]-dmax[0]]
            del cimg, map_x, map_y
            if verbose: print('\tdone in %.4f s' % (time.time() - t, ))
        #if self.deformation_points is not None:

        return img, roi_points, control_points, img_shape, icrop_min, bgsel, tm_bw
    #def _region_load_xforms(self

    # for transforming points based on the alignments being applied.
    # originally this was just for the points defining the roi, but made general for application to any point set.
    def _xform_points(self, ind, points, region_size, target_ctr=None):
        if not self.load_rough_xformed_img:
            ang_rad = self.region_rotations[ind]; ang_deg = ang_rad/np.pi*180
            do_rotation = (np.abs(ang_deg) > 1e-5)
            if do_rotation:
                # create the 2D rotation matrix. all rotations are done about the region center (region_size/2).
                c, s = np.cos(ang_rad), np.sin(ang_rad)
                #R_forwards = np.array([[c, -s], [s, c]]) # forwards for images
                R_backwards = np.array([[c, s], [-s, c]]) # backwards for points

                # previously used rotated image to get the size of the rotated image.
                # can not do this with coords xform, so modify both code paths so that the size
                #   of the rotated image matches what is expected from PIL. as one would expect,
                #   the size is determined by the bounding box on the rotated original corners.
                pts = np.array([[0.,0.], [0.,region_size[1]], region_size, [region_size[0],0.]])
                rot_pts = np.dot(pts - region_size/2, R_backwards.T) + region_size/2
                #rot_pts = np.dot(pts, R_backwards.T)
                rot_region_size = (np.ceil(rot_pts.max(0)) - np.floor(rot_pts.min(0))).astype(np.int64)
                # PIL must have special code for 90/180 rotations
                if float(ang_deg).is_integer():
                    iang_deg = int(ang_deg)
                    if iang_deg % 180 == 0:
                        rot_region_size = region_size
                    elif iang_deg % 90 == 0:
                        rot_region_size = region_size[::-1]

                if points is not None:
                    # rotate around image center (same as image rotation)
                    points = np.dot(points - region_size/2, R_backwards.T) + rot_region_size/2
            else:
                rot_region_size = region_size

            if target_ctr is None:
                # optionally translate the image so that it is centered on the zeiss roi polygon.
                if self.translate_roi_center:
                    if self.translate_roi_center_bbox:
                        # use center of roi polygon bounding box, typically makes more sense
                        target_ctr = (points.max(0) + points.min(0))/2
                    else:
                        # use barymetric center of roi polygon
                        target_ctr = np.array(PolyCentroid(points[:,0], points[:,1]))
                else:
                    target_ctr = rot_region_size/2

                # incorporate any previous manually defined translations into the target_ctr
                target_ctr += self.region_translations[ind,:]

            if points is not None:
                igrid_min, igrid_max, _ = self._center_box_pixels(target_ctr, 1)
                points = points - igrid_min

                if self.region_affines is not None and self.region_affines[ind] is not None:
                    # warp roi points with affine to match image, use inverse affine for the points
                    aff = lin.inv(self.region_affines[ind])
                    # scikit learn puts constant terms on the left, remove augment and flip
                    aff = np.concatenate( (aff[:2,2][:,None], aff[:2,:2], ), axis=1 )
                    pts = self.poly_degree1.fit_transform(points)
                    points = np.dot(pts, aff.T)
        else: # if not self.load_rough_xformed_img:
            target_ctr = None
            rot_region_size = region_size

        if self.deformation_points is not None:
            p = self.deformation_points.astype(np.double)
            v = self.imaged_order_deformation_vectors[ind].astype(np.double)

            # warp the roi points first by interpoating the vectors at the roi points.
            # the transform of the points is the forward mapping, this means just negative of
            #   the inverse mapping used below by map_coordinates.
            if points is not None:
                # use inverse distance weighting (IDW) to interpolate the vectors at the roi points.
                # xxx - need to parameterize any of these?
                idw_p = 2. # squaring consistently seems to be best choice
                scl = 1.5
                min_nbrs = 3
                nbrs = NearestNeighbors(radius=scl*self.griddist, algorithm='ball_tree').fit(p)
                dist, nnbrs = nbrs.radius_neighbors(points, return_distance=True)
                knbrs = NearestNeighbors(n_neighbors=min_nbrs, algorithm='kd_tree').fit(p)
                kdist, knnbrs = knbrs.kneighbors(points, return_distance=True)
                nnbrs_nsel = np.array([x.size < min_nbrs for x in nnbrs])
                nnbrs = [x if y else z for x,y,z in zip(knnbrs,nnbrs_nsel,nnbrs)]
                dist = [x if y else z for x,y,z in zip(kdist,nnbrs_nsel,dist)]
                for i in range(points.shape[0]):
                    # weight the average by the inverse distance to each point.
                    # this is inverse distance weighting interpolation.
                    W = 1/dist[i]**idw_p
                    points[i,:] = points[i,:] + (-v[nnbrs[i],:]*W[:,None]).sum(0)/W.sum()
        #if self.deformation_points is not None:

        return points, target_ctr, rot_region_size
    #def _xform_points(self

    # inverting transformed points back to the original region space coordinates.
    def _inv_xform_points(self, ind, points, region_size, target_ctr):

        # xxx - currently not supporting inverting the deformations

        # invert the affine transformation (inverse is forward transformation for points).
        if self.region_affines is not None and self.region_affines[ind] is not None:
            aff = self.region_affines[ind]
            # scikit learn puts constant terms on the left, remove augment and flip
            aff = np.concatenate( (aff[:2,2][:,None], aff[:2,:2], ), axis=1 )
            pts = self.poly_degree1.fit_transform(points)
            points = np.dot(pts, aff.T)

        # invert the rough bounding box crop
        igrid_min, igrid_max, _ = self._center_box_pixels(target_ctr, 1)
        points = points + igrid_min

        # invert the microscope rotation
        ang_rad = self.region_rotations[ind]; ang_deg = ang_rad/np.pi*180
        do_rotation = (np.abs(ang_deg) > 1e-5)
        if do_rotation:
            # create the 2D rotation matrix. all rotations are done about the region center (region_size/2).
            c, s = np.cos(ang_rad), np.sin(ang_rad)
            R_forwards = np.array([[c, -s], [s, c]]) # forwards for images
            #R_backwards = np.array([[c, s], [-s, c]]) # backwards for points

            # previously used rotated image to get the size of the rotated image.
            # can not do this with coords xform, so modify both code paths so that the size
            #   of the rotated image matches what is expected from PIL. as one would expect,
            #   the size is determined by the bounding box on the rotated original corners.
            pts = np.array([[0.,0.], [0.,region_size[1]], region_size, [region_size[0],0.]])
            rot_pts = np.dot(pts - region_size/2, R_forwards.T) + region_size/2
            rot_region_size = (np.ceil(rot_pts.max(0)) - np.floor(rot_pts.min(0))).astype(np.int64)
            # PIL must have special code for 90/180 rotations
            if float(ang_deg).is_integer():
                iang_deg = int(ang_deg)
                if iang_deg % 180 == 0:
                    rot_region_size = region_size
                elif iang_deg % 90 == 0:
                    rot_region_size = region_size[::-1]

            # rotate around image center (same as image rotation)
            points = np.dot(points - rot_region_size/2, R_forwards.T) + region_size/2
        # if do_rotation:

        return points
    #def _inv_xform_points(self

    # common coordinate manipulations for deformations in _region_load_xforms and _region_load_coords
    def _region_load_setup_deformations(self, p, v, icrop_min, icrop_min_grid, icoords_size, icoords_size_grid,
            calculate_range, verbose, doplots):
        xdtype = self.coordindate_transformations_dtype

        # crop the deformations points to the block being processed
        if not self.single_block:
            p = p - icrop_min_grid[None,:]
            # also remove any deformation points / vectors that are within specified distance of
            #   the edge of the block. prevents reallocation and bad interpolation near edges of block.
            # subtract one on the max end to deal with the ceil of pmax below (otherwise off-by-one is possible)
            sel = np.logical_and(p >= 0, p < icoords_size_grid[None,:] - 1).all(1)
            p = p[sel,:]; v = v[sel,:]

        # keep points to interpolate based on specified range for the grid.
        # only interpolate within the bounds of the selected points.
        # xxx - theoretically griddata can handle filling extrapolated points with default value,
        #   but in the past this did not work for unknown reasons in some cases (typically would hang).
        #   so the cropping here (pmin/pmax) is to avoid any extrapolation when calling griddata
        pmin = np.floor(p.min(0)).astype(np.int64); pmax = np.ceil(p.max(0)).astype(np.int64)+1
        prng = pmax - pmin
        assert( not ((pmax < 0).any() or (pmin > icoords_size_grid).any()) ) # grid points outside rough bounding box
        # xxx - currently griddata does not work in linear mode if the total size exceeds 30 bits
        assert( self.region_interp_type_deltas != 'linear' or prng.prod()/2**30 < 1 )

        assert( (icrop_min >= icrop_min_grid).all() ) # grid overlap must be larger
        assert( (icoords_size_grid >= icoords_size).all() ) # grid overlap must be larger
        # set pmin/pmax for the dense crop (gpmin/gpmax). in the legacy case where the overlap is equal
        #   for the grid crop and the dense crop, gpmin/gpmax (should) default to pmin/pmax.
        # default to the entire dense grid (with overlap)
        gpmin = np.zeros(2, dtype=np.int64); gpmax = icoords_size.copy()
        delta_min = icrop_min - icrop_min_grid
        delta_max = delta_min + icoords_size
        sel = (delta_min <= pmin)
        outside_grid = (sel.any() and (delta_max < pmin).any())
        if outside_grid:
            if verbose: print('Block completely outside grid, set interp deltas to 0')
            # the block is completely outside of the grid points
            vx = vy = np.zeros(icoords_size[::-1], dtype=xdtype)
            cmin = gpmin; cmax = gpmax
        else:
            # gpmin/gpmax are assigned relative to dense grid overlap.
            gpmin[sel] = pmin[sel] - delta_min[sel]
            sel = (delta_max >= pmax)
            gpmax[sel] = pmax[sel] - delta_min[sel]
            gprng = gpmax - gpmin
            # goffset is assigned relative to grid points overlap.
            # different overlaps allows for larger grid context without dense interpolating the whole overlap area.
            goffset = np.zeros(2, dtype=np.int64)
            sel = (pmin < delta_min)
            goffset[sel] = delta_min[sel] - pmin[sel]

            # actually perform the interpolation only on the specific range (not necessarily same as  grid range).
            # this allows for grid points outside of the dense interpolated area to be left as context.
            grid_y, grid_x = np.indices((gprng[1], gprng[0]), dtype=xdtype) + goffset[::-1,None,None]

            # griddata does not handle large offsets, so remove min for interpolation
            p = p - pmin
            vx = interp.griddata(p, v[:,0], (grid_x, grid_y), fill_value=0., method=self.region_interp_type_deltas)
            vy = interp.griddata(p, v[:,1], (grid_x, grid_y), fill_value=0., method=self.region_interp_type_deltas)
            del p,v
            if not calculate_range:
                del grid_x, grid_y
                cmax = cmin = None
            # xxx - griddata returns float64, no matter the input types.... meh
            vx = vx.astype(xdtype); vy = vy.astype(xdtype)

            if calculate_range:
                # get the range over the grid points plus deltas, only large memory variable needed
                #   after this are the interpolated deltas (vx, vy).
                tmpx = grid_x + pmin[0].astype(xdtype) + vx; tmpy = grid_y + pmin[1].astype(xdtype) + vy
                del grid_x, grid_y
                cmin = np.array([tmpx.min(), tmpy.min()]); cmax = np.array([tmpx.max(), tmpy.max()])
                del tmpx, tmpy

        return gpmin, gpmax, cmin, cmax, vx, vy
    #def _region_load_setup_deformations(self,

    def _region_load_coords(self, ind, slice_image_fn, bg_fill_type, do_bg_fill, region_size, img, bgsel, tm_bw,
            load_xform_bg, load_xform_tm, roi_points, control_points, rel_ds, verbose, doplots):
        # get the size of the output image based on the rough bounding box
        _, _, ibox_size = self._center_box_pixels(np.zeros(2), 1)
        ibox_shape = ibox_size[::-1]

        if not self.single_block:
            # get the block ranges in the output image
            _, _, _, rng = tile_nblks_to_ranges(ibox_shape, self.nblocks, self.block_overlap_pix, self.iblock)
            icoords_size = np.array([x[1] - x[0] for x in rng[::-1]])
            icrop_min = np.array([x[0] for x in rng][::-1])

            _, _, _, rng = tile_nblks_to_ranges(ibox_shape, self.nblocks, self.block_overlap_grid_pix, self.iblock)
            icoords_size_grid = np.array([x[1] - x[0] for x in rng[::-1]])
            icrop_min_grid = np.array([x[0] for x in rng][::-1])
        else:
            icoords_size = icoords_size_grid = ibox_size
            icrop_min = icrop_min_grid = np.zeros((2,),dtype=np.int64)

        # NOTE: for this method, everything works backwards. we start with the coordinates in the output block
        #   and then inverse transform them back to the source image.

        if self.deformation_points is not None:
            if verbose: print('Apply coords warping deformation'); t = time.time()
            # the points and the vectors that define the warping, use double for interpolating the vector field.
            # with the addition of the rough bounding box, the deformation points are the same for all regions.
            # NOTE: p is modified, so in reality we need this copy.
            #   However, this copy is memory-expensive for dense grids,
            #     and in practive the deformation points are only used once per-process.
            #p = self.deformation_points.astype(np.double, copy=True)
            p = self.deformation_points.astype(np.double)
            v = self.imaged_order_deformation_vectors[ind].astype(np.double)
            assert( np.isfinite(p).all() ) # non-finite points should be handled by wafer_aggregator

            # creates the dense interpolated warpings
            pmin, pmax, _, _, vx, vy = self._region_load_setup_deformations(p, v,
                    icrop_min, icrop_min_grid, icoords_size, icoords_size_grid, False, verbose, doplots)

        # create final coordinates over the rough bounding box (or subblock)
        # NOTE: all the transformations of coords need to be chosen carefully to minimize reallocation.
        #   these coords are very memory expensive and any extra copies can bloat the memory footprint.
        # xxx - the major remaining issue is that griddata (for order > 0) does not support float32
        #   it will return float64, even if all the inputs are float32
        xdtype = self.coordindate_transformations_dtype
        coords = np.indices((icoords_size[1], icoords_size[0]), dtype=xdtype)
        coords = coords[::-1,:,:] # x/y need to be swapped for coords
        if (icrop_min != 0).any(): coords += icrop_min[:,None,None]

        if self.deformation_points is not None:
            # apply the deformation to the coordinates over the bounding box of the deformation points.
            tmp = coords[0,pmin[1]:pmax[1], pmin[0]:pmax[0]]
            coords[0,pmin[1]:pmax[1], pmin[0]:pmax[0]] = tmp + vx
            del vx
            tmp = coords[1,pmin[1]:pmax[1], pmin[0]:pmax[0]]
            coords[1,pmin[1]:pmax[1], pmin[0]:pmax[0]] = tmp + vy
            del vy, tmp
            if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        # reshape for all other transforms
        coords = coords.reshape(2,-1)

        if self.region_affines is not None and self.region_affines[ind] is not None:
            if verbose: print('Apply coords rough affine xform'); t = time.time()
            tmp = self.region_affines[ind].astype(xdtype)
            coords = np.dot(tmp[:2,:2], coords)
            coords += tmp[:2,2][:,None]
            if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        if verbose: print('Apply coords microscope translation and rotation'); t = time.time()
        # get rotation center and do all xforms for roi points
        roi_points, target_ctr, rot_region_size = self._xform_points(ind, roi_points, region_size)
        igrid_min, igrid_max, _ = self._center_box_pixels(target_ctr, 1)
        # translate the coordinates based on the rough bounding box crop
        if (igrid_min != 0).any():
            if verbose: print('\tapply translation {} {}'.format(igrid_min[0], igrid_min[1]))
            coords += igrid_min[:,None]

        # apply transformations to the control points that are used for tear stitching
        if control_points is not None:
            if self.invert_control_points:
                control_points = self._inv_xform_points(ind, control_points, region_size, target_ctr)
            else:
                control_points, _, _ = self._xform_points(ind, control_points, region_size, target_ctr=target_ctr)

        # the "microscope" angle rotation (angle of the ROI as reported by limi slice detection software).
        # do not apply rotation at all if angle is very very small or zero.
        ang_rad = self.region_rotations[ind]; ang_deg = ang_rad/np.pi*180
        do_rotation = (np.abs(ang_deg) > 1e-5)
        if do_rotation:
            if verbose: print('\tapply rotation {}'.format(ang_deg))
            # rotate points to match image rotation, rotate around image center (same as image rotation).
            c, s = np.cos(ang_rad), np.sin(ang_rad)
            R_forwards = np.array([[c, -s], [s, c]], dtype=xdtype)
            coords -= (rot_region_size/2)[:,None]
            coords = np.dot(R_forwards, coords)
            coords += (region_size/2)[:,None]
        if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        if verbose: print('Use transformed coords to remap image(s)'); t = time.time()
        icoords_shape = icoords_size[::-1]; region_shape = region_size[::-1]
        coords = coords[::-1,:] # swap x/y for remap
        coords_min = coords.min(1); coords_max = coords.max(1)

        # only load from the bounding box on the source image coords
        coords_imin = np.floor(coords_min).astype(np.int64)
        coords_imin[coords_imin < 0] = 0
        coords_imax = np.ceil(coords_max).astype(np.int64)
        sel = (coords_imax >= region_shape); coords_imax[sel] = region_shape[sel] - 1
        custom_rng = [[coords_imin[x], coords_imax[x]] for x in range(2)]
        # shift coordinates to match the loaded source image bounding box
        coords -= coords_imin[:,None]

        # reshape for remap
        coords = coords.reshape((2, icoords_shape[0], icoords_shape[1]))

        # if the coordinates are all out of bounds, then just query for the datatype.
        # xxx - just setting a single pixel so that the _region_load_load_imgs code path
        #   that determines whether the background / tissue mask should be loaded or not is presevered.
        if (coords_max <= 0).any() or (coords_min >= region_shape).any() or \
                any([(x[1] - x[0]) == 0 for x in custom_rng]):
            out_of_bounds = True
            custom_rng = [[0,1], [0,1]]
        else:
            out_of_bounds = False

        # apply the transforms to the images
        # xxx - could load images one at a time, but this did not seem worth the hassle
        #   of pulling apart the image(s) loading logic.
        img, bgsel, tm_bw, _, _, _, _, _, _ = \
            self._region_load_load_imgs(ind, slice_image_fn, do_bg_fill, True, False, custom_rng, verbose)

        # xxx - gah, punted on the tissue mask, one option could be to just keep it stored upsampled,
        #   seems overly wasteful. can not think of a better solution at the moment.
        tm_bw = None

        if not out_of_bounds:
            img = nd.map_coordinates(img, coords, order=self.warping_spline_order_nd, mode='constant', prefilter=False)
            if load_xform_bg and bgsel is not None:
                bgsel = nd.map_coordinates(bgsel, coords, order=0, mode='constant', prefilter=False)
            # if load_xform_tm and tm_bw is not None:
            #     tm_bw = nd.map_coordinates(tm_bw, coords, order=0, mode='constant', prefilter=False)
        else:
            if verbose: print('All coords out of bounds')
            img = np.zeros(icoords_shape, dtype=img.dtype)
            if load_xform_bg and bgsel is not None:
                bgsel = np.zeros(icoords_shape, dtype=bgsel.dtype)
            # if load_xform_tm and tm_bw is not None:
            #     tm_bw = np.zeros(icoords_shape, dtype=tm_bw.dtype)
        if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        return img, roi_points, control_points, ibox_shape, icrop_min, bgsel, tm_bw
    #def _region_load_coords(self

    # this is mostly meant for "back-transforming" masks that were created from "microscope aligned" images.
    def _region_load_coords_inv(self, ind, region_size, img, roi_points, dsstep, verbose, doplots):
        if dsstep > 1:
            pad = (dsstep - region_size % dsstep) % dsstep
            ibox_size = (region_size + pad) // dsstep
        else:
            ibox_size = region_size
        ibox_shape = ibox_size[::-1]

        block_overlap_pix = self.block_overlap_pix // dsstep
        if not self.single_block:
            assert(False) # xxx - placeholder only, did not validate this
            # get the block ranges in the output image
            _, _, _, rng = tile_nblks_to_ranges(ibox_shape, self.nblocks, block_overlap_pix, self.iblock)
            icoords_size = np.array([x[1] - x[0] for x in rng[::-1]])
            icrop_min = np.array([x[0] for x in rng][::-1])
        else:
            icoords_size = ibox_size
            icrop_min = np.zeros((2,),dtype=np.int64)

        # NOTE: for this method, everything works forwards. we start with the coordinates in the region image
        #   and then forward transform them to the output image. this is so the inverse transform can be
        #   applied to an image that has previously been transformed, the masks for example.
        xdtype = self.coordindate_transformations_dtype
        coords = np.indices((icoords_size[1], icoords_size[0]), dtype=xdtype)
        coords = coords[::-1,:,:] # x/y need to be swapped for coords
        if (icrop_min != 0).any(): coords += icrop_min[:,None,None]
        # reshape for affine transforms
        coords = coords.reshape(2,-1)

        if verbose: print('Apply coords microscope translation and rotation'); t = time.time()

        # get rotation center and do all xforms for roi points
        roi_points, target_ctr, rot_region_size = self._xform_points(ind, roi_points, region_size)
        igrid_min, igrid_max, _ = self._center_box_pixels(target_ctr, dsstep)
        if dsstep > 1:
            pad = (dsstep - region_size % dsstep) % dsstep
            region_size = (region_size + pad) // dsstep
            pad = (dsstep - rot_region_size % dsstep) % dsstep
            rot_region_size = (rot_region_size + pad) // dsstep

        # the "microscope" angle rotation (angle of the ROI as reported by limi slice detection software).
        # do not apply rotation at all if angle is very very small or zero.
        ang_rad = self.region_rotations[ind]; ang_deg = ang_rad/np.pi*180
        do_rotation = (np.abs(ang_deg) > 1e-5)
        if do_rotation:
            if verbose: print('\tapply rotation {}'.format(ang_deg))
            # rotate around image center (same as image rotation).
            c, s = np.cos(ang_rad), np.sin(ang_rad)
            R_backwards = np.array([[c, s], [-s, c]], dtype=xdtype)
            coords -= (region_size/2)[:,None]
            coords = np.dot(R_backwards, coords)
            coords += (rot_region_size/2)[:,None]
        if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        # translate the coordinates based on the rough bounding box crop
        if (igrid_min != 0).any():
            if verbose: print('\tapply translation {} {}'.format(igrid_min[0], igrid_min[1]))
            coords -= igrid_min[:,None]

        # get the size of the output image based on the rough bounding box
        _, _, rbox_size = self._center_box_pixels(np.zeros(2), 1)
        if dsstep > 1:
            pad = (dsstep - rbox_size % dsstep) % dsstep
            rbox_size = (rbox_size + pad) // dsstep
        # # use the passed image size... some danger here if the input image is not approximately the correct size
        # rbox_size = np.array(img.shape)[::-1]

        if verbose: print('Use transformed coords to remap image(s)'); t = time.time()
        icoords_shape = icoords_size[::-1]; rbox_shape = rbox_size[::-1]
        coords = coords[::-1,:] # swap x/y for remap
        coords_min = coords.min(1); coords_max = coords.max(1)

        # only load from the bounding box on the source image coords
        coords_imin = np.floor(coords_min).astype(np.int64)
        coords_imin[coords_imin < 0] = 0
        coords_imax = np.ceil(coords_max).astype(np.int64)
        sel = (coords_imax >= rbox_shape); coords_imax[sel] = rbox_shape[sel] - 1
        custom_rng = [[coords_imin[x], coords_imax[x]] for x in range(2)]
        # # shift coordinates to match the loaded source image bounding box
        # coords -= coords_imin[:,None]

        #if verbose:
        #    print('dsstep ibox_size region_size rot_region_size igrid_min img_size:')
        #    print(dsstep, ibox_size, region_size, rot_region_size, igrid_min, img.shape[::-1])
        #    print('rbox_size coords_min coords_max coords_rng:')
        #    print(rbox_size, coords_min, coords_max, custom_rng)

        # reshape for remap
        coords = coords.reshape((2, icoords_shape[0], icoords_shape[1]))

        # special codepath incase all coordinates are out-of-bounds of the image.
        if (coords_max <= 0).any() or (coords_min >= rbox_shape).any() or \
                any([(x[1] - x[0]) == 0 for x in custom_rng]):
            if verbose: print('All coords out of bounds')
            img = np.zeros(icoords_shape, dtype=img.dtype)
        else:
            order = 0 if np.issubdtype(img.dtype, bool) else self.warping_spline_order_nd
            img = nd.map_coordinates(img, coords, order=order, mode='constant', prefilter=False)

        return img, roi_points, ibox_shape, icrop_min
    #def _region_load_coords_inv(self

    def _region_load_load_imgs(self, ind, slice_image_fn, do_bg_fill, load_img, return_pil, custom_rng, verbose):
        if verbose: print('Loading image and/or roi/bg/mask'); t = time.time()
        load_xform_bg = do_bg_fill and load_img
        load_xform_tm = False
        roi_points = tm_bw = bgsel = img = img_shape = rel_ds = control_points = None
        if self.input_data_type == msem_input_data_types.zen_msem_data or \
                self.input_data_type == msem_input_data_types.new_msem_data:
            # in the normal workflow the region outputs are always hdf5 files.
            slice_image_fn = slice_image_fn + '.h5'

            if self.load_rough_xformed_img:
                # the rough xform'ed roi_points need to have been saved along with the image
                roi_points, _ = big_img_load(slice_image_fn, dataset='roi_points')
                _nblks = self.nblocks; _iblk = self.iblock; _novlp = self.block_overlap_pix
            else:
                # this is the ROI that was defined in the limi file as recorded by the mSEM software.
                roi_points = self.region_roi_poly[ind]
                # the rough xforms do not currently support blocking, so need to load/xform the whole slice
                _nblks = [1,1]; _iblk = [0,0]; _novlp = [0,0]

            # control points are used for tear stitching
            if self.invert_control_points:
                control_points, _ = big_img_load(slice_image_fn, dataset='xcontrol_points')
            else:
                try:
                    control_points, _ = big_img_load(slice_image_fn, dataset='control_points')
                except:
                    control_points = None

            if load_img:
                # load the pre-montaged region (slice) image.
                img, img_shape = big_img_load(slice_image_fn, _nblks, _iblk, _novlp, custom_rng=custom_rng)
                if return_pil: img = Image.fromarray(img)
            else:
                img_shape, _ = big_img_info(slice_image_fn)

            if load_xform_bg:
                # another method to avoid spurious correlations, fill any zero-padded areas with noise
                # this has to be done after the image rotation, cropping and affine transformations,
                #   because the transformations also create background.
                bgsel, _ = big_img_load(slice_image_fn, _nblks, _iblk, _novlp, dataset='background',
                        custom_rng=custom_rng)
                if return_pil: bgsel = Image.fromarray(bgsel)

            # xxx - because it is relatively low overhead, decided to just always xform the tissue masks
            #   if they are present in the region file, even if they are not going to be used.
            try:
                attrs = {'ds':0}
                tm_bw, _ = big_img_load(slice_image_fn, dataset='tissue_mask', attrs=attrs)
                load_xform_tm = True
            except:
                load_xform_tm = False
            if load_xform_tm:
                rel_ds = attrs['ds'] // self.dsstep
                if return_pil: tm_bw = Image.fromarray(tm_bw)
        elif self.input_data_type == msem_input_data_types.image_stack:
            slice_image_fn = slice_image_fn + self.region_ext
            img = cached_image_load(slice_image_fn, cache_dir=self.cache_dir, return_pil=True)
            img_shape = img.size[:2][::-1]
            if load_img:
                if not return_pil: img = np.asarray(img) #.copy()
            else:
                # xxx - could query size here instead of loading image.
                #   decided it was not worth the effort since this pathway is typically for smallerish images.
                img = None
        elif self.input_data_type == msem_input_data_types.hdf5_stack:
            slice_image_fn = slice_image_fn + self.region_ext
            roi_points, _ = big_img_load(slice_image_fn, dataset='roi_points')
            if load_img:
                if custom_rng is None:
                    _nblks = self.nblocks; _iblk = self.iblock; _novlp = self.block_overlap_pix
                else:
                    _nblks = [1,1]; _iblk = [0,0]; _novlp = [0,0]
                img, img_shape = big_img_load(slice_image_fn, _nblks, _iblk, _novlp, custom_rng=custom_rng)
                if return_pil: img = Image.fromarray(img)
            else:
                img_shape, _ = big_img_info(slice_image_fn)
        if verbose: print('\tdone in %.4f s' % (time.time() - t, ))

        img_size = np.array(img_shape[::-1])
        return img, bgsel, tm_bw, load_xform_bg, load_xform_tm, roi_points, control_points, rel_ds, img_size
    #def _region_load_load_imgs(self

    def _get_region_filename(self, ind):
        if self.sel_missing_regions[ind]: assert(False) # missing region, deal with this elsewhere
        wafer_ind = 0 if not self.is_cross_wafer else ind
        subfolder = self.load_img_subfolder
        if isinstance(subfolder, (list, tuple)): subfolder = subfolder[ind]
        slice_image_fn = os.path.join(self.alignment_folders[wafer_ind], subfolder,
                self.wafer_strs[wafer_ind] + self.wafer_region_strs[ind] + self.region_suffix)
        #if subfolder and not os.path.isfile(subfolder):
        if subfolder and not os.path.isdir(os.path.join(self.alignment_folders[wafer_ind], subfolder)):
            # xxx - hacky way to specify completely different load folder.
            #   subject to high jack with matching subfolder.... very unlikely.
            slice_image_fn = os.path.join(subfolder,
                    self.wafer_strs[wafer_ind] + self.wafer_region_strs[ind] + self.region_suffix)

        return slice_image_fn

    # this is the main workhouse of wafer. it loads the saved montaged region/slices
    #     which is the result of the 2d alignment (mfov and region) and applies all
    #     of the 3d alignment transformations:
    #   (1) microscope alignment, rotation and region roi center (from czi/cz file)
    #   (2) rough alignment, affine transformation calculated by wafer_solver
    #   (3) fine alignment, from deltas calculated in align_regions method below
    def _region_load(self, ind, bg_fill_type='noise', return_pil=True, verbose=False, doplots=False):
        slice_image_fn = self._get_region_filename(ind)
        print(slice_image_fn)

        do_bg_fill = (self.region_bg_noise_fill and bg_fill_type != 'none')
        if verbose: print('do background fill {}'.format(do_bg_fill))

        load_img = not self.use_coordinate_based_xforms
        img, bgsel, tm_bw, load_xform_bg, load_xform_tm, roi_points, control_points, rel_ds, region_size = \
            self._region_load_load_imgs(ind, slice_image_fn, do_bg_fill, load_img, True, None, verbose)

        # the original load method applied transformations directly to the images in the forward direction.
        # this is left in as _region_load_xforms. this method only supports block processing for the fine alignemnt.
        # the new method transforms the output coordinates in reverse back to the region images,
        # called _region_load_coords and it fully supports block processing, including for the rough transformations.
        f_region_load = self._region_load_coords if self.use_coordinate_based_xforms else self._region_load_xforms
        img, roi_points, control_points, img_shape, icrop_min, bgsel, tm_bw = \
            f_region_load(ind, slice_image_fn, bg_fill_type, do_bg_fill, region_size, img, bgsel, tm_bw,
                load_xform_bg, load_xform_tm, roi_points, control_points, rel_ds, verbose, doplots)

        if type(img) is Image.Image:
            if not return_pil: img = np.asarray(img) #.copy()
        else:
            if return_pil: img = Image.fromarray(img)

        if load_xform_tm and tm_bw is not None:
            # keep the downsampled, xformed tissue mask and its relative downsampling
            self.tissue_mask_bw[ind] = tm_bw
            self.tissue_mask_bw_rel_ds = rel_ds

        return img, roi_points, control_points, img_shape, icrop_min, bgsel
    # def _region_load(self, ind

    # get selects of which grid points are within the roi polygon (if loaded / specified).
    def _get_grid_selects(self, i, grid_points=None, roi_points=None):
        if grid_points is None: grid_points = self.grid_locations_pixels

        grid_selects = None
        if roi_points is not None and self.roi_polygon_scale > 0:
            ctr = PolyCentroid(roi_points[:,0], roi_points[:,1])
            roi_points_scl = (roi_points - ctr)*self.roi_polygon_scale + ctr
            grid_selects = Path(roi_points_scl).contains_points(grid_points)

        bw = None
        if self.use_tissue_masks:
            if self.tissue_mask_path is not None:
                # xxx - OLDtissue - remove this whole conditional and use else when fully moved to saving the masks
                fn = os.path.join(self.tissue_mask_path, self.tissue_mask_fn_str.format(self.zorder[i]))
                bw = tifffile.imread(fn).astype(bool)
                rel_ds = self.tissue_mask_ds // self.dsstep
            else:
                bw = self.tissue_mask_bw[i]
                rel_ds = self.tissue_mask_bw_rel_ds

            if self.tissue_mask_min_size > 0:
                # remove small components
                labels, nlbls = nd.label(bw, structure=nd.generate_binary_structure(2,2))
                if nlbls > 0:
                    sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                    rmv = np.nonzero(sizes < self.tissue_mask_min_size)[0] + 1
                    if rmv.size > 0:
                        bw[np.isin(labels, rmv)] = 0

            if not np.isfinite(self.tissue_mask_min_hole_size):
                # use this to mean fill all topological holes
                # pad image so that no areas outside the main slice mask are considered holes.
                tmp = np.pad(bw,2)
                labels, nlbls = nd.label(np.logical_not(tmp),
                    structure=nd.generate_binary_structure(2,1))
                if nlbls > 0:
                    add = np.arange(1,nlbls+1); add = add[add != labels[0,0]]
                    if add.size > 0:
                        bw[np.isin(labels[2:-2,2:-2], add)] = 1
            elif self.tissue_mask_min_hole_size > 0:
                # remove small holes
                labels, nlbls = nd.label(np.logical_not(bw),
                    structure=nd.generate_binary_structure(2,1))
                if nlbls > 0:
                    sizes = np.bincount(np.ravel(labels))[1:] # component sizes
                    add = np.nonzero(sizes < self.tissue_mask_min_hole_size)[0] + 1
                    if add.size > 0:
                        bw[np.isin(labels, add)] = 1

            if self.tissue_mask_bwdist > 0:
                bw = (nd.distance_transform_edt(np.logical_not(bw)) < self.tissue_mask_bwdist)

            ipts = np.round(grid_points / rel_ds).astype(np.int64)
            mask = np.logical_and((ipts >= 0).all(1), (ipts < np.array(bw.shape)[None,::-1]).all(1))
            mask2 = np.array([bw[x[1],x[0]] for x in ipts[mask,:]]); mask[mask] = mask2
            grid_selects = mask if grid_selects is None else np.logical_and(grid_selects, mask)
        #if self.use_tissue_masks:

        if grid_selects is None:
            grid_selects = np.ones((self.ngrid,), bool)
            roi_points_scl = roi_points

        return grid_selects, roi_points_scl, bw

    def _load_grid_region_crops(self, gridnums, regions, imgs=None, imgs_proc=None, rois_points=None,
                                doproc=True, return_imgs=False, bg_fill_type='noise'):
        ngrid = len(gridnums); nregions = len(regions)
        if self.wafer_verbose:
            print('\tLoading crops for %d regions for %d grid location(s)' % (nregions, ngrid))
            t = time.time()

        # decided to default full_only to True so that all images and templates are always the same size.
        # this rejects any crops for grid points that are very near the edge of the rough bounding box.
        full_only = True

        csz = self.crop_size_pixels; tcsz = self.tcrop_size_pixels # crop sizes in pixels
        # crop out template images to compare from the center each region after rotating.
        timages = [[None]*self.nregions for x in range(ngrid)]
        images = [[None]*self.nregions for x in range(ngrid)]
        timages_proc = [[None]*self.nregions for x in range(ngrid)]
        images_proc = [[None]*self.nregions for x in range(ngrid)]
        grid_centers = np.zeros((ngrid,self.nregions,2), dtype=np.double)
        grid_selects = np.zeros((ngrid,self.nregions), dtype=bool)
        empty = np.empty((0,0))
        full_cnt = np.zeros(self.nregions, dtype=np.int64)

        # mechanism for not having to reload the images
        doload = (imgs is None)
        if doload:
            imgs = [None]*self.nregions
            imgs_proc = [None]*self.nregions
            rois_points = [None]*self.nregions

        for i in regions:
            if doload:
                imgs[i], rois_points[i], _, img_shape, icrop_min, _ = \
                    self._region_load(i, bg_fill_type=bg_fill_type, return_pil=False)

                # this is an optimization for speed, with high image coverage it's better to preproc all at once.
                if ngrid > self.ngrid_cutoff and doproc:
                    imgs_proc[i] = template_match_preproc(imgs[i], whiten_sigma=self._proc_whiten_sigma,
                            aligned_copy=not self.image_crops_copy, clahe_clipLimit=self._proc_clahe_clipLimit,
                            clahe_tileGridSize=self._proc_clahe_tileGridSize, filter_size=self._proc_filter_size,
                            xcorr_img_size=csz.prod())
            img = imgs[i]; img_proc = imgs_proc[i]
            #shape = img.shape if img is not None else img_proc.shape
            # since the addition of the rough bounding box, ctrs are the same for all images
            grid_selects[:,i], _, _ = self._get_grid_selects(i, roi_points=rois_points[i])
            grid_centers[:,i,:] = self.grid_locations_pixels[gridnums[:ngrid],:] - icrop_min
            if rois_points[i] is not None:
                rois_points[i] = rois_points[i] - icrop_min # must be after _get_grid_selects

            for j in range(ngrid):
                # crop out from each specified grid location
                tc = np.round(grid_centers[j,i,:] - tcsz/2).astype(np.int32)
                c = np.round(grid_centers[j,i,:] - csz/2).astype(np.int32)
                beg = c; tbeg = tc; end = c + csz; tend = tc + tcsz
                # clip negative indices, i.e., prevent numpy negative indexing
                beg[beg < 0] = 0; tbeg[tbeg < 0] = 0
                end[end < 0] = 0; tend[tend < 0] = 0

                if img is not None:
                    # crop out specified amount around center of the region
                    #images[j][i] = img[c[1]:c[1]+csz[1],c[0]:c[0]+csz[0]]
                    #timages[j][i] = img[tc[1]:tc[1]+tcsz[1],tc[0]:tc[0]+tcsz[0]]
                    images[j][i] = img[beg[1]:end[1],beg[0]:end[0]]
                    timages[j][i] = img[tbeg[1]:tend[1],tbeg[0]:tend[0]]
                    if full_only and (not all([x==y for x,y in zip(images[j][i].shape,csz)]) or \
                            not all([x==y for x,y in zip(timages[j][i].shape,tcsz)])):
                        images[j][i] = timages[j][i] = empty
                    else:
                        full_cnt[i] += 1
                    if self.image_crops_copy:
                        images[j][i] = images[j][i].copy()
                        timages[j][i] = timages[j][i].copy()

                if doproc:
                    if ngrid > self.ngrid_cutoff:
                        #_img = img_proc[c[1]:c[1]+csz[1],c[0]:c[0]+csz[0]]
                        #_timg = img_proc[tc[1]:tc[1]+tcsz[1],tc[0]:tc[0]+tcsz[0]]
                        _img = img_proc[beg[1]:end[1],beg[0]:end[0]]
                        _timg = img_proc[tbeg[1]:tend[1],tbeg[0]:tend[0]]
                        if full_only and (not all([x==y for x,y in zip(_img.shape,csz)]) or \
                                not all([x==y for x,y in zip(_timg.shape,tcsz)])):
                            _img = _timg = empty
                        if self.image_crops_copy:
                            images_proc[j][i] = pyfftw_alloc_like(_img)
                            images_proc[j][i].flat[:] = _img.flat[:]
                            timages_proc[j][i] = pyfftw_alloc_like(_timg)
                            timages_proc[j][i].flat[:] = _timg.flat[:]
                        else:
                            images_proc[j][i] = _img
                            timages_proc[j][i] = _timg
                    else:
                        # preprocess for template matching
                        images_proc[j][i] = template_match_preproc(images[j][i],
                                whiten_sigma=self._proc_whiten_sigma, clahe_clipLimit=self._proc_clahe_clipLimit,
                                clahe_tileGridSize=self._proc_clahe_tileGridSize, filter_size=self._proc_filter_size,
                                xcorr_img_size=csz.prod())
                        timages_proc[j][i] = template_match_preproc(timages[j][i],
                                whiten_sigma=self._proc_whiten_sigma, clahe_clipLimit=self._proc_clahe_clipLimit,
                                clahe_tileGridSize=self._proc_clahe_tileGridSize, filter_size=self._proc_filter_size,
                                xcorr_img_size=csz.prod())
                    #if ngrid > self.ngrid_cutoff:
                #if doproc:
            #for j in range(ngrid):

        full_cnt0 = full_cnt[regions[0]]
        assert((full_cnt[regions] == full_cnt0).all()) # you screwed up the grid points somehow
        if self.wafer_verbose:
            if full_only: print('\t\t{} total full imgs/tmpls included'.format(full_cnt0))
            print('\t\tdone in %.4f s' % (time.time() - t, ))

        if return_imgs:
            return images, timages, images_proc, timages_proc, grid_centers, grid_selects, imgs, imgs_proc, \
                rois_points, full_cnt0
        else:
            return images, timages, images_proc, timages_proc, grid_centers, grid_selects, full_cnt0

    def _validate_region_grid(self, start=0, per_figure=2, use_solved_order=False, show_proc=False, show_patches=False,
                              bg_fill_type='none', show_grid=True):
        # https://stackoverflow.com/questions/39248245/factor-an-integer-to-something-as-close-to-a-square-as-possible
        def factor_int(n):
            nsqrt = np.ceil(np.sqrt(n))
            solution = False
            val = nsqrt
            while not solution:
                val2 = int(n/val)
                if val2 * val == float(n):
                    solution = True
                else:
                    val-=1
            return int(val), val2
        val, val2 = factor_int(per_figure)
        if val > val2: tmp=val2; val2=val; val=tmp

        if show_patches:
            csz = self.crop_size_pixels; tcsz = self.tcrop_size_pixels # crop sizes in pixels

        for i in range(start, self.nregions, per_figure):
            plt.figure()
            plt.gcf().set_size_inches(8,8)
            for j in range(per_figure):
                x = (self.solved_order[i+j] if use_solved_order else (i+j))
                images, timages, images_proc, timages_proc, grid_centers, grid_selects, imgs, imgs_proc, \
                    rois_points, _ = self._load_grid_region_crops(range(self.ngrid), [x], doproc=show_proc,
                        return_imgs=True, bg_fill_type=bg_fill_type)

                img = imgs_proc[x] if show_proc else imgs[x]
                ax = plt.subplot(val,val2,j+1)
                plt.imshow(img, cmap='gray')
                #pt = np.array(img.shape[::-1])/2; ax.text(pt[0],pt[1], 'x', color='blue', fontsize=16)

                for k in range(self.ngrid):
                    if show_patches:
                        tc = np.round(grid_centers[k,x,:] - tcsz/2).astype(np.int32)
                        c = np.round(grid_centers[k,x,:] - csz/2).astype(np.int32)
                        rect = patches.Rectangle(c,csz[0],csz[1],linewidth=1,edgecolor='g',facecolor='none')
                        ax.add_patch(rect)
                        rect = patches.Rectangle(tc,tcsz[0],tcsz[1],linewidth=1,edgecolor='b',facecolor='none')
                        ax.add_patch(rect)

                    if show_grid:
                        ax.scatter(grid_centers[k,x,0],grid_centers[k,x,1], 12, 'g' if grid_selects[k,x] else 'r', '.')

                if rois_points[x] is not None:
                    poly = patches.Polygon(rois_points[x], linewidth=2, edgecolor='r',
                                           facecolor='none', linestyle='--',)
                    ax.add_patch(poly)
                    if self.roi_polygon_scale > 0:
                        ctr = PolyCentroid(rois_points[x][:,0], rois_points[x][:,1])
                        roi_points_scl = (rois_points[x] - ctr)*self.roi_polygon_scale + ctr
                        poly_scl = patches.Polygon(roi_points_scl, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(poly_scl)

                plt.gca().axis('off')
                plt.title('region %d (base 1)' % (x+1,))
            #plt.savefig('_validate_region_grid.png', dpi=300)
            plt.show()


    def align_regions(self, grid_selects=None, doplots=False, dosave_path='', nworkers=None):
        if self.wafer_verbose:
            print('\nDoing xcorr alignment for solved ordering of %d regions:' % (self.nregions,))
            print(self.solved_order)
            print(self.region_inds[self.solved_order])
            ttotal = time.time()

        # NOTE: can not do query cuda devices in a global init because cuda can not be forked,
        #   meaning any processes that try to use cuda will fail. another option is 'spawn'
        #   but all the conditional code required for this is a nightmare.
        self.query_cuda_devices()
        # nworkers is only used by the rcc-xcorr gpu version, does not matter for other methods.
        # xxx - nworkers is actually now just a hook because more than one worker did not help for small batches.
        #   currently it is force to 1 in _template_match
        if nworkers is None: nworkers = self.nthreads
        print('using {} method for computing xcorrs, backend {}, use_gpu {}, num_gpus {}, num_workers {}'.\
            format(self.fft_method, self.fft_backend, self.use_gpu, self.cuda_device_count, nworkers))

        # grid_selects allows subset of deltas to be computed by caller.
        if grid_selects is None: grid_selects = np.ones((self.ngrid,), dtype=bool)

        self.wafer_grid_deltas = np.zeros((self.nregions, self.ngrid, 2), dtype=np.double)
        self.wafer_grid_Cvals = np.zeros((self.nregions, self.ngrid), dtype=np.double)

        imgs_shape = tpls_shape = None # to make sure they are all the same size

        # use the cross-correlation method to calculate local deltas at grid locations.
        for i in range(self.nregions-1):
            icur = self.solved_order[i]; inxt = self.solved_order[i+1]

            # ignore any crops out of block (should have overlap set); causes problems for the block outliers.
            # allow smaller image crops for full slice comparisons (i.e. not single block).
            _, _, images_proc, timages_proc, grid_centers, grid_selects_poly, full_cnt = \
                self._load_grid_region_crops(range(self.ngrid), [icur,inxt])

            if self.wafer_verbose:
                sel = np.logical_and(grid_selects, grid_selects_poly[:,[icur,inxt]].all(1))
                print('\tAligning region %d to %d, %d of %d, at %d / %d locations (block count %d)' % \
                      (inxt,icur,i,self.nregions,sel.sum(),self.ngrid,full_cnt))
                print('\tCrop size {} x {}'.format(self.crop_um[0], self.crop_um[1]))
                sel = None
                t = time.time()

            # iterate defined grid locations
            C = np.empty((self.ngrid,), dtype=np.double); C.fill(-np.inf)
            A = np.zeros((self.ngrid,), dtype=np.double)
            # template_match_rotate_images already returns deltas as double (between image centers)
            D = np.zeros((self.ngrid,2), dtype=np.double)
            ctemplate0s = [None]*self.ngrid
            nprint = 0
            valid_cnt = 0
            tin = time.time()
            for x in range(self.ngrid):
                # the template matching is done with the next image as the image and the current as the template.
                if grid_selects[x] and grid_selects_poly[x,[icur,inxt]].all() and all([(x < y and x > 0 and y > 0) \
                        for x,y in zip(timages_proc[x][icur].shape, images_proc[x][inxt].shape)]):
                    if imgs_shape is None:
                        imgs_shape = np.array(images_proc[x][inxt].shape)
                        tpls_shape = np.array(timages_proc[x][icur].shape)
                    else:
                        #assert((imgs_shape == np.array(images_proc[x][inxt].shape)).all())
                        #assert((tpls_shape == np.array(tpls_shape[x][inxt].shape)).all())
                        if (imgs_shape != np.array(images_proc[x][inxt].shape)).any():
                            print(imgs_shape, images_proc[x][inxt].shape)
                            assert(False) # image shapes mismatch
                        if (tpls_shape != np.array(timages_proc[x][inxt].shape)).any():
                            print(tpls_shape, timages_proc[x][inxt].shape)
                            assert(False) # template shapes mismatch
                    C[x], A[x], D[x,:], ctemplate0s[x] = template_match_rotate_images([timages_proc[x][icur]],
                        [images_proc[x][inxt]], self.delta_rotation_step, self.delta_rotation_range,
                        interp_type=wafer.region_interp_type, doplots=doplots, dosave_path=dosave_path,
                        return_ctemplate0=self.export_xcorr_comps_path is not None, use_gpu=self.use_gpu,
                        num_gpus=self.cuda_device_count, nworkers=nworkers)
                    valid_cnt += 1
                    if nprint > 0 and valid_cnt % nprint == 0:
                        print('\t\t{} / {} in {:.4f}'.format(valid_cnt,full_cnt,time.time() - tin,)); tin = time.time()

            if self.export_xcorr_comps_path is not None:
                # xxx - hacky way to export xcorr comparisons / results for validating other methods
                import dill
                import scipy.io as io
                assert( self.delta_rotation_range[0] == self.delta_rotation_range[1] == 0 ) # did not implement
                assert( self.nregions == 2 ) # did not implement
                print('Dumping xcorr results for external comparison / validation')
                t = time.time()
                pn = self.export_xcorr_comps_path
                comps = np.zeros((self.ngrid,2), dtype=np.int64)
                Camax = np.zeros((self.ngrid,2), dtype=np.int64)
                Cmax = np.zeros((self.ngrid,), dtype=np.double)

                ncomps = 0
                for x in range(self.ngrid):
                    if np.isfinite(C[x]) and images_proc[x][inxt].size > 0:
                        tifffile.imwrite(os.path.join(pn,'image{:04d}.tif'.format(x)), images_proc[x][inxt])
                        tifffile.imwrite(os.path.join(pn,'templ{:04d}.tif'.format(x)), ctemplate0s[x])
                        comps[ncomps,:] = [x,x]
                        # convert back to amax in the full xcorr output
                        Camax[ncomps,:] = np.round(D[x,:] + \
                            np.array(images_proc[x][inxt].shape)/2 + np.array(ctemplate0s[x].shape)/2 - 1)[::-1]
                        Cmax[ncomps] = C[x]
                        ncomps += 1
                comps = comps[:ncomps,:]
                Camax = Camax[:ncomps,:]
                Cmax = Cmax[:ncomps]

                d = {'comps':comps, 'Cmax':Cmax, 'Camax':Camax}
                with open(os.path.join(pn,'comps.dill'), 'wb') as f: dill.dump(d, f)
                io.savemat(os.path.join(pn,'comps.mat'), d)
                print('\tdone in %.4f s' % (time.time() - t, ))
            #if self.export_xcorr_comps_path is not None:

            if self.wafer_verbose:
                print('\t\tdone in %.4f s' % (time.time() - t, ))

            self.wafer_grid_deltas[inxt,:,:] = D
            self.wafer_grid_Cvals[inxt,:] = C
        #for i in range(self.nregions-1):

        if self.wafer_verbose:
            print('\tdone in %.4f s' % (time.time() - ttotal, ))


    # validate rotation, translation and affine transforms with overlay at specified grid locations.
    def _validate_align_regions(self, images=None, timages=None, images_proc=None, timages_proc=None, gridnums=None,
                                start=0, stop=None, use_delta=True):
        if stop is None: stop = self.nregions-1
        if gridnums is None: gridnums = range(self.ngrid)

        csz = self.crop_size_pixels; tcsz = self.tcrop_size_pixels # crop sizes in pixels
        if images is None:
            _, _, cimages_proc, ctimages_proc, cgrid_centers, cgrid_selects, _ = \
                self._load_grid_region_crops(gridnums, [self.solved_order[x] for x in range(start,stop+1)])

        for k,cnt in zip(range(start, stop), range(self.nregions-1)):
            i = self.solved_order[k]; j = self.solved_order[k+1]
            for x in range(len(gridnums)):
                if use_delta:
                    D = self.wafer_grid_deltas[j,gridnums[x],:]
                    C = self.wafer_grid_Cvals[j,gridnums[x],:]
                    if not np.isfinite(C).all(): continue
                else:
                    D = np.zeros((2,), dtype=np.double)
                plt.figure()
                plt.subplot(2, 1, 1); plt.imshow(timages[x][i], cmap='gray')
                plt.title("order=%d, grid=%d" % (k,gridnums[x]))
                ax = plt.subplot(2, 1, 2); plt.imshow(images[x][j], cmap='gray')
                plt.title("order=%d, grid=%d" % (k,gridnums[x]))
                ax.text(csz[0]/2, csz[1]/2, 'x', color='red')
                ax.text(D[0]+csz[0]/2, D[1]+csz[1]/2, 'x', color='blue')

                # overlay in color
                plt.figure()
                outsize = np.round(2*csz).astype(np.int64)
                img = np.zeros(np.concatenate((outsize[::-1], [3])), dtype=np.double)
                dx, dy = np.round(np.zeros((2,),dtype=np.double) - (csz/2 - outsize/2)).astype(np.int32)
                img[dy:dy+csz[1], dx:dx+csz[0], 0] = images[x][j]
                dx, dy = np.round(D - (tcsz/2 - outsize/2)).astype(np.int32)
                img[dy:dy+tcsz[1], dx:dx+tcsz[0], 1] = timages[x][i]
                ax = plt.subplot(1, 1, 1); plt.imshow(img/img.max())
                plt.title("order=%d, grid=%d, C=%.3f" % (k,gridnums[x],C))
                ll=20; plt.xlim([dx-ll, dx+tcsz[0]+ll]); plt.ylim([dy+tcsz[0]+ll, dy-ll])
                ax.text(D[0]+outsize[0]//2, D[1]+outsize[0]//2, 'x', color='blue')
                ax.text(outsize[0]//2, outsize[0]//2, 'x', color='green')
                plt.show()



    ### EXPORT

    def export_regions(self, outpaths, suffixes, dssteps=[1], use_solved_order=False, crop_to_grid=False,
            start=0, stop=-1, bg_fill_type='none', do_overlays=False, export_solved_order=None, save_h5=False,
            zero_outside_grid=False, save_roi_points=False, convert_hdf5s=False, order_name_str='order',
            tissue_masks=False, init_locks=False, is_excluded=False, save_masks_in='', xform_control_points=False,
            inv_xform_control_points=False, verbose_load=False):
        stop = (self.solved_order.size if use_solved_order else self.nregions) if stop < 0 else stop

        # optionally output at multiple downsamplings, saves time so that image is only loaded once
        ndssteps = len(dssteps)
        assert( len(outpaths) == len(suffixes) == ndssteps )
        assert( stop - start > 0 ) # bad stop and start indices specified
        assert( not (use_solved_order and export_solved_order is not None) ) # do not use both
        assert( self.single_block or not do_overlays ) # no overlays for hdf5 outputs
        assert( not convert_hdf5s or not save_roi_points ) # no saving roi points during hdf5 conversion

        # xxx - currently still allowing both modes, tissue mask saved with regions (new), or as tiffs (old)
        rel_ds = self.tissue_mask_bw_rel_ds if self.tissue_mask_bw_rel_ds > 0 else self.tissue_mask_ds // self.dsstep

        self.invert_control_points = inv_xform_control_points

        if self.wafer_verbose:
            print('Writing output tiffs for %d regions' % (stop-start, ))
            print('\tCrop to grid {}, solved order {}, overlays {}, zero outside grid {}, convert hdf5s {}, init {}'.\
                format(
                    ('yes' if crop_to_grid else 'no'),
                    ('yes' if (use_solved_order or export_solved_order is not None) else 'no'),
                    ('yes' if do_overlays else 'no'), ('yes' if zero_outside_grid else 'no'),
                    ('yes' if convert_hdf5s else 'no'),
                    ('yes' if init_locks else 'no'), ))
            if save_masks_in:
                print('Saving transformed tissue masks back into regions')
            elif tissue_masks:
                print('Exporting tissues masks instead of slices')
            elif inv_xform_control_points:
                print('Saving inverse transformed control points')
            elif xform_control_points:
                print('Saving transformed control points')
            print('\toutpaths: ' + str(outpaths))
            print('\tsuffixes: ' + str(suffixes))
            print('\tdssteps: ' + str(dssteps))
            print('\tiblock: {}, nblocks {}'.format(self.iblock, self.nblocks))
            t = time.time(); dt = t

        # the grid centers were already centered on the rough bounding box in init
        grid_min = self.grid_locations_pixels.min(0)
        grid_max = self.grid_locations_pixels.max(0)
        grid_center = (grid_min + grid_max)/2

        for i,cnt in zip(range(start,stop), range(stop-start)):
            x = ((self.solved_order[i] if export_solved_order is None else export_solved_order[i]) \
                if use_solved_order else i)
            wafer_ind = 0 if not self.is_cross_wafer else x
            # had to deal with possibility of empty regions on the experimental side.
            if self.sel_missing_regions[x]: continue
            if self.wafer_verbose and cnt>0 and cnt%10==0:
                print('\t%d of %d in %.4f s' % (cnt+1,stop-start,time.time() - dt, )); dt = time.time()

            if save_masks_in:
                # load the mask from the specified folder
                tind = self.zorder[cnt]
                slice_image_fn = self._get_region_filename(x)
                if self.tissue_mask_fn_str:
                    pfn = self.tissue_mask_fn_str.format(tind)
                else:
                    pfn = os.path.basename(slice_image_fn)
                fn = os.path.join(save_masks_in, pfn)
                bw = tifffile.imread(fn).astype(bool)

                # invert the alignments in which the mask was generated
                #   in order for it to align with the slice region image.
                _, _, _, _, _, roi_points, _, _, region_size = self._region_load_load_imgs(x,
                        slice_image_fn, False, False, False, None, verbose_load)
                bw, roi_points, ibox_shape, icrop_min = self._region_load_coords_inv(x,
                        region_size, bw, roi_points, rel_ds, verbose_load, False)

                # save the mask back to the region h5 file.
                h5fn = slice_image_fn + '.h5'
                print("Saving back-transformed mask '{}' to '{}'".format(fn, h5fn))
                big_img_save(h5fn, bw, img_shape=bw.shape, dataset='tissue_mask',
                    compression=True, recreate=True, overwrite_dataset=True, attrs={'ds':self.tissue_mask_ds})

                # this is a special code path, so do not do anything else within the loop.
                continue
            elif not convert_hdf5s and not init_locks:
                img, roi_points, control_points, img_shape, icrop_min, bgsel = \
                    self._region_load(x, bg_fill_type=bg_fill_type, verbose=verbose_load)
                # get the image, rough bounding box and grid centers.
                img_sz = np.array(img.size, dtype=np.int64); img_ctr = img_sz/2
                img_shape_blk = img_shape

            if xform_control_points or inv_xform_control_points:
                slice_image_fn = self._get_region_filename(x) + '.h5'
                if inv_xform_control_points:
                    big_img_save(slice_image_fn, control_points, control_points.shape, dataset='control_points',
                        recreate=True, overwrite_dataset=True)
                else:
                    big_img_save(slice_image_fn, control_points, control_points.shape, dataset='xcontrol_points',
                        recreate=True, overwrite_dataset=True)
                # this is a special code path, so do not do anything else within the loop.
                continue

            if crop_to_grid and not init_locks:
                assert(not zero_outside_grid or not self.use_tissue_masks) # xxx - not implemented
                # convert to integer crops so that all cropped output images are exactly the same size.
                # this works because the grid itself remains fixed for all regions.
                igrid_min = np.round(grid_min).astype(np.int64)
                igrid_max = igrid_min + np.round(grid_max-grid_min).astype(np.int64)
                # the final output image shape is from the grid size.
                # this is the total output shape, not the current block size (which is img_sz).
                img_shape = (igrid_max - igrid_min)[::-1]
                if not convert_hdf5s:
                    # this is used to calculate the ranges in big_img_save.
                    # big_img_save is based on image shape not size, so need to swap x/y
                    bcrop = [[igrid_min[1], img_shape_blk[0]-igrid_max[1]],
                             [igrid_min[0], img_shape_blk[1]-igrid_max[0]]]

                if self.single_block or convert_hdf5s:
                    bcrop_min = igrid_min; bcrop_max = igrid_max; icrop_min = igrid_min
                else:
                    # crop out the portion of this block that is outside the grid,
                    #   based on where it was cropped out of the entire image (icrop_min).
                    bcrop_min = np.zeros_like(icrop_min); sel = (icrop_min < igrid_min)
                    bcrop_min[sel] = igrid_min[sel] - icrop_min[sel]
                    icrop_max = icrop_min + img_sz
                    bcrop_max = img_sz.copy(); sel = (icrop_max > igrid_max)
                    bcrop_max[sel] = bcrop_max[sel] - (icrop_max[sel] - igrid_max[sel])

                    # update the cropping locations out of the entire image
                    icrop_min = np.maximum(igrid_min, icrop_min)
                    icrop_max = np.minimum(igrid_max, icrop_max)

                if not convert_hdf5s:
                    # crop the image based on the bounding box of the grid
                    img = img.crop((bcrop_min[0], bcrop_min[1], bcrop_max[0], bcrop_max[1]))
                    if roi_points is not None: roi_points = roi_points - icrop_min
            else: #if crop_to_grid and not init_locks
                bcrop = None

            # zero out any areas outside of the scaled polygon points.
            if (zero_outside_grid or do_overlays) and not convert_hdf5s and roi_points is not None:
                _, roi_points_scl, tissue_mask_bw = self._get_grid_selects(x, roi_points=roi_points)
                if zero_outside_grid:
                    fill_outside_polygon(None, roi_points_scl, imgpil=img)

            if not convert_hdf5s and not init_locks:
                img = np.asarray(img) # convert fully to numpy array from PIL image
                if zero_outside_grid and tissue_mask_bw is not None:
                    if rel_ds > 1:
                        print('Upsampling tissue mask by {}'.format(rel_ds)); t = time.time()
                        tissue_mask_bw = block_construct(tissue_mask_bw, rel_ds)
                        print('\tdone upsampling in %.4f s' % (time.time() - t, ))
                    # need to crop again after upsampling because of padding for downsampling
                    tissue_mask_bw = tissue_mask_bw[:img.shape[0], :img.shape[1]]
                    # zero out everything outside of mask (as with polygon)
                    img[tissue_mask_bw] = 0
                    del tissue_mask_bw

            for j in range(ndssteps):
                dsstep = dssteps[j]; outpath = outpaths[j]; suffix = suffixes[j]
                os.makedirs(outpath, exist_ok=True)
                if len(order_name_str) == 0:
                    order_str = ''
                elif use_solved_order:
                    order_str = '{}{:05d}_'.format(order_name_str,i,)
                elif export_solved_order is not None:
                    order_str = '{}{:05d}_'.format(order_name_str,export_solved_order[i],)
                else:
                    order_str = ''
                fn = self.wafer_strs[wafer_ind] + order_str + self.wafer_region_strs[x] + suffix
                if is_excluded: fn = 'x' + fn
                fn = os.path.join(outpath, fn)
                h5fn = fn + '.h5'

                if init_locks:
                    assert( not (not save_h5 and (self.single_block or convert_hdf5s)) ) # not saving h5
                    print('Initializing h5 and h5 locks files for fine export')
                    big_img_init(h5fn)
                    continue

                # for converting hdf5 images, load the non-downsampled image on the first iteration
                # xxx - not sure how we would support blockwise processsing for exporting the tiffs
                #   exporting tiffs still relies on a single slice being able to fit into memory.
                #   for such a thing to work, the "knossos cubing" step would have to support a different format.
                if convert_hdf5s and j==0:
                    if self.wafer_verbose:
                        print('\tLoading hdf5 "' + h5fn + '"'); lt = time.time()
                    if crop_to_grid:
                        # hdf5s are stored in images space, flip xy
                        rng = [[bcrop_min[1], bcrop_max[1]], [bcrop_min[0], bcrop_max[0]]]
                        print('\t\tCrop to grid slice (y/x) range {}-{}, {}-{}'.format(rng[0][0],rng[0][1],
                            rng[1][0], rng[1][1]))
                    else:
                        rng = None
                    img, _ = big_img_load(h5fn, custom_rng=rng)
                    if self.wafer_verbose:
                        print('\t\tdone in %.4f s' % (time.time() - lt, ))
                    continue # the first index is just for loading the hdf5

                if dsstep > 1:
                    pad = (dsstep - np.array(img.shape) % dsstep) % dsstep
                    oimg = measure.block_reduce(np.pad(img, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                            block_size=(dsstep, dsstep), func=self.blkrdc_func).astype(img.dtype)
                else:
                    oimg = img

                if tissue_masks:
                    bw = self.tissue_mask_bw[x]
                    if dsstep > rel_ds:
                        assert(dsstep % rel_ds == 0)
                        ds = dsstep // rel_ds
                        pad = (ds - np.array(bw.shape) % ds) % ds
                        bw = np.round(measure.block_reduce(np.pad(bw, ((0,pad[0]), (0,pad[1])), mode='reflect'),
                                block_size=(ds, ds), func=self.blkrdc_func)).astype(img.dtype)
                    elif dsstep < rel_ds:
                        assert(rel_ds % dsstep == 0)
                        ds = rel_ds // dsstep
                        bw = block_construct(bw, ds)
                    # paddings at different downsamplings can results in different sizes.
                    # so, crop to the downsampled output image size.
                    bw = bw[:oimg.shape[0], :oimg.shape[1]]
                    oimg = bw

                if do_overlays:
                    # dot legend (update this if changed):
                    #   blue dot == image center
                    #   red dot == roi polygonal center
                    #   green dot == roi bounding box center
                    # NOTE: since the addition of always cropping to the rough bounding box, the image center
                    #   is always the same as the center of the rough bounding box.

                    # gives nice overview image of whole slice overlaid with centers and grid bounding box.
                    oimg = cv2.cvtColor(oimg, cv2.COLOR_GRAY2RGB)
                    # grid center is not interesting, replace with roi points bounding box center (green dot)
                    grid_center = (roi_points.max(0) + roi_points.min(0))/2
                    # show polygonal center for roi_points center (red dot)
                    roi_points_ctr = np.array(PolyCentroid(roi_points[:,0], roi_points[:,1]))

                    # opencv is very picky about datatypes, need ints
                    igrid_min = np.ceil(grid_min/dsstep).astype(np.int32)
                    igrid_max = igrid_min + np.floor(grid_max/dsstep-grid_min/dsstep).astype(np.int32)
                    igrid_center = np.round(grid_center/dsstep).astype(np.int32)
                    #iroi_points = np.round(roi_points/dsstep).astype(np.int32)
                    iroi_points = np.round(roi_points_scl/dsstep).astype(np.int32)
                    iroi_points_ctr = np.round(roi_points_ctr/dsstep).astype(np.int32)
                    iimg_ctr = np.round(img_ctr/dsstep).astype(np.int32)

                    # draw the overlays onto the output image
                    line_thickness = 11 #; circle_rad = 5
                    cv2.rectangle(oimg, tuple(igrid_min.tolist()), tuple(igrid_max.tolist()), (0,255,0),
                        line_thickness)
                    cv2.polylines(oimg, [iroi_points.reshape((-1,1,2))], True, (255,0,0), line_thickness)
                    cv2.circle(oimg, tuple(iimg_ctr.tolist()), 17, (0,0,255), -1)
                    cv2.circle(oimg, tuple(iroi_points_ctr.tolist()), 17, (255,0,0), -1)
                    cv2.circle(oimg, tuple(igrid_center.tolist()), 11, (0,255,0), -1)

                if not save_h5 and (self.single_block or convert_hdf5s):
                    _, ext = os.path.splitext(fn)
                    ext = '' if ext == '.tiff' else '.tiff'
                    tifffile.imwrite(fn + ext, oimg)
                else:
                    lock = not self.single_block
                    _, f1, f2 = big_img_save(h5fn, oimg, img_shape=img_shape, nblks=self.nblocks,
                            iblk=self.iblock, novlp_pix=self.block_overlap_pix, recreate=True, compression=True,
                            lock=lock, keep_locks=lock, wait=True, img_shape_blk=img_shape_blk, bcrop=bcrop)
                    if bgsel is not None:
                        big_img_save(h5fn, bgsel, img_shape=img_shape, nblks=self.nblocks, iblk=self.iblock,
                            novlp_pix=self.block_overlap_pix, dataset='background', recreate=True, compression=True,
                            img_shape_blk=img_shape_blk, bcrop=bcrop, f1=f1, f2=f2)
                    if self.first_block:
                        big_img_save(h5fn, roi_points, roi_points.shape, dataset='roi_points', recreate=True)
                    if lock: gpfs_file_unlock(f1,f2)

                if save_roi_points:
                    sfx, _ = os.path.splitext(suffix); sfx += '_region_coords.csv'
                    fn = os.path.join(outpath, self.wafer_strs[wafer_ind] + order_str + self.wafer_region_strs[x]+sfx)
                    np.savetxt(fn, roi_points/dsstep, delimiter=';')
            #for j in range(ndssteps):
        #for i,cnt in zip(range(start,stop), range(stop-start)):

        if self.wafer_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))
    #def export_regions(self,
