#!/usr/bin/env bash

# number of gpus per node
ngpus=4

# whether gpu jobs are split per node so that only single gpu jobs are packed.
# leave undefined for old behavior that uses CUDA_VISIBLE_DEVICES to control gpu job assignment.
# xxx - currently this only applies to xcorrs for the fine alignment, not for the 2D alignment
split_gpu_packing=1

# total number of wafers in the dataset
nwafers=2

# total sections before excludes:
# 4059 3997 sections
# 1 0 excluded sections == 4058 3997 sections processed
nslices=8055

# number of workers per process for region tile brightness rebalancing
# xxx - would need this to be dynamically configured based on the partition being submitted to
rebal_nw=9 # ok for CPU and CPU-72c

# number of blocks for region export
reg_nblksx=4
reg_nblksy=4

# location of the mask tiffs (optional)
#msk_dir=
#msk_dir=/gpfs/soma_fs/cne/watkins/masks-tiffs/Zebrafish_Retina_Zf2R_t-regions-i1-WATKINS-masks/Zebrafish_Retina_Zf2R_t-regions-i1-WATKINS-masks-corrected
#msk_dir=/gpfs/soma_fs/cne/watkins/masks-tiffs/Zebrafish_Retina_Zf2R_t-regions-i2-WATKINS-masks/segformer-b0-finetuned-Zebrafish_Retina_Zf2R_t-regions-i1-v3/corrected
msk_dir=/gpfs/soma_fs/cne/watkins/masks-tiffs/Zebrafish_Retina_Zf2R_t-regions-i3-WATKINS-masks/segformer-b0-finetuned-Zebrafish_Retina_Zf2R_t-regions-i1-v3/corrected

# how much to downsample the tissue masks on export
dsmasks=8

# how much to downsample tiff exports
dsthumbs=8
#dsthumbs=16 # for tear annotation warping (downsampled test)

# hook to optionally export with all the brightness / contrast balancing featues disabled
noblend=
#noblend=--no-blending-features
# hijack the param to export stage coord stitching (only works for run_regions.py)
#noblend=--stage-coords
#noblend="--stage-coords --no-blending-features"

# whether to use the tissue masks or not, also for order solving
#tm= # 2D alignment iteration 1, order solving
tm=--tissue-masks # 2D alignment iteration 2, rough alignment iterations all and fine alignment iterations all

# define this to process histograms as ranges (and use histos_nw)
# leave undefined for processing reimages (unset is automatic for native)
#histos_ranges=1

# number of workers to use per node for computing region histograms
histos_nw=4

# roi polygon scaling for calculating the histograms.
# use smaller values if running without tissues masks, otherwise 0.9 or 1.
# does not matter if not doing section contrast or brightness balancing beore final region export.
# 2D alignment iterations 1,2 - zebrafish retina, do NOT use section brightness balancing
roi=1. # 2D alignment iteration 2 final region export with masks
#roi=0.8 # 2D alignment iterations 1,2

# whether for contrast matching to apply heuristics (typically if running without masks)
# does not matter if not doing section contrast or brightness balancing beore final region export.
heur= # 2D alignment iteration 2 final region export with masks
#heur=--slice-balance-heuristics # 2D alignment iterations 1,2

# remember that you added the ability to view individual histograms by specifying region_inds to plot_regions.
# then save the target histogram to the meta file with:
# with sliding histogram just for wafers (before order solving):
# wafer01_0299_S299R299_stitched.h5 298 82.0 [82] [192] [0.92754094]
# run_regions.py --run-type save-target-histo --wafer_ids 1 --region_inds 299 --save-target-histo-sliding --save-target-histo-sliding-wafers
# wafer02_3670_S3670R3670_stitched.h5 3669 115.0 [115] [186] [0.9623161]
# run_regions.py --run-type save-target-histo --wafer_ids 2 --region_inds 3670 --save-target-histo-sliding --save-target-histo-sliding-wafers
# with sliding histograms for final regions:
# wafer01_2955_S2955R2955_stitched.h5 91 55.0 [55] [184] [0.98907146]
# wafer01_3450_S3450R3450_stitched.h5 1489 63.0 [63] [185] [0.99564409]
# wafer01_3493_S3493R3493_stitched.h5 2989 72.0 [72] [185] [0.99448301]
# wafer01_3478_S3478R3478_stitched.h5 3905 81.0 [81] [185] [0.99440345]
# run_regions.py --run-type save-target-histo --wafer_ids 1 --region_inds 2955 3450 3493 3478 --save-target-histo-sliding
# wafer02_0426_S426R426_stitched.h5 472 103.0 [103] [183] [0.9996037]
# wafer02_0374_S374R374_stitched.h5 988 122.0 [122] [182] [0.99759748]
# wafer02_0303_S303R303_stitched.h5 1452 132.0 [132] [182] [0.99390973]
# wafer02_0272_S272R272_stitched.h5 2105 142.0 [142] [182] [0.98277296]
# wafer02_1229_S1229R1229_stitched.h5 2983 146.0 [146] [182] [0.94281103]
# wafer02_0725_S725R725_stitched.h5 3852 150.0 [150] [182] [0.90259759]
# run_regions.py --run-type save-target-histo --wafer_ids 2 --region_inds 426 374 303 272 1229 725 --save-target-histo-sliding

# for only processing special (reimaged, torn, etc) slices
reimg=
#reimg="--reimage-index 1"
#reimg="--torn-regions"

# define this to export in solved order without rough alignment without downsampling (16 nm)
export_no_rough_ds1=1

# block overlap for the tear stitching when corrections are applied
# with the coordinates method, this only needs to be a small multiple of tear_grid_density (3-4x)
tear_bovlp="2 2"

# directory for testing tear stitching on downsampled images
warp_dns="/gpfs/soma_fs/cne-mSEM/mSEM-proc/2025/Zebrafish_Retina_Zf2R_t/meta/region_exports/wafer01_stitched_ds16 /gpfs/soma_fs/cne/watkins/annotations/tears/Zebrafish_Retina_Zf2R_t/test"

# optional run string flag for wafer solver thumbnails
thumbs=
#thumbs="--thumbs-run-str new_ds8"

# run string for the keypoints / matches files.
# useful for running with different configs (with / without masks, different roi scalings, etc).
kpts=1p0
#kpts=1p0_masks

# number of workers for keypoints calculation
# xxx - would need this to be dynamically configured based on the partition being submitted to
kpts_nw=6 # CPU-72c

# IMPORTANT: this value has to match that defined in def_common_params.py
# xxx - automate grepping it out of there... meh
kpts_np=32

# number of parallel ransacs for calculating percent matching features
nransac=3

# number of processes for computing matches
#matches_np=64 # good for smaller slices or ds16
matches_np=256 # need this for ds8 for larger slices (> 1 mm2)

# number of skips (neighbors) to use for rough alignment
#rough_maxskips=0 # for final tweaks during order solving
#rough_maxskips=3 # rough alignment iteration 1, sift rigid
rough_maxskips=4 # rough alignment iteration 2, fine to rough

# identifiers for the rough alignment runs. used both for skip0 and full rough alignment runs.
# different options used during order solving
#afftype=rigid
#afftype=rigid_masks
#solvestr=solving_rigid
#solvestr=solving_rigid_masks
#rough_id=sift
# rough alignment iteration 1, sift nonuniform
#afftype=snonuniform
#solvestr=solved_sift
#rough_id=sift
# rough alignment iteration 2, fine to rough
afftype=nonuniform
solvestr=solved_fine
rough_id=fine

# allow for rough alignments to be "stacked" on each other
#prev_rough_runs= # rough alignment iteration 1, sift rigid
prev_rough_runs=snonuniform_sift_rigid # rough alignment iteration 2, fine to rough

# define this to the initial rough run for rerough mode with fine alignment computed on top of fine-to-rough.
# NOTE: important that this is ONLY set for the filtering when computing the fine alignment on top of
#   the fine-to-rough (which would be the first iteration of the fine alignment), i.e.,
#   do NOT define this for subsequent fine iterations.
prev_rerough=
#prev_rerough=${prev_rough_runs} # fine alignment iteration 1 ONLY

# block options for rough export
rough_nblksx=3
rough_nblksy=4

# allow for fine alignments to be "stacked" on each other (but always after all rough alignments)
#prev_fine_runs= # fine alignment iteration 1
#prev_fine_runs="fine_i1-l2_0." # fine alignment iteration 2
#prev_fine_runs="fine_i1-l2_0. fine_i2-l2_0." # fine alignment iteration 3
prev_fine_runs="fine_i1-l2_0. fine_i2-l2_0. fine_i3-l2_0." # ultrafine alignment

# number of skips (neighbors) to use for fine alignment
fine_maxskips=4

# fine alignment identifier string
#align_run=32p0-4p8-16p0-1_img-tpl-spc-i # fine-to-rough and fine alignment iteration 1
#align_run=24p0-4p8-12p0-2_img-tpl-spc-i # fine alignment iteration 2
#align_run=16p0-4p8-8p0-3_img-tpl-spc-i # fine alignment iteration 3
align_run=13p6-3p2-2p0-4_img-tpl-spc-i-uf # ultrafine alignment

# fine alignment aggregate identifier string
#fine_run=nonuniform # rough alignment iteration 2, fine-to-rough
#fine_run=fine_i1 # fine alignment iteration 1
#fine_run=fine_i2 # fine alignment iteration 2
#fine_run=fine_i3 # fine alignment iteration 3
fine_run=ufine_i4 # ultrafine alignment

# fine alignment crop size iteration (base zero)
ic=0

# fine alignment max crops (for steps following the alignment xcorrs) (base one)
maxcrops=1

# fine alignment number of workers per process, set depending on packing.
# fine alignment iterations all, ultrafine alignment
fnt=1

# block options for fine alignment
# fine alignment iteration 1
#fine_nblksx=2
#fine_nblksy=4
# fine alignment iterations 2-4, ultrafine alignment
fine_nblksx=4
fine_nblksy=7
#fine_bovlp="16.128 16.128" # fine alignment iteration 1
#fine_bovlp="12.128 12.128" # fine alignment iteration 2
#fine_bovlp="8.128 8.128" # fine alignment iteration 3
fine_bovlp="6.928 6.928" # ultrafine alignment
# IMPORTANT: the number of grid points included in overlap does make a difference.
#   for example, some pixels differences up to +/- 6 pixels, mostly along overlap were detected at 25 vs 37 um
# NOTE: keep this at the overlap needed for the largest grid spacing when doing iterative fine alignments.
#   Thus, same value is also used for the fine export.
# calculate as 2.5 x 2.5 of grid spacing (use sqrt(3)/2 for y)
#efine_govlp="60 52"
efine_govlp="40 34.7"

# to only fine aggregate a slice range
fine_rng=
#fine_rng="--order-range 2438 2532"

# used by most portions of the fine alignment after the xcorrs (outliers, interp, filter...)
#   to decide how many sections to compute in a single process.
# xxx - for longer outlier runtimes, want this smaller (approaching 1) for better parallelization.
#   BUT, for shorter jobs having this small can produce problems because slurm does not like
#   lots of really short running jobs (i.e., less than a few minutes).
#nsections_per_proc=10 # fine alignment iterations 1-2
#nsections_per_proc=5 # fine alignment iteration 3, ultrafine alignment outliers
nsections_per_proc=1 # ultrafine alignment interp

# block options for fine outliers
# NOTE: recommend 3.5x grid spacing for block overlap (not used for single block)
# fine alignment iteration 1
#out_nblksx=2
#out_nblksy=2
#out_bovlp="56 56"
# fine alignment iteration 2
#out_nblksx=2
#out_nblksy=4
#out_bovlp="42 42"
# fine alignment iteration 3
#out_nblksx=3
#out_nblksy=6
#out_bovlp="28 28"
# ultrafine alignment
out_nblksx=11
out_nblksy=19
out_bovlp="7. 6.06"

# max ransac iterations for fine outliers
#fine_ransac_max=50000 # fine alignment iterations all
# up to 25k still reasonable for smaller datasets, use 15k for bigger datasets.
fine_ransac_max=15000 # ultrafine alignment

# ultrafine interpolation blocks and nworkers (for the MLS neighborhood method)
# should not need any block overlap for the MLS interpolation
# fine alignment iterations all
#fint_nblksx=1
#fint_nblksy=1
#fint_nt=2
#fint_nw=1
# ultrafine alignment
fint_nblksx=1
fint_nblksy=1
fint_nt=2
fint_nw=2

# block options for reslice, also used for reconcile.
# should not use these except for very large alignment grids (typically ultrafine).
# NOTE: for the parallelization options for ultrafine reslice
#   fslc_nw should not be larger than fslc_nblksx * fslc_nblksy
# fine alignment iterations all
#fslc_nblksx=1
#fslc_nblksy=1
#fslc_bovlp="0. 0."
#fslc_nw=1
#fslc_np=1
# ultrafine alignment
fslc_nblksx=25
fslc_nblksy=25
# NOTE: recommend 4.5x grid spacing for block overlap (not used for single block).
#   and use sqrt(3)/2 to compute y overlap.
fslc_bovlp="9. 7.8"
fslc_nw=16
fslc_np=45

# other options for reslice.
# whether to save the xcorrs in the reslice or not.
#fslc_kx=--keep-xcorrs # fine alignment iterations all
fslc_kx=  # ultrafine alignment
# whether to zero out deltas at the transitions that were specified to be blurred (for large gaps).
#fslc_zb=  # fine alignment iterations all
fslc_zb=--zero-blur # ultrafine alignment

# specify to use the affine-filtered deltas during the fine reconcile
#use_filt=--filtered-fine-deltas # fine alignment iterations all
use_filt=   # ultrafine alignment

# fine reconcile nworkers and nprocesses
frec_nw=12
frec_tpw=6
#frec_np=256 # fine alignment iterations all
frec_np=1 # ultrafine alignment

# fine reconcile l2, typically 0 (off) along with affine filtering
#fine_l2=0. # fine alignment iterations all
fine_l2=0.05 # ultrafine alignment

# block options for fine export
efine_nblksx=4
efine_nblksy=5

# for native regions, whether to copy masks from 16nm hdf5 files, or re-save them from tiffs
mcopy=
#mcopy="--copy-masks"
