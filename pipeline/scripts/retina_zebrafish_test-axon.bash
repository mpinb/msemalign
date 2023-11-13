#!/usr/bin/env bash

# number of gpus per node
ngpus=1

# number of cores per node (for reconcile only?)
ncores=40

# total number of wafers in the dataset
nwafers=3

# wafers nimgs before exclude lists:
# [739 959 894] = 2592 slices
# Processed wafers nimgs:
# [737 955 881] = 2573 slices
# total number of slices in solved order (minus excludes)
nslices=2573

# number of workers per process for region tile brightness rebalancing
rebal_nw=10

# number of blocks for region export
reg_nblksx=4
reg_nblksy=4

# location of the mask tiffs (optional)
msk_dir=/axon/scratch/pwatkins/mSEM-proc/2019/briggman/2019-10-17-ZF_retinaa_test/meta/annotations/ZF_retinaa_test-regions-JELLI---ptl_v2__all_2023-04-14---v0_step43250---masks-edited

# how much to downsampled tiff exports
dsthumbs=8

# whether to use the tissue masks or not
#tm=
tm=--tissue-masks

# define this to process histograms as ranges (and use histos_nw)
# leave undefined for processing reimages (unset is automatic for native)
histos_ranges=1

# number of workers to use per node for computing region histograms
histos_nw=4

# roi polygon scaling for calculating the histograms.
# use smaller values (0.5 for example) if running without tissues masks, otherwise 0.9 or 1.
#roi=0.5
roi=1.0

# whether for contrast matching to apply heuristics (typically if running without masks)
heur=
#heur=--slice-balance-heuristics

# remember that you added the ability to view individual histograms by specifying region_inds to plot_regions.
# then save the target histogram to the meta file with:
# run_regions.py --run-type save-target-histo --wafer_ids 2 --region_inds 433

# define this to export in solved order without rough alignment without downsampling (16 nm)
#export_no_rough_ds1=1

# block overlap for the tear stitching when corrections are applied
tear_bovlp="32 32"

# optional run string flag for wafer solver thumbnails
thumbs=
#thumbs="--thumbs-run-str ds16"
#thumbs="--thumbs-run-str ds4"

# run string for the keypoints / matches files.
# useful for running with different configs (with / without masks, different roi scalings, etc).
#kpts=1p0_ds4
kpts=1p0

# number of workers for keypoints calculation
# xxx - would need this to be dynamically configured based on the partition being submitted to
kpts_nw=4 # cpu 4, gpu 2

# number of parallel ransacs for calculating percent matching features
nransac=5

# number of processes for computing matches
matches_np=64 # good for smaller slices or ds16
#matches_np=256 # need this for ds8 for larger slices (> 1 mm2)

# number of skips (neighbors) to use for rough alignment
#rough_maxskips=0 # for final tweaks during order solving
rough_maxskips=3

# identifiers for the rough alignment runs. used both for skip0 and full rough alignment runs.
#afftype=rigid
afftype=nonuniform
#afftype=full
#solvestr=solved_rigid
solvestr=solved_nonuniform
#solvestr=solved_full
rough_id=localz

# number of skips (neighbors) to use for fine alignment
fine_maxskips=3

# fine alignment identifier string
align_run=32p0-4p8-16p0_img-tpl-spc

# fine alignment aggregate identifier string
fine_run=fine-icrop0

# fine alignment crop size iteration (base zero)
ic=0

# fine alignment max crops (for steps following the alignment xcorrs) (base one)
maxcrops=1

# block options for fine alignment
fine_bovlp="16.128 16.128"
fine_nblksx=2
fine_nblksy=2

# to only fine aggregate a slice range
fine_rng=
#fine_rng="--order-range 2438 2532"

# block options for fine outliers
out_nblksx=1
out_nblksy=1
out_bovlp="84 84"

# fine reconcile nworkers and nprocesses
frec_nw=11
frec_tpw=4
frec_np=80

# fine reconcile l2, typically 0 (off) along with affine filtering
fine_l2=0.

# block options for fine export
efine_nblksx=4
efine_nblksy=4
# IMPORTANT: the number of grid points included in overlap does make a difference.
#   for example, some pixels differences up to +/- 6 pixels, mostly along overlap were detected at 25 vs 37 um
efine_govlp="65 65"

# number of skips (neighbors) to use for ultrafine alignment
ufine_maxskips=2

# ultrafine alignment identifier string
ufine_in=12p8-3p2-2p0_img-tpl-spc

# ultrafine alignment aggregate identifier string
ufine_out=ufine-l2_0.05

# ultrafine alignment crop size iteration (base zero)
uic=0

# ultrafine alignment max crops (for steps following the alignment xcorrs) (base one)
umaxcrops=1

# block options for ultrafine alignment
ufine_bovlp="0. 0."
ufine_nblksx=1
ufine_nblksy=1

# whether to use the tissue masks or not for the ultrafine alignment
utm=
#utm=--tissue-masks

# block options for ultrafine outliers
uout_nblksx=6
uout_nblksy=6
uout_bovlp="8.5 8.5"

# ultrafine interpolation blocks and nworkers (for the MLS neighborhood method)
# should not need any block overlap for the MLS interpolation
ufi_nblksx=1
ufi_nblksy=1
ufi_nw=4

# block options for ultrafine reslice, also used for ultrafine reconcile
ufre_nblksx=4
ufre_nblksy=4
ufre_bovlp="8.5 8.5"

# parallelization options for ultrafine reslice
# NOTE: ufre_nw should not be larger than ufre_nblksx * ufre_nblksy
ufre_nw=16
ufre_np=24

# ultrafine reconcile l2
ufine_l2=0.05

# ultrafine reconcile nworkers
ufrec_nw=11
ufrec_tpw=4

# block options for ultrafine export
eufine_nblksx=4
eufine_nblksy=4
# IMPORTANT: the number of grid points included in overlap does make a difference.
#   for example, some pixels differences up to +/- 6 pixels, mostly along overlap were detected at 25 vs 37 um
eufine_govlp="8.5 8.5"

# override stack location for the ultrafine native export
native_stack=/axon/scratch/pwatkins/mSEM-proc/2019/briggman/2019-10-17-ZF_retinaa_test/meta/fine_alignment_exports/fine-icrop0-l2_0._native
