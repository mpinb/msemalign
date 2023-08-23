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
# [735 955 884] = 2574 slices
# total number of slices in solved order (minus excludes)
nslices=2574

# number of workers per process for region tile brightness rebalancing
rebal_nw=10

# number of blocks for region export
reg_nblksx=2
reg_nblksy=2

# location of the mask tiffs (optional)
msk_dir=

# how much to downsampled tiff exports
dsthumbs=8

# whether to use the tissue masks or not
tm=
#tm=--tissue-masks

# number of workers to use per node for computing region histograms
histos_nw=4

# roi polygon scaling for calculating the histograms.
# use smaller values (0.5 for example) if running without tissues masks, otherwise 0.9 or 1.
roi=0.5

# whether for contrast matching to apply heuristics (typically if running without masks)
#heur=
heur=--slice-balance-heuristics

# remember that you added the ability to view individual histograms by specifying region_inds to plot_regions.
# then save the target histogram to the meta file with:
# run_regions.py --run-type save-target-histo --wafer_ids 2 --region_inds 1169

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
afftype=rigid
#afftype=nonuniform
#afftype=full
solvestr=solved_rigid
#solvestr=solved_nonuniform
#solvestr=solved_full
rough_id=globalz
