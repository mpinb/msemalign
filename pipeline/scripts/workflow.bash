#!/usr/bin/env bash

# Before jobs can be run:
# Create a new folder for def_common_params.py for this experiment.
# Add an alias that exports the folder to the PYTHONPATH.

echo workflow.bash
echo   '<dataset_name>-<cluster_name>'
echo   init downsample region0 region wafer-init rough-order rough order-iter fine

## parameters
args=("$@")

# Source the specified parameter file.
# Typically these will be called <dataset_name>-<cluster_name>.bash
source ${args[0]}

# Second parameter is which portion of the pipeline to generate swarm files for.
runtype=${args[1]}


if [ "${runtype}" == "init" ]; then

# This exports alignment manifest automatically if it is not present.
# There should not be any errors.
create_swarm --help

fi


if [ "${runtype}" == "downsample" ]; then

# run 4 (oversubscribe) cores per job (cpu limited, io limited).
# NOTE: there is no pipeline file for this step, use rolling_submit directly.
create_swarm --all-wafers --no-wafer-id-arg --all-slices --run-script run_downsample_wafer.py --beg-arg region-range --end-arg --id-str downsample --format-str '%a %b 4' --other-flags " --thumbnails-only --nworkers 1 3 1 1 "
echo "run 4 (oversubscribe) cores per job (cpu limited, io limited)."
echo "NOTE: there is no pipeline file for this step, use rolling_submit directly."

fi


if [ "${runtype}" == "region0" ]; then

# alignment first pass
# MSEM_FFT_TYPE - scipy_fft = 0; numpy_fft = 1; pyfftw_fft = 2; cupy_fft = 3; rcc_xcorr = 4
# MSEM_FFT_BACKEND - none = 0; mkl = 1; fftw = 2; cupy = 3
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str align-pass1 --set-env "MSEM_NUM_THREADS=2 MSEM_FFT_TYPE=0,3 MSEM_FFT_BACKEND=1" --other-flags " --run-type align --no-brightness-balancing --twopass_align_first_pass "
# alignment second pass
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str align --set-env "MSEM_NUM_THREADS=3" --other-flags " --run-type align --no-brightness-balancing --save-residuals "

# balance mean mfov first pass
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str balance-mean-mfov --set-env "MSEM_NUM_THREADS=3" --other-flags " --run-type balance-mean-mfov "
# balance mean mfov second pass
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str balance-mean-mfov-pass2 --set-env "MSEM_NUM_THREADS=3" --other-flags " --run-type balance-mean-mfov --re-brightness-balancing "

# brightness with rebalance
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str balance --set-env "MSEM_BLKRDC_TYPE=1 MSEM_NUM_THREADS=${rebal_nw}" --other-flags " --run-type balance --re-brightness-balancing --roi-polygon-scale 0. --dsstep 4 --nworkers ${rebal_nw}"

fi


if [[ "${runtype}" == region* ]]; then

# export init
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str export-init --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-type export --block-overlap-um 8 8 --nblocks -${reg_nblksx} -${reg_nblksy} ${noblend} "
# export
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg iblock iblock region_inds --iwafer-iter-arg 2 --id-str export --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-type export --block-overlap-um 8 8 --nblocks ${reg_nblksx} ${reg_nblksy} ${noblend} " --iterate-ranges 0 ${reg_nblksx} 0 ${reg_nblksy}

# downsampled slice histograms
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region-inds-rng --iterate-ranges-split 128 128 128 128 128 --id-str histos-ds16 --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-histos --roi-polygon-scale ${roi} --nworkers ${histos_nw} --dsstep 16 ${tm} "
# # without ranges (for reimages for example)
# create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str histos-ds16 --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-histos --roi-polygon-scale ${roi} --dsstep 16 ${tm} "

# slice brightness balance
create_swarm --run-script run_regions.py --beg-arg nworkers --id-str slice-balance --set-env "MSEM_NUM_THREADS=32" --other-flags " --all-wafers --run-type slice-balance --dsstep 16 " --iterate-ranges 32 33

# slice brightness adjust
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str slice-brightness-adjust --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-brightness-adjust "

# export tiffs
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str convert-h5-to-tiff --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-type convert-h5-to-tiff --dsexports ${dsthumbs} ${noblend} "
# export tiffs masks
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str convert-h5-to-tiff-masks --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-type convert-h5-to-tiff --dsexports ${dsthumbs} --tissue-masks "

fi


if [ "${runtype}" == "region" ]; then

# save masks
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region-inds-rng --iterate-ranges-split 128 128 128 128 128 --id-str save_masks --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-type save-masks --save-masks-in ${msk_dir} "

# slice histograms
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region-inds-rng --iterate-ranges-split 128 128 128 128 128 --id-str histos --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-histos --roi-polygon-scale ${roi} --nworkers ${histos_nw} ${tm} "
# # without ranges (for reimages for example)
# create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str histos --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-histos --roi-polygon-scale ${roi} 1 ${tm} "

# analyze histograms (for picking out a template)
create_swarm --run-script plot_regions.py --beg-arg wafer_ids --id-str histo-width --set-env "MSEM_NUM_THREADS=1" --other-flags " --run-type histo-width " --iterate-wafers

# slice contrast matching
create_swarm --all-wafers --all-slices --run-script run_regions.py --beg-arg region_inds --id-str slice-contrast-match --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type slice-contrast-match ${heur} "

fi


if [ "${runtype}" == "wafer-init" ]; then

# only needed once, or to clear temporary items from meta dill or
#   if you want to change the czi angle or something with reading the manifest angles.
for ((w=1;w<=nwafers;w++)); do
  run_wafer.py --run-type export_rough_dills --w ${w}
done
run_wafer.py --run-type update_meta_dill --w 1

fi


if [ "${runtype}" == "rough-order" ]; then

# compute percent matching SIFT features
# NOTE: nworkers controls the number of processes per gpu, MSEM_NUM_THREADS controls the number of ransac workers
#   nransac should ==  ncores / (ngpus * nw), but can oversubscribe
# NOTE: only ransac-repeats is parallelized across the ransac workers,
#   i.e. --ransac-repeats should be a multiple of nransac.
nw=5
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg iprocess --id-str matches --set-env "MSEM_NUM_THREADS=${nransac}" --other-flags " --matches-only --keypoints-run-str ${kpts} --matches-run-str ${kpts} --nprocesses ${matches_np} --nworkers ${nw} --matches-ransac --ransac-repeats $((2*nransac)) --ransac-max 1000 ${thumbs} " --iterate-ranges 0 ${matches_np}
# merge
create_swarm --run-script run_wafer_solver.py --beg-arg wafer_ids --id-str matches-merge --other-flags " --matches-only --keypoints-run-str ${kpts} --matches-run-str ${kpts} --nprocesses ${matches_np} --nworkers ${nw} --merge " --iterate-wafers

# rough order solver only, writes out initial solved order for each wafer
# NOTE: be careful, this overwrites the solved order file
topn=
#topn="--percent-matches-topn 24" # 6, 12, 24
#topn="--percent-matches-normalize"
#topn="--percent-matches-normalize --percent-matches-normalize-minmax"
#excl=
excl=--no-exclude-regions
create_swarm --run-script run_wafer_solver.py --beg-arg wafer_ids --id-str solve --other-flags " --keypoints-run-str ${kpts} --matches-run-str ${kpts} --write-order ${topn} ${excl} " --iterate-wafers

fi


if [ "${runtype}" == "rough" ]; then

# rough alignment affines, all skips
# NOTE: automatic cross wafer support in run_create_swarm only works when fully parallelized per slice (comparison)
#   decided it was easier to stick with creating them this way (usually skip count is relatively low anyways)
np=512
for ((s=0;s<=rough_maxskips;s++)); do
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg iprocess --id-str affines_skip${s} --set-env "MSEM_NUM_THREADS=4" --other-flags " --affine-rerun --run-str ${afftype} --keypoints-run-str ${kpts} --skip-slices ${s} --nprocesses ${np} --ransac-repeats 5 --ransac-max 400000 --nworkers 4 ${thumbs} " --iterate-ranges 0 ${np}
done
# concatenate swarms together
cat $(date '+%Y%m%d')-run_wafer_solver-affines_skip?.swarm > $(date '+%Y%m%d')-run_wafer_solver-affines_all_skips.swarm
rm $(date '+%Y%m%d')-run_wafer_solver-affines_skip?.swarm
# merge affines
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg skip-slices --id-str merge_affines_skips --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-str ${afftype} --keypoints-run-str ${kpts} --nprocesses ${np} --merge " --iterate-ranges 0 $((${rough_maxskips}+1))

# rough alignment affines, cross wafer
# NOTE: see note above regarding automatic cross wafer generation
for ((w=1;w<nwafers;w++)); do
create_swarm --run-script run_wafer_solver.py --beg-arg skip-slices --id-str affines_xwafer${w} --set-env "MSEM_NUM_THREADS=4" --other-flags " --wafer_ids ${w} $((w+1)) --affine-rerun --run-str ${afftype} --keypoints-run-str ${kpts} --ransac-repeats 5 --ransac-max 400000 --nworkers 8 ${thumbs} " --iterate-ranges 0 $((${rough_maxskips}+1))
done
# concatenate swarms together
cat $(date '+%Y%m%d')-run_wafer_solver-affines_xwafer?.swarm > $(date '+%Y%m%d')-run_wafer_solver-affines_xwafer.swarm
rm $(date '+%Y%m%d')-run_wafer_solver-affines_xwafer?.swarm

# rough alignment, verify no bad matches:
#grep 'bad matches$' <save-rough-seq_file>
mkdir -p order_seqs
create_swarm --run-script run_wafer_aggregator.py --beg-arg wafer_ids --id-str status --other-flags " --run-type rough_status --run-str-in ${afftype} --use-rough-skip 0 --save-rough-seq ../order_seqs/order_seqs-${solvestr}.txt " --iterate-wafers

# rough alignment aggregation
rough_run=${afftype}_${rough_id}
npr=64
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str rough --set-env "MSEM_NUM_THREADS=4 MSEM_UUID=$(uuidgen)" --other-flags " --all-wafers --run-type rough --run-str-in ${afftype} --run-str-out-rough ${rough_run} --max-skips 3 --nworkers 12 --nprocesses ${npr} " --iterate-ranges 0 ${npr}
# single process merge, should be sufficient (multiple process merge is possible)
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str rough-merge --set-env "MSEM_NUM_THREADS=8" --other-flags " --all-wafers --run-type rough --run-str-in ${afftype} --run-str-out-rough ${rough_run} --max-skips 3 --nprocesses ${npr} --merge " --iterate-ranges 0 1

# rough alignment export
# NOTE: rigid appended at end is applying rigid xform to the solved deltas (without is full affine)
#   recommend for using the input as rigid to also use rigid for the solved deltas
rough_run=${afftype}_${rough_id}_rigid
create_swarm --all-wafers --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --rough-run-str ${rough_run} --dsexports ${dsthumbs} "
# to create the rough hdf5 exports (faster turnaround with fine steps)
#create_swarm --all-wafers --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_h5_export --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --rough-run-str ${rough_run} --rough-hdf5 "
# to export the rough alignment as hdf5 in order to cube at 16 nm
create_swarm --all-wafers --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_h5 --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --rough-run-str ${rough_run} --dsexports 1 --export-h5 "

# ordered export but without rough alignment (microscope only)
# for working on order / hand editing masks
create_swarm --all-wafers --all-slices --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_order --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --no-rough-alignment --dsexports ${dsthumbs} ${noblend} "
# export masks
create_swarm --all-wafers --all-slices --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_order_masks --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --no-rough-alignment --dsexports ${dsthumbs} --tissue-masks "
# to save edited masks into regions (where to put this?)
create_swarm --all-wafers --all-slices --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_save_masks --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --no-rough-alignment --save-masks-in ${msk_dir} "

fi


if [[ "${runtype}" == rough* ]]; then

# export thumbnails
create_swarm --all-wafers --all-slices --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_thumbnails --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --no-rough-alignment --no-order ${thumbs} "
# export masks
create_swarm --all-wafers --all-slices --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str rough_export_thumb_masks --set-env "MSEM_NUM_THREADS=4" --other-flags " --run-type rough_export --no-rough-alignment --no-order ${thumbs} --tissue-masks "

# compute keypoints
# NOTE: number of processes and workers is controlled in def_common_params because of dynamic keypoint dill loading
#   nworkers controls number of threads in cv2 (used for SIFT features)
#IMPORTANT: iterate-ranges here must match the number of keypoint processes defined in def_common_params
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg iprocess --id-str keypoints --other-flags " --keypoints-only --keypoints-run-str ${kpts} ${tm} --nworkers ${kpts_nw} ${thumbs} " --iterate-ranges 0 32

# rough alignment affines, skip 0, lower ransac iterations just to verify no bad matches
# NOTE: nworkers controls the number of parallel ransac workers, repeats are run serially
#   so nworkers divides ransac-max (max trials) so that total ransac iterations
#   is still ransac-repeats * ransac-max (in the worst case without early stopping).
np=256
if [ "${runtype}" == "rough-order" ]; then
  excl=--no-exclude-regions
else
  excl=
fi
ovly=
#ovly=--save-keypoints-overlays
#ovly=--save-keypoints-matches
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg iprocess --id-str affines_skip0 --set-env "MSEM_NUM_THREADS=4" --other-flags " --affine-rerun --run-str ${afftype} --keypoints-run-str ${kpts} --skip-slices 0 ${excl} --nprocesses ${np} --ransac-repeats 5 --ransac-max 6000 --nworkers 4 ${ovly} ${thumbs} " --iterate-ranges 0 ${np}
# merge affines, 20 per node
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg skip-slices --id-str merge_affines_skip0 --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-str ${afftype} --keypoints-run-str ${kpts} ${excl} --nprocesses ${np} --merge " --iterate-ranges 0 1

# rough alignment, verify no bad matches:
#grep 'bad matches$' <save-rough-seq_file>
mkdir -p order_seqs
create_swarm --run-script run_wafer_aggregator.py --beg-arg wafer_ids --id-str status_skip0 --other-flags " --run-type rough_status --run-str-in ${afftype} --use-rough-skip 0 --save-rough-seq ../order_seqs/order_seqs-${solvestr}.txt " --iterate-wafers

fi


# xxx - this is kindof a kludgy solution for generating "multiple sequence takes"
if [ "${runtype}" == "order-iter" ]; then

mkdir -p order_seqs

np=256
#excl=
excl=--no-exclude-regions
ovly=

# normal order solving runs
topns=("" "--percent-matches-topn 6" "--percent-matches-topn 12" "--percent-matches-topn 24" "--percent-matches-normalize" "--percent-matches-normalize --percent-matches-normalize-minmax")
solvestrs=(all top6 top12 top24 norm normm)
#solvestrs=(all_ds8 top6_ds8 top12_ds8 top24_ds8 norm_ds8 normm_ds8)
niters=5

# order solving sensitivity test
#topns=("" "--random-exclude-perc 0.01" "--random-exclude-perc 0.02" "--random-exclude-perc 0.05" "--random-exclude-perc 0.1" "--random-exclude-perc 0.15" "--random-exclude-perc 0.2" "--random-exclude-perc 0.3" "--random-exclude-perc 0.4" "--random-exclude-perc 0.5")
#solvestrs=(zero one two five ten fifteen twenty thirty fourty fifty)
#topns=("--random-exclude-perc 0.6" "--random-exclude-perc 0.7" "--random-exclude-perc 0.8" "--random-exclude-perc 0.9" "--random-exclude-perc 0.95")
#solvestrs=(sixty seventy eighty ninety ninetyfive)
#topns=("--random-exclude-perc 0.8" "--random-exclude-perc 0.9")
#solvestrs=(eighty ninety)
#niters=10

for j in ${!topns[@]}; do
  topn=${topns[$j]}
  for ((i=1;i<=niters;i++)); do
  solvestr=${solvestrs[$j]}-${i}

create_swarm --run-script run_wafer_solver.py --beg-arg wafer_ids --id-str solve-${solvestr} --other-flags " --keypoints-run-str ${kpts} --matches-run-str ${kpts} --write-order ${topn} ${excl} " --iterate-wafers
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg iprocess --id-str affines-${solvestr} --set-env "MSEM_NUM_THREADS=4" --other-flags " --affine-rerun --run-str ${afftype} --keypoints-run-str ${kpts} --skip-slices 0 ${excl} --nprocesses ${np} --ransac-repeats 5 --ransac-max 6000 --nworkers 4 ${ovly} ${thumbs} " --iterate-ranges 0 ${np}
create_swarm --all-wafers --run-script run_wafer_solver.py --beg-arg skip-slices --id-str merge_affines-${solvestr} --set-env "MSEM_NUM_THREADS=2" --other-flags " --run-str ${afftype} --keypoints-run-str ${kpts} ${excl} --nprocesses ${np} --merge " --iterate-ranges 0 1
create_swarm --run-script run_wafer_aggregator.py --beg-arg wafer_ids --id-str status-${solvestr} --other-flags " --run-type rough_status --run-str-in ${afftype} --use-rough-skip 0 --save-rough-seq ../order_seqs/order_seqs-${solvestr}.txt " --iterate-wafers

  done
done

fi


if [ "${runtype}" == "fine" ]; then

rough_run=${afftype}_${rough_id}_rigid
fine_out=${fine_run}-l2_${fine_l2}

# fine alignment, pre-create dills, runs quickly
create_swarm --run-script run_wafer.py --beg-arg wafer_ids --id-str fine-init --other-flags " --solved-order-ind $((-fine_maxskips-1)) 1 --run-type fine --rough-run-str ${rough_run} --delta-run-str ${align_run} --skip-slices 0 --crops-um-ind 0 " --iterate-wafers

# fine alignment
waf_reg=--all-wafers
#waf_reg="--wafer-region-in tear_area_wafer-region-in.txt"
#rh5=
#rh5=--rough-hdf5
rh5=--use-coordinate-based-xforms
blur=
#blur=--fine-blur-only
for ((s=0;s<=fine_maxskips;s++)); do
# MSEM_FFT_TYPE - scipy_fft = 0; numpy_fft = 1; pyfftw_fft = 2; cupy_fft = 3; rcc_xcorr = 4
# MSEM_FFT_BACKEND - none = 0; mkl = 1; fftw = 2; cupy = 3
create_swarm ${waf_reg} --run-script run_wafer.py --base-zero --add-to-range-end $((-s-1)) --beg-arg iblock iblock solved-order-ind --iwafer-iter-arg 2 --cross-wafer-max $((-s-1)) --id-str fine-skip${s}-crop${ic} --set-env "MSEM_NUM_THREADS=2 MSEM_FFT_TYPE=0,4 MSEM_FFT_BACKEND=1 CUDA_VISIBLE_DEVICES=" --other-flags " --run-type fine --rough-run-str ${rough_run} --delta-run-str ${align_run} --skip-slices ${s} --crops-um-ind ${ic} ${blur} ${tm} ${rh5} --nblocks ${fine_nblksx} ${fine_nblksy} --block-overlap-um ${fine_bovlp} " --iterate-ranges 0 ${fine_nblksx} 0 ${fine_nblksy}
create_swarm ${waf_reg} --run-script run_wafer.py --base-zero --add-to-range-end $((-s-1)) --beg-arg iblock iblock solved-order-ind --iwafer-iter-arg 2 --cross-wafer-max $((-s-1)) --id-str fine-skip${s}-crop${ic}-invert --set-env "MSEM_NUM_THREADS=2 MSEM_FFT_TYPE=0,4 MSEM_FFT_BACKEND=1 CUDA_VISIBLE_DEVICES=" --other-flags " --run-type fine --rough-run-str ${rough_run} --delta-run-str ${align_run} --skip-slices ${s} --invert-order --crops-um-ind ${ic} ${blur} ${tm} ${rh5} --nblocks ${fine_nblksx} ${fine_nblksy} --block-overlap-um ${fine_bovlp} " --iterate-ranges 0 ${fine_nblksx} 0 ${fine_nblksy}
done
# concatenate swarms together
cat $(date '+%Y%m%d')-run_wafer-fine-skip?-crop${ic}*.swarm > $(date '+%Y%m%d')-run_wafer-fine-all.swarm
rm $(date '+%Y%m%d')-run_wafer-fine-skip?-crop${ic}*.swarm
# then insert rolling gpu indices, jobs packed per node needs to be a multiple of number of gpus
awk 'BEGIN{cnt=0; torepl="CUDA_VISIBLE_DEVICES=";} /CUDA_VISIBLE_DEVICES=/{repl="CUDA_VISIBLE_DEVICES="cnt; sub(torepl, repl, $0); cnt=(cnt+1)%'${ngpus}';} {print}' $(date '+%Y%m%d')-run_wafer-fine-all.swarm > $(date '+%Y%m%d')-run_wafer-fine-all-gpuind.swarm

# fine outliers
# init block dills
create_swarm --run-script run_wafer_aggregator.py --beg-arg iblock --id-str fine-outliers-init --set-env 'MSEM_NUM_THREADS=4' --other-flags " 0 --all-wafers --run-type fine_outliers --run-str-out-rough ${rough_run} --run-str-in ${align_run} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nblocks -${out_nblksx} -${out_nblksy} --block-overlap-um ${out_bovlp} --ransac-repeats 100 --ransac-max 50000 --nworkers 4 ${fine_rng} " --iterate-ranges 0 ${out_nblksx}
# outliers
np=${nslices}
create_swarm --run-script run_wafer_aggregator.py --beg-arg iblock iblock iprocess --id-str fine-outliers --set-env 'MSEM_NUM_THREADS=4' --other-flags " --all-wafers --run-type fine_outliers --run-str-out-rough ${rough_run} --run-str-in ${align_run} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${np} --nblocks ${out_nblksx} ${out_nblksy} --block-overlap-um ${out_bovlp} --ransac-repeats 100 --ransac-max 50000 --nworkers 4 ${fine_rng} " --iterate-ranges 0 ${out_nblksx} 0 ${out_nblksy} 0 ${np}
# merge
# NOTE: number of parallel procs for merge does not have to match fine outliers run
if [ ${out_nblksx} -eq 1 ] && [ ${out_nblksy} -eq 1 ]; then
  np=0
else
  np=$((nslices // 5))
fi
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-outliers-merge --set-env 'MSEM_NUM_THREADS=4' --other-flags " --all-wafers --run-type fine_outliers --run-str-out-rough ${rough_run} --run-str-in ${align_run} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${np} --nblocks ${out_nblksx} ${out_nblksy} --block-overlap-um ${out_bovlp} --merge ${fine_rng} " --iterate-ranges 0 ${np}

# fine interpolation
np=${nslices}
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-interp --set-env 'MSEM_NUM_THREADS=2' --other-flags " --all-wafers --run-type fine_interp --run-str-out-rough ${rough_run} --run-str-in ${align_run} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${np} ${fine_rng} " --iterate-ranges 0 ${np}

# fine affine filtering
np=${nslices}
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-filter --set-env 'MSEM_NUM_THREADS=2' --other-flags " --all-wafers --run-type fine_filter --run-str-out-rough ${rough_run} --run-str-in ${align_run} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${np} " --iterate-ranges 0 ${np}

# xxx - maybe remove l2 because settled on fine alignment using affine filter with l2 off?

# fine reslice
nw=1
np=1
bovlp="0. 0."
nblksx=1
nblksy=1
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-reslice-init --set-env 'MSEM_NUM_THREADS=8' --other-flags " --all-wafers --run-type fine_reslice --run-str-in ${align_run} --run-str-out-rough ${rough_run} --run-str-out-fine ${fine_out} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nworkers ${nw} --nprocesses ${np} --L2_norm ${fine_l2} --keep-xcorrs --block-overlap-um ${bovlp} --nblocks -${nblksx} -${nblksy} " --iterate-ranges 0 ${np}
# reslice
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-reslice --set-env 'MSEM_NUM_THREADS=8' --other-flags " --all-wafers --run-type fine_reslice --run-str-in ${align_run} --run-str-out-rough ${rough_run} --run-str-out-fine ${fine_out} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${np} --L2_norm ${fine_l2} --keep-xcorrs --block-overlap-um ${bovlp} --nblocks ${nblksx} ${nblksy} " --iterate-ranges 0 ${np}

# fine reconcile
uuid=$(uuidgen)
# need this option in combination with affine filtering
#f=
f=--filtered-fine-deltas
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-reconcile --set-env "MSEM_UUID=${uuid} MSEM_NUM_THREADS=$(($ncores / $frec_nw))" --other-flags " --all-wafers --run-type fine --run-str-out-rough ${rough_run} --run-str-in ${align_run} --run-str-out-fine ${fine_out} ${f} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nworkers ${frec_nw} --nprocesses ${frec_np} --L2_norm ${fine_l2} ${fine_rng} " --iterate-ranges 0 ${frec_np}
# merge, lightweight but can still take a few mins depending on number of grid points
# if --filtered-fine-deltas then filtering takes some time, parallelization is with nworkers.
#   currently no multi-node parallelization support, but runtime still reasonable without this.
create_swarm --run-script run_wafer_aggregator.py --beg-arg iprocess --id-str fine-reconcile_merge --set-env "MSEM_NUM_THREADS=4" --other-flags " --all-wafers --run-type fine --run-str-out-rough ${rough_run} --run-str-in ${align_run} --run-str-out-fine ${fine_out} ${f} --verbose --max-skips ${fine_maxskips} --max-crops ${maxcrops} --nprocesses ${frec_np} --merge --nworkers ${frec_nw} ${fine_rng} " --iterate-ranges 0 1

# fine export
#rh5=
#rh5=--rough-hdf5
rh5=--use-coordinate-based-xforms
ovlp="0.128 0.128" # 8 pixels at 16 nm, seems plenty safe, probably ok same for all datasets
#ovlp="0 0" # some edge effects, so do not do this
# init
create_swarm --all-wafers --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str fine_export_init --set-env 'MSEM_NUM_THREADS=4' --other-flags " --run-type fine_export --rough-run-str ${rough_run} --fine-run-str ${fine_out} ${rh5} --nblocks -${efine_nblksx} -${efine_nblksy} --block-overlap-um ${ovlp} --crop-to-grid --export-h5 --dsexports 1 "
# export
create_swarm --all-wafers --run-script run_wafer.py --beg-arg iblock iblock export-region-beg --end-arg None None export-region-end --iwafer-iter-arg 2 --id-str fine_export --set-env 'MSEM_NUM_THREADS=4' --other-flags " --run-type fine_export --rough-run-str ${rough_run} --fine-run-str ${fine_out} ${rh5} --nblocks ${efine_nblksx} ${efine_nblksy} --block-overlap-um ${ovlp} --block-overlap-grid-um ${efine_govlp} --crop-to-grid --export-h5 --dsexports 1 " --iterate-ranges 0 ${efine_nblksx} 0 ${efine_nblksy}

# export downsampled tiffs
create_swarm --all-wafers --run-script run_wafer.py --beg-arg export-region-beg --end-arg export-region-end --id-str fine_export_tiffs --set-env 'MSEM_NUM_THREADS=2' --other-flags " --run-type fine_export --rough-run-str ${rough_run} --fine-run-str ${fine_out} --dsexports ${dsthumbs} --convert-h5-to-tiff "

fi
