# msemalign

#### tools for alignment of serial section multi Scanning Electron Microscopy (ssmSEM) acquired datasets

For a high-level introduction, please refer to the following manuscripts: 

[msemalign: A pipeline for serial section multibeam scanning electron microscopy volume alignment](https://doi.org/10.3389/fnins.2023.1281098)  
&nbsp;&nbsp;&nbsp;&nbsp;Describes the 2D and 3D alignment of petabyte-scale ssmSEM datasets  
[GAUSS-EM: Guided accumulation of ultrathin serial sections with a static magnetic field for volume electron microscopy](https://doi.org/10.1101/2023.11.13.566828 )  
&nbsp;&nbsp;&nbsp;&nbsp;Describes the volume sectioning and also the section order solving methodology

## Installation / Dependencies

The msem package is implemented and actively developed using [scientific python](https://www.scipy.org/).

Refer to the [bash install script](python/install.sh) for installing package dependencies in a conda environment using either [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

## Tutorial

### Setup parameter files

Two new param files must be created that specify parameters for the alignment and for parallelization on a cluster. This tutorial assumes that you have a SLURM HPC cluster and that you will utilize [swarm-scripts](https://github.com/mpinb/swarm-scripts). If you intend to follow this tutorial exactly, first follow the installation instructions for `swarm-scripts`. The mSEM raw data must either be in the format as collected by Zeiss Zen or in the acquisition format developed at MPINB (not currently released).

The first param file is named `def_common_params.py` and contains all path specifications and alignment parameters.
An example param file is available for the [zebrafish retina](python/msem/params-datasets/ZF_retinaa_test/def_common_params.py) dataset. The path for the param file for the dataset being aligned needs to be added to the [PYTHONPATH](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH).

The second param file is typically named `<dataset_name>-<cluster_name>.bash` and is saved into [scripts](pipeline/scripts/). This file mostly contains parallelization / scaling parmeters. This folder must be added to the system `PATH`. This pipeline tutorial has only been tested with the [bash shell](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html). A sample param file is available for the [zebrafish retina](pipeline/scripts/retina_zebrafish_test-axon.bash) dataset.

### Initialization

At first invocation, routines defined by `def_common_params.py` need to initialize manifest files that contain a listing and some basic acquistion information for all sections (or slices) that comprise the dataset. This can be easily invoked with `workflow.bash` via the command, `workflow.bash <param-file>.bash init`, for example:

```
workflow.bash retina_zebrafish_test-axon.bash init
```

which should then show this as part of the output (for each wafer):

```
wafer 1 manifest not found, exporting
```

Any errors encountered during this step must be corrected so that the manifest is correctly generated before continuing.

For the remainder of the tutorial the `workflow.bash` commands will just be shown as `workflow.bash <param-file>.bash <command>`, and the approprite param file that specifies information regarding the dataset and the cluster on which the alignment being run should be selected appropriately.

Essentially all of the remaining pipeline steps involve generating `.swarm` files and then submitting them using `swarm-scripts`, typically using the top level `pipeline` workflow script. This is part of submission hierarchy then ultimately submits the jobs to a SLURM cluster via `sbatch` utilizing SLURM array jobs and contoling pipeline dependencies via SLURM job dependencies.

### Software downsampling

The majority of the pipeline operates on images downsampled to 16 nm. The original resolution for the dataset described in this tutorial is 4nm. To generate the `.swarm` file required for downsampling:

```
workflow.bash retina_zebrafish_test-axon.bash downsample
```

This will create a `.swarm` file named `<date>-run_downsample_wafer-downsample.swarm` that contains the downsampling command line for each section in the dataset.

The remainder of the pipeline uses `pipeline` files to specify how each swarm file is to be submitted and that specifieds step dependencies. The downsampling step, because it is also important for validating the raw data (for example detecting corrupted image tiles), is run without a pipeline file. Instead this step can be submitted using `rolling_submit`, for example:

```
rolling_submit --swarms <date-str>-run_downsample_wafer-downsample.swarm --swarm_opts " --partition p.axon,p.gpu -pt 10,5 "
```

When using `swarm-scripts`, the primary parameter, either `-p` or `-pt` specifies how many jobs can be packed into a single node. `-p` has swarm pack the jobs, so they appear as a single exclusive node jobs to SLURM, whereas `-pt` submits each line from the swarm file as a separate SLURM array job, but by requesting core and memory resources such that this number of jobs could run on the same node.

### Major workflow steps

Each major series of steps in the overall msemalign pipeline are submitted with dependencies based on workflow files read by the `pipeline` command, if using `swarm-scripts` for parallelization via an HPC SLURM cluster. `pipeline` workflow files are typically different depending on HPC cluster resources, henced named as `<pipeline_step_name>-<cluster_name>-pipeline.txt`, and can be copied to a working directory from [here](pipeline/scripts/pipeline).

The swarm files that are submitted via `pipeline` are generated using the `workflow.bash` script for all msemalign workflow steps.

#### 2D section alignment and initial region export

swarm files containing command lines for the initial section alignment can be generated with:

```
workflow.bash retina_zebrafish_test-axon.bash region0
```

and then submitted with:

```
pipeline --workflow-file region0-axon-pipeline.txt --no-pack
```

Alignment temporary files and exported files are typically written into a directory different from the raw data directory, for example, `mSEM-proc` but that preserves the same subdirectory structure as the raw data. Upon successfull completion of the initial region alignment steps, downsampled tiffs of each region are available under the specified `mSEM-proc` folder at `meta/region_exports`.

After this step, tissue areas within sections can be optionally annotated. This serves as training data for a model which then can geneate as inference the tissue masks for all sections.

#### msemalign pipeline workflow

Subsequent portions of the emalign workflow are also controlled via pipeline files. Typically steps complete sequentially such that each step does not start before the successful completion of all jobs in the previous step (pipeline files define a topologically ordered DAG of workflow steps). The swarm files for each pipeline step are generated via the `workflow.bash` script.

The major steps (parameters) to `workflow.bash` are avaible by invoking it with only the parameter argument:

```
workflow.bash retina_zebrafish_test-axon.bash
```

which produces the output:

```
workflow.bash
<dataset_name>-<cluster_name>
init downsample region0 region wafer-init tears rough-order rough order-iter fine ufine-init ultrafine
native-region native-tears native-fine native-ultrafine
```

The first three portions of the workflow have already been discussed. The remaining steps are:

- `region` generates swarm files that regenerates the regions, but includes contrast balancing between slices as well, typically run along with tissues masks which allows.
- `wafer-init` runs the initialization steps that are required for 3D alignment steps (starting with ordering solving and rough alignment).
- `rough-order` generates swarm files for computing the solved order for wafers that were cut without preserving the ordering of the sections.
- `rough` generates the swarm files for computing the rough alignment (single affine per section).
- `fine` generates the swarm files for the fine alignment.
- `ufine-init` runs the initialization steps for the 3D ultrafine alignment which is computed based on the exported fine alignment stack.
- `ultrafine` generates the swarm files for computing the ultrafine alignment.
- `native-region` generates the swarm files for exporting each section at original resolution.
- `native-fine` generates the swarm files for the exporting the fine alignment at original resolution.
- `native-ultrafine` generates the swarm files for exporting the ultrafine alignment at original resolution.

Each step that generates swarm files utilizes a pipeline file that starts with the same name. The second part of the pipeline file name is the name of the cluster that the jobs will be submitted to. This is necessary because the number of jobs packed per node can be different on different clusters.

For example, the rough alignment steps can be submitted with:

```
pipeline --workflow-file rough-axon-pipeline.txt --no-pack
```

## Top-level Scripts

Command lines for each step are single lines in each `.swarm` file that is generated by `workflow.bash`, so examples are readily available by viewing these files. Help for command lines arguments is available for all the top level scripts by invoking with `--help`. The following is a brief description of the major top level scripts that are invoked for different steps of the msemalign pipeline:

- `run_downsample_wafer.py` creates the downsampled thumbnails for mSEM data.
- `run_regions.py` is the top-level script controling the 2D section alignment and 2D/3D section brightness and contrast balancing.
- `run_wafer_solver.py` is the top-level script controling the rough alignment and section order solving.
- `run_wafer.py` is the top-level script controling the computing section cross-correlations and for rough/fine/ultrafine-aligned section exports.
- `run_wafer_aggregator.py` is the top-level script that solves the 3D rough/fine/ultrafine alignments based on matching features (rough) and cross-correlations (fine/ultrafine) measured between neighboring sections.

Other top-level scripts generate plots for validating / presenting different aspects of the msemalign pipeline:

- `plot_regions.py` generates plots involving the 2D section alignment.
- `plot_matches.py` generates plots involving the section order solving.
- `plot_aggregation.py` generates plots involving the solved rough/fine/ultrafine alignments.

## Contents / Directory Structure

- python

  - msem

    - msem

      The core msem package. Class hierarchy roughly parallels that of the mSEM data:

      images -> mFOVs -> slices -> wafers

    - params-datasets

      Each subdirectory here contains a def_common_params.py file which defines all the parameters used for a particular dataset. This allows top-level scripts to be shared across datasets. The environment variable PYTHONPATH can be modified to point to the dataset of choice dynamically.

    - scripts

      These are the top-level command line interfaces that execute the different steps in the alignment pipeline workflow. This directory should be added to the shell PATH.

  - aicspylibczimsem

    This is an extension of [aicspylibczi](https://github.com/AllenCellModeling/aicspylibczi). aicspylibczi is a python wrapper of the Zeiss-provided C++ library [libCZI](https://github.com/zeiss-microscopy/libCZI), both open source tools that expose a high level interface in python for reading Zeiss CZI files. aicspylibczimsem additionally parses the meta-data of mSEM-specific czi files.

- pipeline

  Contains top-level scripts for the generation of swarm files that are submitted to an HPC cluster. Also contains
  workflow files (pipeline) that control the submission hierarchies and dependencies.

## License

[GPL v3](https://choosealicense.com/licenses/gpl-3.0/)
