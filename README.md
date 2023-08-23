# msemalign

#### tools for alignment of serial section multi Scanning Electron Microscopy (ssmSEM) acquired datasets

For a high-level introduction, please refer to the manuscript:
xxx - add reference to the msemalign paper

## Installation / Dependencies

The msem package is implemented and actively developed using [scientific python](https://www.scipy.org/).

Refer to the [bash install script](python/install.sh) for installing package dependencies in a conda environment using either [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

## Contents / Directory Structure

- python

  - msem

    - msem

      The core msem package. Class hierarchy roughly parallels that of the Zeiss data:

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

## Tutorial

Under development.

## License

[GPL v3](https://choosealicense.com/licenses/gpl-3.0/)

## Active Contributors

All work is carried out in the [Computational Neuroethology](https://mpinb.mpg.de/en/research-groups/groups/computational-neuroethology/research-focus.html) department at [MPINB](https://mpinb.mpg.de/en/).
