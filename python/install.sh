# theoretically this can work with any conda/mamba base install by setting variables appropriately below.
# if you install a fresh base (minconda/mambaforge) and are using bash,
#   then this script should work without user interaction.
# if you like your existing clever environment, take this more as a requirement list than an install script.

# FIRST: install mambaforge or miniconda
# https://github.com/conda-forge/miniforge#mambaforge
# https://docs.conda.io/en/latest/miniconda.html
# NOTE: it seems that mambaforge and conda do not play nice with each other,
#   so if you want both installed, you will have to custom manage the init blocks in ~/.bashrc
# NOTE: installing mamba in a conda environment did also work when tested,
#   but the mamba documentation claims that this is unsupported.

# name of the environment to use.
# WARNING: existing environment with this name is automatically deleted below.
env_name=msem

# allow this script to use either conda or mamba
#conda=conda
conda=mamba

# location of the root dir for the conda/mamba install.
#conda_dir=${HOME}/miniconda3
conda_dir=${HOME}/mambaforge

# CAUTION: automatically deletes existing env with same name
${conda} env remove --name ${env_name}

# make an env specifically for the msem package.
# NOTE: had too many problems mixing conda base channel with condaforge,
#   so everything that conda installs is done with the conda-forge channel.
#   for mambaforge this is the default channel.
${conda} create -y --name ${env_name} --channel conda-forge python=3.9 matplotlib scikit-learn imageio scikit-image sympy hdf5plugin mkl blas=*=*mkl mkl-service mkl_fft mkl_random

# activate the new env, conda activate does not work within bash scripts:
#https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# CAUTION: this did not work in all user environments that were tested (why??).
source ${conda_dir}/bin/activate ${env_name}
python --version

# these are separated out of the initial conda create because
#   they frequently have dependency compatibility issues.
#mamba install -y -c conda-forge pyfftw
# :( https://github.com/conda-forge/pyfftw-feedstock/issues/51
mamba install -y -c conda-forge "pyfftw=0.13.0=py39h51d1ae8_0"
mamba install -y -c conda-forge faiss-gpu cudatoolkit=11.8
mamba install -y -c conda-forge cupy cudatoolkit=11.8
mamba install -y -c conda-forge scikit-learn-intelex
mamba install -y -c conda-forge nvtx # here because not sure about cudatoolkit dep?

# will get Qt errors without using opencv headless
pip install opencv-contrib-python-headless tifffile dill aicspylibczi==2.8.0 scikit-fmm gputil tqdm

python -m pip install git+ssh://git@github.com/mpinb/rcc-xcorr.git

# install msem and aicspylibczimsem as development
pip install --no-binary :all: -e ~/gits/emalign/python/msem
pip install --no-binary :all: -e ~/gits/emalign/python/aicspylibczimsem

# for reference
# CAUTION: updates base conda but in some instances this can update more than just conda.
#conda update -y -n base -c defaults conda
# CAUTION: clears conda caches / etc
#conda clean -y --all
