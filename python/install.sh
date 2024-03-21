# theoretically this can work with any conda/mamba base install by setting variables appropriately below.
# if you install a fresh base (minconda/miniforge3) and are using bash,
#   then this script should work without user interaction.
# if you like your existing clever environment, take this more as a requirement list than an install script.

# FIRST: install miniforge (mambaforge deprecated) or miniconda
# https://github.com/conda-forge/miniforge?tab=readme-ov-file#download
# https://docs.conda.io/en/latest/miniconda.html
# NOTE: it seems that miniforge3/mambaforge and conda do not play nice with each other,
#   so if you want both installed, you will have to custom manage the init blocks in ~/.bashrc
# NOTE: installing mamba in a conda environment did also work when tested,
#   but the mamba documentation claims that this is unsupported.
# NOTE: current best solution is using miniconda but with the libmamba solver:
#   https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

# python 3.11 does not work with aicspylibczi because the PyFrames object changed and pybind11 uses it.
#   need <= 3.10 in order to build it, and the pip install for 2.8.0 is only available up to python 3.9.
# there are newer versions of aicspylibczi, but they made breaking changes starting with 3.0.0, and
#   aicspylibczimsem was not updated to work with the breaking changes.
# the best strategy is probably to go back to our own minimalist version of pylibczi.
# aicspylibczi / aicspylibczimsem are only needed to support the legacy zen format.
#
# python 3.11 is also not compatible with the pinned version of pyfftw=0.13.0=py39h51d1ae8_0
#   something about building with pthreads broke in this version and as of 1.1.2024 it is not fixed (see below).
# with the new version that does not have this fixed, this error occurs when running normxcorr with pyfftw:
#   ValueError: threads > 1 requested, but pyFFTW was not built with multithreaded FFTW.
# pyfftw is only used as one option for normxcorrs in _template_match, that in all recent tests is slower
#   than the mkl-enabled ffts anyways.
#use_python_311=1 # comment to install the previous working env using python 3.9
if [[ -n "$use_python_311" ]]; then
pyver=3.11
use_pyfftw=pyfftw
use_aicspylibczi=
unset install_aicspylibczimsem
else
pyver=3.9
use_pyfftw="pyfftw=0.13.0=py39h51d1ae8_0"
use_aicspylibczi="aicspylibczi==2.8.0"
install_aicspylibczimsem=1
fi

#install_rcc_xcorr=1
if [[ -n "$install_rcc_xcorr" ]]; then
rcc_conda="nvtx"
rcc_pip="gputil tqdm"
else
rcc_conda=
rcc_pip=
fi

# name of the environment to use.
# WARNING: existing environment with this name is automatically deleted below.
env_name=msem
#env_name=msem-311

# allow this script to use either conda or mamba
conda=conda
#conda=mamba

# location of the root dir for the conda/mamba install.
conda_dir=${HOME}/miniconda3
#conda_dir=${HOME}/mambaforge
#conda_dir=${HOME}/miniforge3

# CAUTION: automatically deletes existing env with same name
${conda} env remove --name ${env_name}

# make an env specifically for the msem package.
# NOTE: suggest not mixing conda base channel with condaforge,
#   likely to experience dependency problems.
# NOTE: needed to add intel channel for base install so that mkl
#   and blas using mkl installs correctly and with more recent versions.
${conda} create -y --name ${env_name} -c intel -c conda-forge python=${pyver} matplotlib scikit-learn scikit-learn-intelex imageio scikit-image sympy hdf5plugin mkl blas mkl-service mkl_fft mkl_random

# activate the new env, conda activate does not work within bash scripts:
#https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# CAUTION: this did not work in all user environments that were tested (why??).
source ${conda_dir}/bin/activate ${env_name}
python --version

# these are separated out of the initial conda create because
#   they frequently have dependency compatibility issues.
# ongoing MP issue with pyfftw: https://github.com/conda-forge/pyfftw-feedstock/issues/51
${conda} install -y -c conda-forge ${use_pyfftw} cudatoolkit=11.8 faiss-gpu cupy ${rcc_conda}

# will get Qt errors without using opencv headless
pip install opencv-contrib-python-headless tifffile dill ${use_aicspylibczi} scikit-fmm ${rcc_pip}

if [[ -n "$install_rcc_xcorr" ]]; then
# clone and install using local pip install with requirements commented out.
# reinstalls the mkl which causes problems, also forces unnecessary numpy/scipy updates.
#python -m pip install git+ssh://git@github.com/mpinb/rcc-xcorr.git
pip install ~/gits/rcc-xcorr
fi

# install msem and aicspylibczimsem as development
pip install --no-binary :all: -e ~/gits/emalign/python/msem
if [[ -n "$install_aicspylibczimsem" ]]; then
pip install --no-binary :all: -e ~/gits/emalign/python/aicspylibczimsem
fi

# for reference
# CAUTION: updates base conda but in some instances this can update more than just conda.
#conda update -y -n base conda
# CAUTION: clears conda caches / etc
#conda clean -y --all
