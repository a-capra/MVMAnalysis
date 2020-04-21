# MVM


## Getting started
Checkout the package via
```
  git clone https://github.com/vippolit/MVM.git
```

On a linux PC, run:
```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
```
while on a Mac:
```
  curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  bash Miniconda3-latest-MacOSX-x86_64.sh
```

Then, create the conda environment:
```
  conda config --add channels conda-forge
  conda create --name piton3 root=6 python=3 mkl jupyter numpy scipy matplotlib scikit-learn h5py pandas pytables root_pandas pytables google-auth-oauthlib google-api-python-client lmfit
```

and activate it:
```
  source activate piton3
```

In order to deactivate the environment and unsetup all packages (thus restoring your standard environment), simply do:

```
  source deactivate
```

If you use distributed computing resources, you may have access to CMVFS, where you can use
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh
pip install --user pytables
```

## To run
For example,
```
python combine.py ../Data -p --mvm-col='mvm_col_arduino' -d plotsdir_tmp
python latexify.py ../Plots/Run\ 9\ Apr\ 3\ 2020/*txt*pdf > ../Plots/Run\ 9\ Apr\ 3\ 2020/summary.tex
python get_tables.py plots_iso/*json --output-dir=plots_iso
```
## Repository structure

Folders:
  * `python`: python code
  * `cpp`: C++ code
  * `scripts`: scripts (plotting, etc)
