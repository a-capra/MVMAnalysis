# MVM


## Getting started
Checkout the package via
```
  git clone https://github.com/MechanicalVentilatorMilano/MVMAnalysis.git
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
For example, the standard workflow is:
```
python combine.py ../Data -p --mvm-col='mvm_col_arduino' -d plotsdir_tmp
python latexify.py ../Plots/Run\ 9\ Apr\ 3\ 2020/*txt*pdf > ../Plots/Run\ 9\ Apr\ 3\ 2020/summary.tex
python get_tables.py plots_iso/*json --output-dir=plots_iso
```

A few more examples:

- to read data from the Run_15_Apr_12 campaign at Elemaster/Monza (using ```add_good_shift```, see below)
```
py combine.py path_to_the_Run_15_Apr_12_folder  -f  VENTILATOR_12042020_CONTROLLED_FR12_PEEP10_PINSP35_C20_R50_RATIO025_leak.txt -p -show
```

- to read data from the Napoli campaign (reading json DAQ from mvm-control)
```
py combine.py path_to_the_Napoli_folder -f run051_MVM_NA_O2_wSIM_Spain_C50R05.json --db-range-name "Napoli\!A2:AZ" -json -p -show
```
- older dataset with compact json format
```
python combine.py path_to_the_Napoli_folder  -f run021_MVM_NA_2001_O2_wSIM_C30R05.json  --db-range-name "Napoli\!A2:AZ"  -json  -p -show  -o 79.5  
```

The time synchronisation between datasets is a critical step to compare MVM and ASL datasets. By default, there is currently no time shift between the time axes. The ```-o``` option can be used manually set an offset (units of seconds) between the two datasets. Two more refined algorithms are implemented for automatic calculation of the shift: ```apply_rough_shift``` and ```apply_good_shift```. These functions need to be manually activated in ```combine.py```.

## Repository structure

Folders:
  * `python`: python code
  * `cpp`: C++ code
  * `scripts`: scripts (plotting, etc)
