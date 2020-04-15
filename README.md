# MVM

Add upstream repo
```
git remote add upstream https://github.com/vippolit/MVM
```

Verify
```
git remote -v
```


Syncing
```
git fetch upstream
git checkout master
git merge upstream/master
```

Create the conda environment:
```
  conda config --add channels conda-forge
  conda create --name mvm python=3.8
  conda activate mvm
  conda install mkl jupyter numpy scipy matplotlib scikit-learn h5py pandas pytables google-auth-oauthlib google-api-python-client lmfit
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
