import glob, os, shutil
import argparse

parser = argparse.ArgumentParser(prog='copy_sim_files')
parser.add_argument('src', help="origin")
parser.add_argument('dest', help="destination")
args = parser.parse_args()


files = glob.iglob(os.path.join(args.src, "*.rwa"))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, args.dest)

files = glob.iglob(os.path.join(args.src, "*.dta"))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, args.dest)
