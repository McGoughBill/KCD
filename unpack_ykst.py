# this script unzips a very large file and then saves the contents
# this zip file should contain a set of directories, subdirectories, and files
# unpack all files and save them within this directory structure

import os
import zipfile
import sys

# define the path to the zip file
zip_file = '/bask/projects/p/phwq4930-renal-canc/YKST/wcm23-20240611_170702_.zip'

# define the path to the directory where the contents of the zip file will be saved
save_dir = '/bask/projects/p/phwq4930-renal-canc/YKST/unpacked/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# open the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    # extract all files
    zip_ref.extractall(save_dir)

