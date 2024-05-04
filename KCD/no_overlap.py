# want to copy all images, labels, and fingerprints from all_add to a new folder,
# add_nooverlap, that do not exist in masked_test_set and masked_coreg_ncct.
#all folders have the following file structures:
# all_add
#     - images
#         - <filename>.npy
#     - labels
#         - <filename>.npy
#     - fingerprints
#         - <filename>.npy

import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
import shutil

# define the folders to copy from and to
from_folder = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'all_add','1mm_canceronly_customPP')
to_folder = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'add_nooverlap','1mm_canceronly_customPP')

if not os.path.exists(to_folder):
    os.makedirs(to_folder)
    os.makedirs(os.path.join(to_folder, 'images'))
    os.makedirs(os.path.join(to_folder, 'labels'))
    os.makedirs(os.path.join(to_folder, 'fingerprints'))

# define the folders to exclude
exclude_folders = [os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'masked_test_set','2,2,2mm'),
                     os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'masked_coreg_ncct','2mm_binary_canceronly')]
exclude_files = []

# get the list of files in the from_folder
from_files = []
for root, dirs, files in os.walk(from_folder):
    for file in files:
        if file != '.DS_Store':
            from_files.append(os.path.join(root, file))

# get the list of files in the exclude_folders
exclude_files = []
for exclude_folder in exclude_folders:
    for root, dirs, files in os.walk(exclude_folder):
        for file in files:
            if file != '.DS_Store':
                exclude_files.append(file.lower())

# get the list of files to copy
to_files = []
for from_file in from_files:
    if os.path.split(from_file)[1].lower() not in exclude_files:
        to_files.append(from_file)
    else:
        print(f'Excluding {from_file}')

# copy the files to the to_folder
for to_file in to_files:
    dest = to_file.replace('all_add', 'add_nooverlap')
    shutil.copy(to_file, dest)
