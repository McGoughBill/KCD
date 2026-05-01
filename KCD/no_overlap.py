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
os.environ['OV_DATA_BASE'] = "/path/to/kcd_data/data/seg_data/"
import shutil

source_dataset = 'kits23'
destination_dataset = 'kits_nooverlap_v2'

# define the folders to copy from and to
from_folder = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', source_dataset)
to_folder = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', destination_dataset)

if not os.path.exists(to_folder):
    os.makedirs(to_folder)
    os.mkdir(os.path.join(to_folder,'images'))
    os.mkdir(os.path.join(to_folder,'labels'))

# define the folders to exclude
exclude_folders = [os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'add_ncct','images'),
                     os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits_ncct','images'),
                     os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits_ncct','unseen')]
exclude_files = []

# get the list of files in the from_folder
from_files = []
for root, dirs, files in os.walk(from_folder):
    for file in files:
        if file != '.DS_Store':
            from_files.append(os.path.join(root,*dirs, file))

# get the list of files in the exclude_folders
exclude_files = []
for exclude_folder in exclude_folders:
    is_kits = 'kits' in exclude_folder
    for file in os.listdir(exclude_folder):
        print(file)
        if file != '.DS_Store':
            if is_kits:
                #convert from ncct filename convention to cect filename convention
                # ncct:KiTS-XXXXX.nii.gz, cect:case_XXXXX.nii.gz
                exclude_files.append(os.path.join(from_folder,'images','case_' + file[5:]))
                exclude_files.append(os.path.join(from_folder,'labels','case_' + file[5:]))
            else:
                #converrt from ncct filename convention to cect filename convention
                # ncct:RCC_XXX.nii.gz, cect:Rcc_XXX.nii.gz
                exclude_files.append(os.path.join(from_folder,'images','Rcc_' + file[4:]))
                exclude_files.append(os.path.join(from_folder,'labels','Rcc_' + file[4:]))

print(f'Excluding {len(exclude_files)} files')
# get the list of files to copy
to_files = []
for from_file in from_files:
    if from_file not in exclude_files:
        to_files.append(from_file)
    else:
        print(f'Excluding {from_file}')

print(f'Copying {len(to_files)} files from this many files: {len(from_files)}')

# copy the files to the to_folder
for to_file in to_files:
    dest = to_file.replace(source_dataset, destination_dataset)
    shutil.copy(to_file, dest)
