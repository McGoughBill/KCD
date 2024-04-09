# job is to identify all small cancers in src_dir/labels and save them in dest_dir/labels,
# along with their corresponding images and fingerprints.

#small is defined as having a volume less than 4189 voxels (radius of 2cm sphere, 8mm^3 voxel size)
#src_dir has the following file structure:
# src_dir
#     - images
#         - <filename>.npy
#     - labels
#         - <filename>.npy
#     - fingerprints
#         - <filename>.npy

import os
os.environ['OV_DATA_BASE'] = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data"
import shutil
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

# define the folders to copy from and to
src_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/preprocessed/kits23_add_nooverlap/2mm_binary_canceronly'
dest_dir = '/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/preprocessed/kits23_add_nooverlap_small/2mm_binary_canceronly'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    os.makedirs(os.path.join(dest_dir, 'images'))
    os.makedirs(os.path.join(dest_dir, 'labels'))
    os.makedirs(os.path.join(dest_dir, 'fingerprints'))

# get the list of files in the src_dir
label_dir = os.path.join(src_dir, 'labels')
image_dir = os.path.join(src_dir, 'images')
fingerprint_dir = os.path.join(src_dir, 'fingerprints')

src_files = []
for file in os.listdir(label_dir):
    if file != '.DS_Store':
        src_files.append(file)

# get the list of small cancers
dest_files = []
for src_file in src_files:
    print(f'Processing {src_file}')
    label_data = np.load(os.path.join(label_dir, src_file))
    label_data = label_data.astype(np.uint8)
    label_data, num_features = label(label_data)
    max_label_size=0
    for i in range(1, num_features+1):
        if np.sum(label_data==i)>max_label_size:
            max_label_size=np.sum(label_data==i)

    if max_label_size<4189:
        dest_files.append(src_file)
        print(f'{src_file} is small, size of {max_label_size} voxels')

# copy the files to the dest_dir
for dest_file in dest_files:
    shutil.copy(os.path.join(label_dir, dest_file), os.path.join(dest_dir, 'labels', dest_file))
    shutil.copy(os.path.join(image_dir, dest_file), os.path.join(dest_dir, 'images', dest_file))
    shutil.copy(os.path.join(fingerprint_dir, dest_file), os.path.join(dest_dir, 'fingerprints', dest_file))