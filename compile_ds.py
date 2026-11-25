#copy all files ending in .nii.gz from dir_a and dir_b to dir_c
import os
import shutil

# dir_a = '/path/to/kcd_data/raw_data/tcia_cect/images'
# dir_b = '/path/to/kcd_data/raw_data/ce/images'
# dir_c = '/path/to/kcd_data/raw_data/paired_combined_cect/images'

dir_a = '/path/to/kcd_data/raw_data/successful_tcia_ncct_reg/images'
dir_b = '/path/to/kcd_data/raw_data/successful_artist_nc_reg/images'
dir_c = '/path/to/kcd_data/raw_data/paired_combined_ncct/images'


if not os.path.exists(dir_c):
    os.makedirs(dir_c)

for file in os.listdir(dir_a):
    if file.endswith('.nii.gz'):
        shutil.copy(os.path.join(dir_a, file), dir_c)

for file in os.listdir(dir_b):
    if file.endswith('.nii.gz'):
        shutil.copy(os.path.join(dir_b, file), dir_c)