import nibabel as nib
import numpy as np
import os

# this scripts copies the npy files from directory A to directory B if it exists as a nifti file in directory C

# directory A
dir_A = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tcia_ncct_reg/2mm_canceronly_customPP/images'
dir_B = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tcia_ncct_reg/cmd_ready_2/images'
dir_C = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/raw_data/abdominal_tcia_cect/images'

# get all the files in directory A
files_A = [f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f)) and f.endswith('.npy')]
files_B = [f for f in os.listdir(dir_B) if os.path.isfile(os.path.join(dir_B, f)) and f.endswith('.npy')]
files_C = [f for f in os.listdir(dir_C) if os.path.isfile(os.path.join(dir_C, f)) and f.endswith('.nii.gz')]

# get the list of files that are in A and C
files_AC = []
for f in files_A:
    for g in files_C:
        if f[:-4] in g:
            files_AC.append(f)

# copy files to C
for f in files_AC:
    if f not in files_B:
        os.system(f'cp {os.path.join(dir_A, f)} {os.path.join(dir_B, f)}')
        print(f'copied {f} to {dir_B}')
    else:
        print(f'{f} already exists in {dir_B}')