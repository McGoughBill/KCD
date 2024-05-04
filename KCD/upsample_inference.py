# take a directory full of 3d binary segmentations and downsample them to 1/2 the resolution using
# pytorch's nn.functional.interpolate trilinear interpolation, and then convert to integer numpy arrays

import os
import numpy as np
import torch
import torch.nn.functional as F
os.environ['OV_DATA_BASE'] ='/bask/projects/p/phwq4930-renal-canc/data/seg_data'


def upsample_segmentations(src_dir, dest_dir,flipped=False):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in os.listdir(src_dir):
        if file.endswith('.npy'):
            print(f'Downsampling {file}')
            seg = np.load(os.path.join(src_dir, file))
            seg = torch.squeeze(torch.tensor(seg)).unsqueeze(0).unsqueeze(0).float()
            seg = F.interpolate(seg, scale_factor=2, mode='trilinear')
            seg = seg.squeeze().squeeze().numpy().astype(np.uint16)
            if flipped:
                seg = np.rot90(np.flip(seg,axis=-1),axes=(0,1),k=3)
            np.save(os.path.join(dest_dir, file), seg)

src_dir = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_npy/tcia_cect/[4 4 4]mm')
dest_dir2 = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_npy/tcia_cect/[4 4 4]mm_to2mm_flipped')
upsample_segmentations(src_dir, dest_dir2, flipped=True)