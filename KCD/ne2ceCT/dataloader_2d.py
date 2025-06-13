import os
import torch
from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':
    import os
    os.environ['OV_DATA_BASE']='/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data'
import torchvision
import pandas as pd
import numpy as np
from random import random
from math import ceil
import sys
# import regionprops and label from skimage.measure
from skimage.measure import regionprops
from scipy.ndimage import label


# want to select random 128x128x128 patches from the 3D images, padding if need be,
# and then apply random rotations, flips, and noise to the patches, cropping them back to 64x64x64
# before returning them
class CrossPhaseDataset(Dataset):
    def __init__(self, ne_path, ce_path,split_path,background_value=-2,background_threshold=0.05, device=None, patch_size=224,
                 blur_kernel=torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                 is_train=True,patches_per_case=10):
        assert (os.path.exists(ne_path)) and os.path.exists(ce_path)
        self.ne_path = os.path.join(ne_path,'images')
        self.ce_path = os.path.join(ce_path,'images')
        self.split_path = split_path
        if os.path.exists(self.split_path):
            if os.path.exists(os.path.join(self.split_path,'split_train.npy')) and os.path.exists(os.path.join(self.split_path,'split_test.npy')):
                self.split = [np.load(os.path.join(self.split_path,'split_train.npy'),allow_pickle=True), np.load(os.path.join(self.split_path,'split_test.npy'),allow_pickle=True)]
            else:
                self.split = None
        else:
            raise FileNotFoundError(f'{self.split_path} does not exist')
        self.device = device


        #save cases as a list of cases present in both ne_path and ce_path
        self.cases = [case for case in os.listdir(self.ne_path) if case in os.listdir(self.ce_path) and case.endswith('.npy')]
        self.cases = np.random.permutation(np.array(self.cases))

        assert (len(self.cases) > 0)
        self.background_value = background_value
        self.background_threshold = background_threshold
        self.patch_size = patch_size
        self.is_foldsplit = False
        self.is_train = is_train
        self.blur_kernel = blur_kernel
        self.patches_per_case = patches_per_case
        self.case_patch_count = {case: 0 for case in self.cases}
        self.case_coordinates = {}
        self.slices_per_patch = 5

        self.patch_per_val_case=10


    def apply_foldsplit(self, split_ratio=0.8, train_cases=None):
        if self.split is not None:
            self.train_cases = self.split[0]
            self.test_cases = self.split[1]
            self.is_foldsplit = True
            print('Using saved split')
        else:
            if type(train_cases) == type(None):
                self.train_cases = np.random.choice(self.cases, int(split_ratio * len(self.cases)), replace=False)
            else:
                self.train_cases = train_cases

            self.test_cases = self.cases[~np.isin(self.cases, self.train_cases)]
            self.is_foldsplit = True
            # save the split
            np.save(os.path.join(self.split_path, 'split_train.npy'), self.train_cases)
            np.save(os.path.join(self.split_path, 'split_test.npy'), self.test_cases)


    def __len__(self):
        assert (self.is_foldsplit)

        if self.is_train:
            return len(self.train_cases)*self.patches_per_case
        else:
            return len(self.test_cases) * self.patch_per_val_case


    def _add_noise(self, tensor1, tensor2, p=0.3, noise_strength=0.3):
        if random() > p: return tensor1, tensor2
        random_noise_stdev = random() * noise_strength
        noise = torch.randn(tensor1.shape) * random_noise_stdev
        return tensor1 + noise, tensor2

    def _rotate(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        rot_extent = np.random.randint(1, 4)
        return torch.rot90(tensor1, rot_extent, dims=[-2, -1]), torch.rot90(tensor2, rot_extent, dims=[-2, -1])

    def _flip(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        flip = int(np.random.choice([-3, -2, -1], 1, replace=False))
        return torch.flip(tensor1, dims=[flip]),torch.flip(tensor2, dims=[flip])

    def _blur(self, tensor1, tensor2, p=0.3):
        if random() > p: return tensor1, tensor2
        return self.blur_kernel(tensor1), tensor2

    def _contrast(self, tensor1, tensor2, p = 0.2):
        if random() > p: return tensor1, tensor2

        fac = np.random.uniform(*[0.9, 1.1])
        mean = tensor1.mean()
        mn = tensor1.min().item()
        mx = tensor1.max().item()
        tensor1 = (tensor1 - mean) * fac + mean

        return tensor1.clip(mn, mx), tensor2

    def reset_patch_count(self):
        self.case_patch_count = {case: 0 for case in self.cases}

    def _get_random_patch_indices(self, nc, ce,case):
        '''

        Parameters
        ----------
        image1, image2 : torch.Tensor
            3D image to take a random patch from
            image1 is the non-contrast enhanced image
            image2 is the contrast enhanced image
        possible_patches : list
            list of possible patches to take from the image
        Returns
        -------
        tuple
            tuple of the random patch and the indices of the patch
        '''
        # delete all training data where image1 and image2 are not the same size within 3 pixels,
        # or there is inconsistent slice interval (n=10)
        for dim in range(1,4):
            assert (nc.shape[dim] - ce.shape[dim])<4
        # get the shape of the image
        shape = [min(nc.shape[dim], ce.shape[dim]) for dim in range(4)]
        # pad the image to the size of the patch if the image is smaller than the patch - use torch padding
        padding = [0,0,0]
        for dim in range(1,4):
            if shape[dim] < self.patch_size:
                pad = self.patch_size - shape[dim]
                padding[dim-1] = pad

        nc = torch.nn.functional.pad(nc.unsqueeze(0), (0,padding[2], 0,padding[1], 0,padding[0]),'constant',-2)
        ce = torch.nn.functional.pad(ce.unsqueeze(0), ( 0,padding[2], 0,padding[1], 0,padding[0]),'constant',-2)

        # get the random index to take the patch from. if dimensions are smaller than the patch size, make the index 0
        zxy = [np.random.randint(0, shape[dim] - self.patch_size) if shape[dim] > self.patch_size else 0 for dim in range(1,4) ]
        count = 0
        # if zxy returns a patch with over background thresh% of the patch being background, try again
        while torch.sum(ce[0, zxy[0]:zxy[0]+self.patch_size, zxy[1]:zxy[1]+self.patch_size, zxy[2]:zxy[2]+self.patch_size]>(self.background_value+0.01)) < (self.patch_size**3)*self.background_threshold:
            zxy = [np.random.randint(0, shape[dim] - self.patch_size) if shape[dim] > self.patch_size else 0 for dim in range(1,4) ]
            count += 1
            if count >50000:
                print(f'count exceeded 50000 for case {case}')
                sys.stdout.flush()
                assert(1==2)


        return nc,ce,tuple(zxy)

    def __getitem__(self, idx: int):
        assert (self.is_foldsplit)
        if self.is_train:
            transforms = [self._blur, self._add_noise, self._rotate, self._rotate, self._rotate,
                          self._flip, self._flip,self._flip, self._contrast]
            possible_cases = [case for case in self.train_cases if self.case_patch_count[case] <= self.patches_per_case]
            case = possible_cases[idx % len(possible_cases)]
            self.case_patch_count[case] += 1
        else:
            transforms = None
            case = self.test_cases[idx % (len(self.test_cases)-1)]

        ne = torch.Tensor(np.load(os.path.join(self.ne_path, case),allow_pickle=True))
        ce = torch.Tensor(np.load(os.path.join(self.ce_path, case),allow_pickle=True))
        ne,ce,indices = self._get_random_patch_indices(ne,ce,case)

        ne_patch = ne[0,0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]
        ce_patch = ce[0,0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]

        if self.is_train:
            if transforms:
                # shuffles order of transforms every time
                np.random.shuffle(transforms)
                for transform in transforms:
                    ne_patch,ce_patch = transform(ne_patch,ce_patch)

        #extract slices (first dimension) from the patch that have non-background values
        fg_slices = torch.any(ne_patch > (self.background_value + 0.01), dim=(1, 2))

        # if no foreground slices, return empty tensor all -2
        if torch.sum(fg_slices) == 0:
            ne_patch = torch.full((self.patch_size, self.patch_size, self.patch_size), self.background_value).float()
            ce_patch = torch.full((self.patch_size, self.patch_size, self.patch_size), self.background_value).float()
            return ne_patch.unsqueeze(1), ce_patch.unsqueeze(1)

        if torch.sum(fg_slices) < self.slices_per_patch:
            # if less than slices_per_patch, repeat the slices to fill the patch
            ne_path = ne_patch[fg_slices]
            ce_patch = ce_patch[fg_slices]
            rand = np.random.randint(0, ne_path.shape[0], torch.sum(fg_slices).item())
            ne_patch = ne_patch[rand]
            ce_patch = ce_patch[rand]
        else:
            # if more than slices_per_patch, randomly select slices_per_patch slices
            rand = np.random.choice(np.where(fg_slices)[0], self.slices_per_patch, replace=False)
            ne_patch = ne_patch[rand]
            ce_patch = ce_patch[rand]

        return ne_patch.unsqueeze(1), ce_patch.unsqueeze(1)

def collate_fn(batch):
    ne, ce = zip(*batch)
    ne = torch.cat(ne, dim=0)
    ce = torch.cat(ce, dim=0)
    #shuffle order using indexing to prevent batch_norm issues
    shuffle_indices = torch.randperm(ne.shape[0])
    return ne[shuffle_indices], ce[shuffle_indices]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    dataset_ne_path = '/Users/mcgoug01/Downloads/assess_ch/nc'
    dataset_ce_path = '/Users/mcgoug01/Downloads/assess_ch/ce'
    split_path = '/Users/mcgoug01/Downloads/assess_ch'
    # test the dataset
    # ce_path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/coltea_art/coltea_art_2/images/'
    # ne_path = '/media/mcgoug01/Crucial X6/ovseg_test/preprocessed/coltea_nat/coltea_nat_2/images/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    dataset = CrossPhaseDataset(dataset_ne_path, dataset_ce_path, split_path,
                                device=device, is_train=True)

    dataset.apply_foldsplit()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True,collate_fn=collate_fn)
    for i, (ne, ce) in enumerate(dataloader):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(ne[0,0 ].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        ax[1].imshow(ce[0,0 ].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        plt.show(block=True)