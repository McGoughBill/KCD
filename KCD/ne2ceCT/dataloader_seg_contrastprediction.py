import os
import torch
from torch.utils.data import Dataset, DataLoader
if __name__ == '__main__':
    import os
    os.environ['OV_DATA_BASE']='/path/to/kcd_data'
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
    def __init__(self, ne_path, ce_path,split_path,seg_path, device=None, patch_size=64,
                 blur_kernel=torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                 is_train=True,patches_per_case=10):
        assert (os.path.exists(ne_path)) and os.path.exists(ce_path)
        self.ne_path = os.path.join(ne_path,'images')
        self.ce_path = os.path.join(ce_path,'images')
        self.split_path = split_path
        self.seg_path = seg_path
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
        # make sure cases also exist in the segmentation path
        self.cases = [case for case in self.cases if case in os.listdir(self.seg_path) and case.endswith('.npy')]
        self.cases = np.random.permutation(np.array(self.cases))

        assert (len(self.cases) > 0)
        self.patch_size = patch_size
        self.is_foldsplit = False
        self.is_train = is_train
        self.blur_kernel = blur_kernel
        self.patches_per_case = patches_per_case
        self.case_selected_count = {}
        self.case_coordinates = {}
        for case in self.cases:
            seg = np.load(os.path.join(self.seg_path, case),allow_pickle=True)
            #if any of the dimensions of the segmentation are less than the patch size, pad the segmentation with zeros
            padding = [0,0,0]
            for dim in range(3):
                if seg.shape[dim] < patch_size:
                    pad = patch_size - seg.shape[dim]
                    padding[dim] = pad
            seg = torch.nn.functional.pad(torch.Tensor(seg).unsqueeze(0).unsqueeze(0), (0,padding[2], 0,padding[1], 0,padding[0]),'constant',0).numpy()[0,0]

            # extract from each labelled region the volume and bounding box
            label_im, num_labels = label(seg)
            regions = regionprops(label_im)
            if len(regions) == 0:
                # remove the case from the list of cases
                self.cases = self.cases[self.cases != case]
                continue
            self.case_selected_count[case] = [0,len(regions)]
            for index,region in enumerate(regions):
                key = case + '_'+str(index)
                bbox = region.bbox
                possible_patches = [bbox[0]-patch_size//2,bbox[1]-patch_size//2,bbox[2]-patch_size//2,
                                    bbox[3]+patch_size//2,bbox[4]+patch_size//2,bbox[5]+patch_size//2]

                # the possible patch bounding max should not exceed the image size, or be less than 0, and should be at least 1 greater than the min
                possible_patches = [max(0,min(possible_patches[i],seg.shape[i]-(self.patch_size+1))) for i in range(3)] + [max(1,min(seg.shape[i]-self.patch_size,possible_patches[i+3])) for i in range(3)]
                assert all(possible_patches[i] < possible_patches[i+3] for i in range(3))
                assert all(possible_patches[i] >= 0 for i in range(3))
                assert all(possible_patches[i] <= seg.shape[i]-self.patch_size for i in range(3))

                self.case_coordinates[key] = possible_patches

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


    def _add_noise(self, tensor1, tensor2,seg, p=0.3, noise_strength=0.3):
        if random() > p: return tensor1, tensor2,seg
        random_noise_stdev = random() * noise_strength
        noise = torch.randn(tensor1.shape) * random_noise_stdev
        return tensor1 + noise, tensor2,seg

    def _rotate(self, tensor1, tensor2,seg, p=0.3):
        if random() > p: return tensor1, tensor2,seg
        rot_extent = np.random.randint(1, 4)
        return torch.rot90(tensor1, rot_extent, dims=[-2, -1]), torch.rot90(tensor2, rot_extent, dims=[-2, -1]), torch.rot90(seg, rot_extent, dims=[-2, -1])

    def _flip(self, tensor1, tensor2, seg, p=0.3):
        if random() > p: return tensor1, tensor2, seg
        flip = int(np.random.choice([-3, -2, -1], 1, replace=False))
        return torch.flip(tensor1, dims=[flip]),torch.flip(tensor2, dims=[flip]), torch.flip(seg, dims=[flip])

    def _blur(self, tensor1, tensor2, seg, p=0.3):
        if random() > p: return tensor1, tensor2, seg
        return self.blur_kernel(tensor1), tensor2, seg

    def _contrast(self, tensor1, tensor2, seg, p = 0.2):
        if random() > p: return tensor1, tensor2, seg

        fac = np.random.uniform(*[0.9, 1.1])
        mean = tensor1.mean()
        mn = tensor1.min().item()
        mx = tensor1.max().item()
        tensor1 = (tensor1 - mean) * fac + mean

        return tensor1.clip(mn, mx), tensor2, seg

    def reset_patch_count(self):
        self.case_patch_count = {case: 0 for case in self.cases}

    def _get_random_patch_indices(self, image1, image2, seg, possible_patches):
        '''

        Parameters
        ----------
        image1, image2 : torch.Tensor
            3D image to take a random patch from
            image1 is the non-contrast enhanced image
            image2 is the contrast enhanced image
        seg : torch.Tensor
            3D segmentation mask
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
            assert (image1.shape[dim] - image2.shape[dim])<4
        # get the shape of the image
        shape = [min(image1.shape[dim], image2.shape[dim]) for dim in range(4)]
        # pad the image to the size of the patch if the image is smaller than the patch - use torch padding
        padding = [0,0,0]
        for dim in range(1,4):
            if shape[dim] < self.patch_size:
                pad = self.patch_size - shape[dim]
                padding[dim-1] = pad


        image1 = torch.nn.functional.pad(image1.unsqueeze(0), (0,padding[2], 0,padding[1], 0,padding[0]),'constant',-2)
        image2 = torch.nn.functional.pad(image2.unsqueeze(0), ( 0,padding[2], 0,padding[1], 0,padding[0]),'constant',-2)
        seg = torch.nn.functional.pad(seg.unsqueeze(0).unsqueeze(0), (0,padding[2], 0,padding[1], 0,padding[0]),'constant',0)

        zxy = [np.random.randint(possible_patches[i], possible_patches[i + 3]) for i in range(3)]
        return image1,image2,seg,tuple(zxy)

    def __getitem__(self, idx: int):
        assert (self.is_foldsplit)
        if self.is_train:
            transforms = [self._blur, self._add_noise, self._rotate, self._rotate, self._rotate,
                          self._flip, self._flip,self._flip, self._contrast]
            # possible cases are those in the training cases that have not yet had 10 patches taken
            possible_cases = [case for case in self.train_cases if self.case_selected_count[case][0] <= self.patches_per_case]
            case = possible_cases[idx % len(possible_cases)]
            self.case_selected_count[case][0] += 1
        else:
            transforms = None
            case = self.test_cases[idx % (len(self.test_cases)-1)]

        _,num_regions = self.case_selected_count[case]
        if num_regions == 0:
            sys.stdout.write(f'num regions: {num_regions}\n')
            sys.stdout.write(f'case: {case}\n')
            sys.stdout.flush()
        #choose a random region from the case
        region = np.random.randint(0,num_regions)
        case_key = case + '_'+str(region)
        possible_patches = self.case_coordinates[case_key]

        ne = torch.Tensor(np.load(os.path.join(self.ne_path, case),allow_pickle=True))
        ce = torch.Tensor(np.load(os.path.join(self.ce_path, case),allow_pickle=True))
        seg = torch.Tensor(np.load(os.path.join(self.seg_path, case),allow_pickle=True))

        # if ne.shape != ce.shape:
        #     sys.stdout.write(f'ne shape: {ne.shape}, ce shape: {ce.shape}\n')
        #     sys.stdout.write(f'case: {case}\n')
        #     sys.stdout.flush()
        # get random patch indices

        ne,ce,seg,indices = self._get_random_patch_indices(ne,ce,seg, possible_patches)

        ne_patch = ne[0,0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]
        ce_patch = ce[0,0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]
        seg_patch = seg[0,0, indices[0]:indices[0]+self.patch_size, indices[1]:indices[1]+self.patch_size, indices[2]:indices[2]+self.patch_size]

        if self.is_train:
            if transforms:
                # shuffles order of transforms every time
                np.random.shuffle(transforms)
                for transform in transforms:
                    ne_patch,ce_patch,seg_patch = transform(ne_patch,ce_patch,seg_patch)

        contrast_enhancedment = ce_patch - ne_patch

        shapes = [ne_patch.shape == torch.Size([64,64,64]),ce_patch.shape == torch.Size([64,64,64]),
              seg_patch.shape == torch.Size([64,64,64]),contrast_enhancedment.shape == torch.Size([64,64,64])]

        if not all(shapes):
            # print actual shapes and case
            sys.stdout.write(f'ne shape: {ne_patch.shape}, ce shape: {ce_patch.shape}, seg shape: {seg_patch.shape}, diff shape: {contrast_enhancedment.shape}\n')
            sys.stdout.write(f'case: {case}\n')
            sys.stdout.flush()

        print('DL patch shapes equal to torch.Size([64,64,64])',ne_patch.shape == torch.Size([64,64,64]),ce_patch.shape == torch.Size([64,64,64]),
              seg_patch.shape == torch.Size([64,64,64]),contrast_enhancedment.shape == torch.Size([64,64,64]))
        sys.stdout.flush()

        return ne_patch.unsqueeze(0), ce_patch.unsqueeze(0), seg_patch.unsqueeze(0), contrast_enhancedment.unsqueeze(0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    dataset_ne_path = '/path/to/kcd_data/Downloads/assess_ch/nc'
    dataset_ce_path = '/path/to/kcd_data/Downloads/assess_ch/ce'
    split_path = '/path/to/kcd_data/Downloads/assess_ch'
    seg_path = '/path/to/kcd_data/Downloads/assess_ch/seg/images'
    # test the dataset
    # ce_path = '/path/to/kcd_data/Crucial X6/ovseg_test/preprocessed/coltea_art/coltea_art_2/images/'
    # ne_path = '/path/to/kcd_data/Crucial X6/ovseg_test/preprocessed/coltea_nat/coltea_nat_2/images/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    dataset = CrossPhaseDataset(dataset_ne_path, dataset_ce_path, split_path, seg_path,
                                device=device, is_train=True)

    dataset.apply_foldsplit()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (ne, ce,seg,diff) in enumerate(dataloader):
        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(ne[0, 0, 32, ].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        ax[0,1].imshow(ce[0, 0, 32, ].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        ax[1,0].imshow(seg[0, 0, 32,].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        ax[1,1].imshow(diff[0, 0, 32,].cpu().detach().numpy(), cmap='gray', vmin=-2, vmax=2)
        plt.show(block=True)