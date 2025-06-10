import os
import torch
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import numpy as np
from random import random

# when creating an instance of this class, you will then want to wrap it in a DataLoader
#i.e.
# dataset = image_dataset(path)
# dataloader = DataLoader(dataset, batch_size=n, shuffle=False)

#then, you will run inference in a loop like this:
# for batch in dataloader:
#     patch, row, col = batch
#     # do something with patch
#     # do something with row and col



class image_dataset(Dataset):
    def __init__(self, path, device=None):
        assert(os.path.exists(path))
        # write code to list all patches here - create self.patch_list

        self.device = device


    def __len__(self):
        # this function should return the length of self.patch_list
        pass

    def __getitem__(self,idx:int):
        # write code that opens the idx-th patch in self.patch_list and returns it as a tensor
        #based on previous discussions, you will also want this function to return the index of the patch
        # in the row and column of the image from which it was extracted
        # send all tensor data to self.device

        # something like: return patch, row, col
        pass
