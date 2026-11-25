from KCD.sLDCT_synthesis.jack_additive import calculate_noise
import os
import numpy as np
import nibabel as nib
import sys

def get_nii_image(image_path):
    '''
    :param image_path: str
    :return hu_array: arr
    '''
    image = nib.load(image_path).get_fdata()
    aff = nib.load(image_path).affine
    return image, aff

def simulate_noise(image, dose_fraction=0.75, exposure_time=0.5, current = 400):
    '''
    :param image_path: str
    :param dose_fraction: float
    :param current: float
    :return: numpy array_like
    '''
    noised_image = np.zeros_like(image)
    for i in range(image.shape[-1]):
            slice = image[:,:,i]
            slice = slice[:,: ,np.newaxis]
            noisy_slice = calculate_noise(slice, current= current*exposure_time, dose_fraction=dose_fraction)
            noised_image[:,:,i] = noisy_slice
    sys.stdout.flush()
    return noised_image

def save_nii_image(image, aff, save_path,dtype=np.float32):
    '''
    :param image: numpy array_like
    :param header: nibabel.nifti1.Nifti1Header
    :param save_path: str
    :return: None
    '''
    image = image.astype(dtype)
    nii_image = nib.Nifti1Image(image, affine=aff.astype(dtype))
    nib.save(nii_image, save_path)


def main(nc_data_path,save_nc_path,ce_data_path,save_ce_path,
         dose_fraction=0.75,current=400,exposure_time=0.5):
    '''
    :param raw_data_path: str
    :return: None
    '''

    assert os.path.exists(nc_data_path), f'Path not found: {nc_data_path}'
    assert os.path.exists(ce_data_path), f'Path not found: {ce_data_path}'

    if not os.path.exists(save_nc_path):
        os.makedirs(save_nc_path)

    if not os.path.exists(save_ce_path):
        os.makedirs(save_ce_path)

    for file in [f for f in os.listdir(nc_data_path) if f.endswith('.nii.gz')][::-1]:
        image_path = os.path.join(nc_data_path, file)
        image, aff = get_nii_image(image_path)
        print(file,image.shape)
        sys.stdout.flush()
        save_nii_image(image, aff, os.path.join(save_nc_path, file))
        # copy ce image from ce_data_path to save_ce_path using os.system
        os.system(f'cp {os.path.join(ce_data_path,file)} {os.path.join(save_ce_path,file)}')

        save_file = file.split('.')[0] + '_noised.nii.gz'


        noised_image = simulate_noise(image,dose_fraction=dose_fraction,current=current,exposure_time=exposure_time)
        save_nii_image(noised_image, aff, os.path.join(save_nc_path, save_file))
        # copy ce image from ce_data_path to save_ce_path using os.system
        os.system(f'cp {os.path.join (ce_data_path,file)} {os.path.join(save_ce_path,save_file)}')

if __name__ == '__main__':
    nc_data_path = '/path/to/kcd_data/raw_data/cropped_paired_combined_ncct/images'
    ce_data_path = '/path/to/kcd_data/raw_data/cropped_paired_combined_cect/images'
    save_ce_path = '/path/to/kcd_data/raw_data/final_pairedpretrain_ce/images'
    save_nc_path = '/path/to/kcd_data/raw_data/final_pairedpretrain_nc/images'
    dose_fractions = [0.7]
    for dose_fraction in dose_fractions[::-1]:
        main(nc_data_path,save_nc_path,ce_data_path,save_ce_path,
             dose_fraction=dose_fraction,current=1000,exposure_time=0.5)