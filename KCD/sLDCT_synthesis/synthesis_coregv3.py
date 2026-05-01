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
    return image, aff, nib.load(image_path).header

def simulate_noise(image, is_kits, dose_fraction=0.75, exposure_time=0.5, current = 400):
    '''
    :param image_path: str
    :param dose_fraction: float
    :param current: float
    :return: numpy array_like
    '''
    noised_image = np.zeros_like(image)
    if is_kits:
        print('KITS',image.shape)
        for i in range(image.shape[0]):
            slice = image[i]
            slice = slice[:,: ,np.newaxis]
            noisy_slice = calculate_noise(slice, current= current*exposure_time, dose_fraction=dose_fraction)
            noised_image[i] = noisy_slice
    else:
        print('RCC',image.shape)
        for i in range(image.shape[-1]):
            slice = image[:,:,i]
            slice = slice[:,: ,np.newaxis]
            noisy_slice = calculate_noise(slice, current= current*exposure_time, dose_fraction=dose_fraction)
            noised_image[:,:,i] = noisy_slice
    sys.stdout.flush()
    return noised_image

def save_nii_image(image, aff, save_path,header,dtype=np.float32):
    '''
    :param image: numpy array_like
    :param header: nibabel.nifti1.Nifti1Header
    :param save_path: str
    :return: None
    '''
    image = image.astype(dtype)
    nii_image = nib.Nifti1Image(image, affine=aff.astype(dtype))
    nib.save(nii_image, save_path)


def main(raw_data_path, output_data_path,dose_fraction=0.75,current=400,exposure_time=0.5):
    '''
    :param raw_data_path: str
    :return: None
    '''

    assert os.path.exists(raw_data_path), f'Path not found: {raw_data_path}'

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    for file in [f for f in os.listdir(raw_data_path) if f.endswith('.nii.gz')]:
        is_kits =  'kits' in file.lower()
        image_path = os.path.join(raw_data_path, file)
        image, aff, header = get_nii_image(image_path)
        noised_image = simulate_noise(image,is_kits,dose_fraction=dose_fraction,current=current,exposure_time=exposure_time)
        save_nii_image(noised_image, aff, os.path.join(output_data_path, file),header)

if __name__ == '__main__':
    raw_data_path = '/path/to/kcd_data/data/seg_data/raw_data/coreg_ncct_v3/images'
    dose_fractions = [0.8]
    for dose_fraction in dose_fractions[::-1]:
        output_data_path = '/path/to/kcd_data/data/seg_data/raw_data/coreg_ncct_v3/noised_ims_{}'.format(dose_fraction)
        main(raw_data_path, output_data_path,dose_fraction=dose_fraction,current=1000,exposure_time=0.5)

    # copy the original images, noised images, and the corresponding masks to a new folder
    #when copying masks, copy the masks with the original name and the noised name

    new_folder = '/path/to/kcd_data/data/seg_data/raw_data/coreg_v3_noised'
    nf_im = os.path.join(new_folder,'images')
    nf_mk = os.path.join(new_folder,'labels')

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        os.makedirs(nf_im)
        os.makedirs(nf_mk)

    for file in [f for f in os.listdir(raw_data_path) if f.endswith('.nii.gz')]:
        # copy the original images
        os.system(f'cp {os.path.join(raw_data_path, file)} {os.path.join(nf_im, file)}')
        # copy the noised images
        noised_name = file.replace('.nii.gz','_noised.nii.gz')
        os.system(f'cp {os.path.join(output_data_path, file)} {os.path.join(nf_im, noised_name)}')
        # copy the masks
        os.system(f"cp {os.path.join(raw_data_path.replace('images','labels'), file)} {os.path.join(nf_mk, file)}")
        os.system(f"cp {os.path.join(raw_data_path.replace('images','labels'), file)} {os.path.join(nf_mk, noised_name)}")

