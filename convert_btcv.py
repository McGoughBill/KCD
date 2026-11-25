import os

# go into folder 'images' and convert all images from 'imgXXXX.nii.gz' to 'btcvXXXX.nii.gz'
# go into folder 'labels' and convert all images from 'labelXXXX.nii.gz' to 'btcvXXXX.nii.gz'

path = '/path/to/kcd_data/Downloads/btcv/Abdomen/RawData/Training'
for folder in ['images', 'labels']:
    for file in os.listdir(os.path.join(path, folder)):
        if folder == 'images' and file.endswith('.nii.gz'):
            os.rename(os.path.join(path, folder, file), os.path.join(path, folder, 'btcv' + file[3:]))
        elif folder == 'labels' and file.endswith('.nii.gz'):
            os.rename(os.path.join(path, folder, file), os.path.join(path, folder, 'btcv' + file[5:]))
