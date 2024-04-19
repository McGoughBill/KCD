import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import numpy as np
import torch


def Segmentation_Preparation(home, seg_spacing=np.array([4,4,4]),
                             data_name='test',preprocessed_name='test', is_cect=False):

    pp_save_path = os.path.join(home, "preprocessed", data_name,preprocessed_name, 'images')

    rawdata_path = os.path.join(home, "raw_data", data_name, 'images')

    print("##SEG PREPROCESS##\nPreprocessing CT Volumes to {}\n Stored in location {}.".format(seg_spacing,
                                                                                               pp_save_path))
    print("")
    if is_cect:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-48, 280]),
                                                  scaling=np.array([55.6, 65.34]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)
    else:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-116, 130.]),
                                                  scaling=np.array([41.3, 12.3]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)

    preprocessing.preprocess_raw_data(raw_data=data_name,
                                      preprocessed_name=preprocessed_name,
                                      data_name=None,
                                      save_as_fp16=True)
    print("")

def Segmentation_Preparation_SSL(home, seg_spacing=np.array([4,4,4]), background_thresh=0.5,
                             data_name='test',preprocessed_name='test', is_cect=False):

    #base lung attenuation preprocess values on https://pubmed.ncbi.nlm.nih.gov/35607427/

    pp_save_path = os.path.join(home, "preprocessed", data_name,preprocessed_name, 'images')

    rawdata_path = os.path.join(home, "raw_data", data_name, 'images')

    print("##SEG PREPROCESS##\nPreprocessing CT Volumes to {}\n Stored in location {}.".format(seg_spacing,
                                                                                               pp_save_path))
    print("")
    if is_cect:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-1024,2048]),
                                                  scaling=np.array([3072, -1024]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)
    else:
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-500, 500]),
                                                  scaling=np.array([400, -500]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=False,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels=1)

    preprocessing.preprocess_raw_data(raw_data=data_name,
                                      preprocessed_name=preprocessed_name,
                                      data_name=None,
                                      save_as_fp16=True)
    print("")

    #copy all data from SSL_1 folder to PATH below
    if not is_cect:
        dst = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tciaandcoltea_ncct/1mm/images'
        src = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/{}/{}/images'.format(data_name, preprocessed_name)

        import shutil
        for file in [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith('.npy')]:
            shutil.copy(os.path.join(src, file), dst)

        #prepare indices
        prepare_indices(dst,background_thresh=background_thresh,patch_size=64)

def prepare_indices(preprocessed_folder,background_thresh=0.5,patch_size=64):
    # this function will prepare the indices for the training and validation sets, and save them to the appropriate
    # directories on a per-case basis. Indices are saved as .npy files, with the same name as the case they correspond to
    # in the 'images' directory. Patches are accepted as indices if they contain >background_thresh proportion of
    # non-background voxels (==0).
    assert os.path.exists(preprocessed_folder) and os.path.exists(os.path.join(preprocessed_folder, 'images'))

    index_path = os.path.join(preprocessed_folder, 'indices')
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    images = [f for f in os.listdir(os.path.join(preprocessed_folder, 'images')) if f.endswith('.npy')]

    for im in images:
        image = torch.Tensor(np.load(os.path.join(preprocessed_folder, 'images', im)))
        shape = image.shape
        # pad patch using torch if image is less than patch size
        padding = []
        for dim in range(1, 4):
            if shape[dim] < patch_size:
                pad = patch_size - shape[dim]
                padding = padding + [0, pad]
            else:
                padding = padding + [0, 0]

        image = torch.nn.functional.pad(image, tuple(padding[::-1]), 'constant', -3.15)

        #get full list of indices
        zxy_list = []
        for z in range(0, shape[1]-patch_size+1):
            for x in range(0, shape[2]-patch_size+1):
                for y in range(0, shape[3]-patch_size+1):
                    zxy_list.append((z, x, y))

        #get indices that are not background
        indices = []
        for z, x, y in zxy_list:
            patch = image[0, z:z + patch_size, x:x + patch_size, y:y + patch_size]
            if torch.sum(patch == 0) / patch_size ** 3 < background_thresh:
                indices.append((z, x, y))

        #save indices and padded image
        np.save(os.path.join(index_path, im), np.array(indices))



if __name__ == "__main__":
    Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='abdominal_tcia_ncct',preprocessed_name='SSL_1_ncct', is_cect=False)

    Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='abdominal_tcia_cect',preprocessed_name='SSL_1_cect', is_cect=True)

    # Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='cropped_tcia_ncct',preprocessed_name='tcia_cmd_2_ncct', is_cect=False)
    #
    # Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='cropped_tcia_cect',preprocessed_name='tcia_cmd_2_cect', is_cect=True)

    Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='coltea_art',preprocessed_name='SSL_1', is_cect=True)

    Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='coltea_nat',preprocessed_name='SSL_1', is_cect=False)

    # Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='add_cect',preprocessed_name='add_cect_2', is_cect=False)
