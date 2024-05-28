import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data'
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import numpy as np
import torch
import shutil


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
                                                  window=np.array([-200, 300]),
                                                  scaling=np.array([100, 0]),
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
                                                  window=np.array([-200, 200]),
                                                  scaling=np.array([100, 0]),
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

def Segmentation_Preparation_AbdAtlas(home, seg_spacing=np.array([4,4,4]),
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
                                                  window=np.array([-278, 279]),
                                                  scaling=np.array([55.6, 68]),
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
                                                  scaling=[100, 0],
                                                  window=[-200,200],
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

    if is_cect:
        dst = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tciaandcoltea_cect/{}mm_AA_ISBI/images'.format(seg_spacing[0])
        src = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/{}/{}/images'.format(data_name, preprocessed_name)
    else:
        dst = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tciaandcoltea_ncct/{}mm_AA_ISBI/images'.format(seg_spacing[0])
        src = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/{}/{}/images'.format(data_name,
                                                                                                    preprocessed_name)
    if not os.path.exists(dst):
        os.makedirs(dst)

    for file in [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith('.npy')]:
        shutil.copy(os.path.join(src, file), dst)


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
                                                  scaling=[100, 0],
                                                  window=[-200,200],
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
        dst = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/tciaandcoltea_ncct/{}mm_ISBI/images'.format(seg_spacing[0])
        src = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/preprocessed/{}/{}/images'.format(data_name, preprocessed_name)

        import shutil
        for file in [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith('.npy')]:
            shutil.copy(os.path.join(src, file), dst)



if __name__ == "__main__":
    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
                             data_name='coltea_nat',preprocessed_name='2mm_customPP', is_cect=False)

    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
                             data_name='coltea_art',preprocessed_name='2mm_customPP', is_cect=True)

    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='coltea_nat',preprocessed_name='1mm_customPP', is_cect=False)

    Segmentation_Preparation(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
                             data_name='coltea_art',preprocessed_name='1mm_customPP', is_cect=True)

    #
    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='abdominal_tcia_ncct',preprocessed_name='SSL_2_ISBI', is_cect=False)
    #
    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='abdominal_tcia_cect',preprocessed_name='SSL_2_ISBI', is_cect=True)

    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([4,4,4]),
    #                          data_name='abdominal_tcia_ncct',preprocessed_name='tcia_AA_4_ncct_ISBI', is_cect=False)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([4,4,4]),
    #                          data_name='abdominal_tcia_cect',preprocessed_name='tcia_AA_4_cect_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([4,4,4]),
    #                          data_name='coltea_art',preprocessed_name='coltea_AA_4_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([4,4,4]),
    #                          data_name='coltea_nat',preprocessed_name='coltea_AA_4_ISBI', is_cect=False)
    #
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='abdominal_tcia_ncct',preprocessed_name='tcia_AA_2_ncct_ISBI', is_cect=False)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='abdominal_tcia_cect',preprocessed_name='tcia_AA_2_cect_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='coltea_art',preprocessed_name='coltea_AA_2_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_AbdAtlas(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='coltea_nat',preprocessed_name='coltea_AA_2_ISBI', is_cect=False)

    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
    #                          data_name='coltea_art',preprocessed_name='SSL_1_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([1,1,1]),
    #                          data_name='coltea_nat',preprocessed_name='SSL_1_ISBI', is_cect=False)
    #
    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='coltea_art',preprocessed_name='SSL_2_ISBI', is_cect=True)
    #
    # Segmentation_Preparation_SSL(home=os.environ['OV_DATA_BASE'], seg_spacing=np.array([2,2,2]),
    #                          data_name='coltea_nat',preprocessed_name='SSL_2_ISBI', is_cect=False)
