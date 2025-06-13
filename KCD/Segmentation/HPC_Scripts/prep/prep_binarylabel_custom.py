import os
os.environ['OV_DATA_BASE'] = "/rds/project/rds-sDuALntK11g"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import sys

spacing = 1
preprocessed_name = '{}mm_kidneybinary_customPP'.format(spacing)
lb_classes = [2]
target_spacing=[spacing]*3


data_name = 'masked_final_pairedpretrain_ce'
prep = SegmentationPreprocessing(apply_resizing=True, 
                                    apply_pooling=False, 
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    scaling = [100, 0],
                                    window = [-200,300],
                                    reduce_lb_to_single_class = True,
                                    pooling_stride=None)
prep.initialise_preprocessing()
prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)

data_name = 'masked_final_pairedpretrain_nc'
prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False,
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    scaling = [100, 0],
                                    window = [-200,200],
                                    reduce_lb_to_single_class = True,
                                    pooling_stride=None)
prep.initialise_preprocessing()
prep.preprocess_raw_data(raw_data=data_name,
                          preprocessed_name=preprocessed_name)