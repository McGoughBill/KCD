import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data/"
from KCD.Segmentation.ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

spacing = 1
preprocessed_name = '1mm_canceronly_customPP'

lb_classes = [2]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False,
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True,
                                     window=[-200, 300],
                                     scaling=[100, 0],
                                     pooling_stride=None)

prep.initialise_preprocessing()

prep.preprocess_raw_data(raw_data='all_add',
                          preprocessed_name=preprocessed_name)
prep.preprocess_raw_data(raw_data='kits23_nooverlap',
                          preprocessed_name=preprocessed_name)

spacing = 2
preprocessed_name = '2mm_canceronly_customPP'

lb_classes = [2]
target_spacing=[spacing]*3
prep = SegmentationPreprocessing(apply_resizing=True,
                                    apply_pooling=False,
                                    apply_windowing=True,
                                    lb_classes=lb_classes,
                                    target_spacing=target_spacing,
                                    reduce_lb_to_single_class = True,
                                     window=[-200, 300],
                                     scaling=[100, 0],
                                     pooling_stride=None)

prep.initialise_preprocessing()

prep.preprocess_raw_data(raw_data='all_add',
                          preprocessed_name=preprocessed_name)
prep.preprocess_raw_data(raw_data='kits23_nooverlap',
                          preprocessed_name=preprocessed_name)