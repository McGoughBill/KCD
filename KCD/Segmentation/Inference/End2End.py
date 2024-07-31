

import os
os.environ['OV_DATA_BASE'] = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/'
from KCD.Segmentation.Inference.end2end import EnsembleSeg, Masking
from KCD.Segmentation.Inference.EnsembleSeg import Ensemble_Seg
import numpy as np

def EndToEnd(dataset, stage1_path, stage_2path, save_path='None'):

    s1 = EnsembleSeg.Ensemble_Seg(dataset,name='s1',
                            seg_fp=stage1_path,
                            spacing=np.array([4] * 3),
                            do_prep=do_prep, do_infer=do_infer,
                            is_cect=False, cont=False,overlap=0.5,
                            batch_size=4, patch_size=64)

    dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', dataset, 'images'),
    masked_dataset_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'masked_' + dataset, 'images')
    pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions_nii', dataset_name, 's1_[2 2 2]mm')

    Masking.mask_dataset(dataset_path, pred_path, masked_dataset_path)

    s2 = EnsembleSeg.Ensemble_Seg('masked_' + dataset,name='s2',
                            seg_fp=stage_2path,
                            spacing=np.array([2] * 3),
                            do_prep=do_prep, do_infer=do_infer,
                            is_cect=False, cont=True,overlap=0.5,
                            batch_size=4, patch_size=64)

if __name__ == '__main__':
    dataset = 'coreg_ncct'
    s1_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/coreg_v2_noised/4mm_kidneybinary_customPP/6_finetune'
    s1_path = '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/masked_coreg_v2_noised/2mm_cancerbinary_customPP/6_finetune_l2-0.0_cce1.0_cos1.0_2e-05lr_0.999997lg_16bs_2000ep_4gpus_efr0_dfr0_seg_epochs200'

    EndToEnd(dataset, s1_path, s2_path)