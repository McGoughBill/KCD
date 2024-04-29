import torch
import os
os.environ['OV_DATA_BASE'] = '/Users/mcgoug01/Downloads/assessing_latent'
from KCD.ne2ceCT.dataloader_SSL import CrossPhaseDataset
from KCD.Segmentation.Inference import infer_network
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import SparsePCA as PCA
from fmcib.models import fmcib_model


# this script loads the trained models and uses them to generate latent spaces for the same patient
# in CECT and NCCT. It then compares the latent spaces and the output spaces of the models
# to assess the similarity between the two modalities. The nc2ceCT model is trained to match
# the latent spaces of the two modalities, so should hopefully be more similar in latent space than
# the ce model just naively applied to the nc data. We visualise the latent spaces with the first
# 3 principal components of the CE model's latent space on CE data, assuming this is 'the answer'.

if __name__ == '__main__':
    os.chdir('/Users/mcgoug01/Downloads/assessing_latent_resnet')
    # load model
    ce_model = fmcib_model()

    bs80,bs160 = fmcib_model(), fmcib_model()
    ce_model.eval()
    bs80.load_state_dict(torch.load('/Users/mcgoug01/Downloads/assessing_latent_resnet/l1-0.2_l2-1.0_var0.01_cov0.2_cce0.0_cos1.0_clip0.0_2e-05lr_0.99998lg_80bs_5000ep_4gpus/network_weights', map_location='cpu'))
    bs160.load_state_dict(torch.load('/Users/mcgoug01/Downloads/assessing_latent_resnet/l1-0.2_l2-0.5_var0.01_cov0.2_cce0.0_cos1.0_clip0.0_2e-05lr_0.99998lg_160bs_5000ep_4gpus/network_weights', map_location='cpu'))

    ce_data_path = '/Users/mcgoug01/Downloads/assessing_latent_resnet/ce'
    ne_data_path = '/Users/mcgoug01/Downloads/assessing_latent_resnet/nc'
    split_path = '/Users/mcgoug01/Downloads/assessing_latent_resnet'

    dataset = CrossPhaseDataset(ne_data_path, ce_data_path, split_path,
                                device='cpu', is_train=False,patches_per_case=200,
                                background_threshold=0.5)
    dataset.apply_foldsplit(split_ratio=0)

    # get an inference sample
    sum_80, sum_160, sum_ce = 0, 0, 0
    for count,(ne, ce) in enumerate(dataset):

        ne = ne.unsqueeze(0)
        ce = ce.unsqueeze(0)

        #print min max of each image
        print(f'NC min: {ne.min()} max: {ne.max()}')
        print(f'CE min: {ce.min()} max: {ce.max()}')

        ce_equiv_ne = (ne/(2.5*3.072)) + (524/3072)
        # ce_equiv_ne[ne==0] = 0
        print(f'ce_equiv_ne min: {ce_equiv_ne.min()} max: {ce_equiv_ne.max()}')
        # get the latent space
        with torch.no_grad():
            latent_80 = bs80(ne)[0].reshape(64,64)
            latent_160 = bs160(ne)[0].reshape(64,64)
            latent_ce = ce_model(ce)[0].reshape(64,64)
            latent_nc_from_ce = ce_model((ce_equiv_ne))[0].reshape(64,64)

        abs_diff_80 = torch.abs(latent_80 - latent_ce).mean()
        abs_diff_160 = torch.abs(latent_160 - latent_ce).mean()
        abs_diff_ce = torch.abs(latent_nc_from_ce - latent_ce).mean()

        sum_80 += abs_diff_80
        sum_160 += abs_diff_160
        sum_ce += abs_diff_ce

        # plot a 3 row 2 column set of plots
        # top row is the ce image and the nc image
        # middle row is the ce latent space and the nc latent space
        # bottom row is the nc latent space from the ce model and the difference between the ce and nc latent spaces
        if (count+1) %300 == 0 and count > 0:
            plt.switch_backend('TkAgg')
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            # set location of each figure to the same point on the monitor
            fig.canvas.manager.window.wm_geometry("+0+0")
            axs[0, 0].imshow(ne[0, 0, 32, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=2.5)
            #add title
            axs[0, 0].set_title('NC')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(ce[0, 0, 32, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            axs[0, 1].set_title('CE')
            axs[0, 1].axis('off')
            axs[1, 0].imshow(latent_80,vmin=-1, vmax=2.5)
            axs[1, 0].set_title('(a) Domain-Adapted ClipCosMSE Latent Space')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(latent_ce,vmin=-1, vmax=2.5)
            axs[1, 1].set_title('(b) Original Model Latent Space')
            axs[1, 1].axis('off')
            axs[2, 0].imshow(latent_nc_from_ce,vmin=-1, vmax=2.5)
            axs[2, 0].set_title('(c) Original Model Latent Space using NC input')
            axs[2, 0].axis('off')
            axs[2, 1].imshow(torch.abs(latent_ce - latent_160),vmin=0, vmax=2)
            axs[2, 1].set_title('(d) Difference between Latent Spaces in (a) and (b)')
            axs[2, 1].axis('off')

            # insert the absolute differences as a figure-wide title up to 3 sigfig
            fig.suptitle(f'Average absolute difference of bs80: {sum_80/(count+1):.3f}\n'
                            f'Average absolute difference of bs160: {sum_160/(count+1):.3f}\n'
                            f'Average absolute difference of ce: {sum_ce/(count+1):.3f}\n'
                            f'Number of samples: {count+1}', fontsize=12)
            plt.tight_layout()

            plt.show(block=True)
