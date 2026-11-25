import torch
import os
os.environ['OV_DATA_BASE'] = '/path/to/kcd_data/Downloads/assessing_latent'
from KCD.ne2ceCT.dataloader_SSL import CrossPhaseDataset
from KCD.Segmentation.Inference import infer_network
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import SparsePCA as PCA
from fmcib.models import fmcib_model
from KCD.ne2ceCT.loss_function import LatentSimilarityLoss_resnet as LatentSimilarityLoss


# this script loads the trained models and uses them to generate latent spaces for the same patient
# in CECT and NCCT. It then compares the latent spaces and the output spaces of the models
# to assess the similarity between the two modalities. The nc2ceCT model is trained to match
# the latent spaces of the two modalities, so should hopefully be more similar in latent space than
# the ce model just naively applied to the nc data. We visualise the latent spaces with the first
# 3 principal components of the CE model's latent space on CE data, assuming this is 'the answer'.

if __name__ == '__main__':
    os.chdir('/path/to/kcd_data/Downloads/assessing_latent_resnet')
    # load model
    ce_model = fmcib_model()

    converted_dam = fmcib_model()
    converted_dam.load_state_dict(torch.load('/path/to/kcd_data/Downloads/assessing_latent_resnet/l1-0.2_l2-1.0_var0.01_cov0.2_cce0.0_cos1.0_clip0.0_2e-05lr_0.99998lg_80bs_5000ep_4gpus/network_weights', map_location='cpu'))

    ce_model.eval(), converted_dam.eval()

    ce_data_path = '/path/to/kcd_data/Downloads/assessing_latent_resnet/ce'
    ne_data_path = '/path/to/kcd_data/Downloads/assessing_latent_resnet/nc'
    split_path = '/path/to/kcd_data/Downloads/assessing_latent_resnet'

    dataset = CrossPhaseDataset(ne_data_path, ce_data_path, split_path,
                                device='cpu', is_train=False,patches_per_case=200,
                                background_threshold=0.6,background_value=0)
    dataset.apply_foldsplit(split_ratio=0)

    # get an inference sample
    sum_converted, sum_ce, sum_loss_converted, sum_loss_ce = 0, 0, 0, 0
    sum_latent_converted, sum_latent_ce = 0, 0
    for count,(ne, ce) in enumerate(dataset):

        ne = ne.unsqueeze(0)
        ce = ce.unsqueeze(0)

        #print min max of each image
        print(f'NC min: {ne.min()} max: {ne.max()}')
        print(f'CE min: {ce.min()} max: {ce.max()}')

        ce_equiv_ne = (ne/(2.5*3.072)) + (524/3072)
        ce_equiv_ne[ne==0] = 0
        print(f'ce_equiv_ne min: {ce_equiv_ne.min()} max: {ce_equiv_ne.max()}')
        # get the latent space
        with torch.no_grad():
            latent_converted = converted_dam(ne)[0].reshape(64,64)
            latent_ce = ce_model(ce)[0].reshape(64,64)
            latent_nc_from_ce = ce_model((ce_equiv_ne))[0].reshape(64,64)

        abs_diff_latent = torch.abs(latent_converted - latent_ce).mean()
        abs_diff_ce = torch.abs(latent_nc_from_ce - latent_ce).mean()

        loss_func = LatentSimilarityLoss(l1_weight=0.2, l2_weight=1,
                                         cce_weight=0, cos_weight=1,
                                         var_loss_weight=0, cov_loss_weight=0.2,
                                         clip_loss_weight=0).cpu()

        sum_loss_ce += loss_func(latent_ce.unsqueeze(0).reshape(1,-1), latent_nc_from_ce.unsqueeze(0).reshape(1,-1)).item()
        sum_loss_converted += loss_func(latent_ce.unsqueeze(0).reshape(1,-1), latent_converted.unsqueeze(0).reshape(1,-1)).item()
        sum_converted += abs_diff_latent
        sum_ce += abs_diff_ce
        sum_latent_ce += latent_ce.mean()
        sum_latent_converted += latent_converted.mean()

        # plot a 3 row 2 column set of plots
        # top row is the ce image and the nc image
        # middle row is the ce latent space and the nc latent space
        # bottom row is the nc latent space from the ce model and the difference between the ce and nc latent spaces
        if (count+1) %5000 == 0 and count > 0:
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
            axs[1, 0].imshow(latent_converted,vmin=0, vmax=2.5)
            axs[1, 0].set_title('(a) Domain-Adapted Latent Space')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(latent_ce,vmin=-1, vmax=2.5)
            axs[1, 1].set_title('(b) Original Model Latent Space')
            axs[1, 1].axis('off')
            axs[2, 0].imshow(latent_nc_from_ce,vmin=-1, vmax=2.5)
            axs[2, 0].set_title('(c) Original Model Latent Space using NC input')
            axs[2, 0].axis('off')
            axs[2, 1].imshow(torch.abs(latent_ce - latent_converted),vmin=0, vmax=2.5)
            axs[2, 1].set_title('(d) Difference between Latent Spaces in (a) and (b)')
            axs[2, 1].axis('off')

            # insert the absolute differences as a figure-wide title up to 3 sigfig
            fig.suptitle(f'Average absolute difference of converted: {sum_converted/(count+1):.3f}\n'
                            f'Average absolute difference of ce: {sum_ce/(count+1):.3f}\n'
                         f'Average latent space of converted: {sum_latent_converted/(count+1):.3f}\n'
                            f'Average latent space of ce: {sum_latent_ce/(count+1):.3f}\n'
                            f'Average loss of converted: {sum_loss_converted/(count+1):.3f}\n'
                            f'Average loss of ce: {sum_loss_ce/(count+1):.3f}\n'
                            f'Number of samples: {count+1}', fontsize=12)
            plt.tight_layout()

            plt.show(block=True)
