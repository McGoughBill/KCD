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

    nc2ceCT_model = fmcib_model()
    nc2ceCT_model.load_state_dict(torch.load('/Users/mcgoug01/Downloads/assessing_latent_resnet/network_weights', map_location='cpu'))

    ce_model.eval()
    nc2ceCT_model.eval()

    ce_data_path = '/Users/mcgoug01/Downloads/assessing_latent_resnet/ce'
    ne_data_path = '/Users/mcgoug01/Downloads/assessing_latent_resnet/nc'

    dataset = CrossPhaseDataset(ne_data_path, ce_data_path, device='cpu', is_train=False,patches_per_case=200)
    dataset.apply_foldsplit(split_ratio=0)

    # get an inference sample
    sum_cene = 0
    sum_cene_fromce = 0
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
            latent_nc = nc2ceCT_model(ne)[0].reshape(64,64)
            latent_ce = ce_model(ce)[0].reshape(64,64)
            latent_nc_from_ce = ce_model((ce_equiv_ne))[0].reshape(64,64)

        #normalise the latent spaces
        latent_nc = (latent_nc - latent_nc.mean()) / latent_nc.std()
        latent_ce = (latent_ce - latent_ce.mean()) / latent_ce.std()
        latent_nc_from_ce = (latent_nc_from_ce - latent_nc_from_ce.mean()) / latent_nc_from_ce.std()


        # plot a 3 row 2 column set of plots
        # top row is the ce image and the nc image
        # middle row is the ce latent space and the nc latent space
        # bottom row is the nc latent space from the ce model and the difference between the ce and nc latent spaces
        plt.switch_backend('TkAgg')
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        # set location of each figure to the same point on the monitor
        fig.canvas.manager.window.wm_geometry("+0+0")
        axs[0, 0].imshow(ne[0, 0, 32, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=2.5)
        #add title
        axs[0, 0].set_title('NC')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(ce[0, 0, 32, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[0, 1].set_title('CE')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(latent_nc,vmin=0, vmax=1)
        axs[1, 0].set_title('(a) Domain-Adapted Latent Space')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(latent_ce,vmin=0, vmax=1)
        axs[1, 1].set_title('(b) Original Model Latent Space')
        axs[1, 1].axis('off')
        axs[2, 0].imshow(latent_nc_from_ce,vmin=0, vmax=1)
        axs[2, 0].set_title('(c) Original Model Latent Space using NC input')
        axs[2, 0].axis('off')
        axs[2, 1].imshow((latent_ce - latent_nc),vmin=0, vmax=1)
        axs[2, 1].set_title('(d) Difference between Latent Spaces in (a) and (b)')
        axs[2, 1].axis('off')

        abs_diff = torch.abs(latent_ce - latent_nc).mean()
        abs_diff_ce = torch.abs(latent_nc_from_ce - latent_nc).mean()

        sum_cene+=abs_diff
        sum_cene_fromce+=abs_diff_ce

        # insert the absolute differences as a figure-wide title up to 3 sigfig
        fig.suptitle(f'Absolute difference between CE and NC latent spaces: {abs_diff:.3f}\n'
                     f'Absolute difference between CE and NC latent spaces from CE model: {abs_diff_ce:.3f}\n'
                     f'Average differences: {sum_cene/(count+1):.3f}\n'
                     f'Average differences from CE model: {sum_cene_fromce/(count+1):.3f}'
                     )

        print(f'Absolute difference between CE and NC latent spaces: {abs_diff}')
        print(f'Absolute difference between CE and NC latent spaces from CE model: {abs_diff_ce}')

        # print size of the average weight and bias in each model
        print(f'Average weight size in CE model: {torch.abs(torch.cat([p.view(-1) for p in ce_model.parameters()])).mean()}')
        print(f'Average weight size in NC model: {torch.abs(torch.cat([p.view(-1) for p in nc2ceCT_model.parameters()])).mean()}')

        plt.show(block=True)
