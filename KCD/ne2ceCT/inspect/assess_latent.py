import torch
import os
os.environ['OV_DATA_BASE'] = '/path/to/kcd_data/Downloads/assessing_latent'
from KCD.ne2ceCT.dataloader import CrossPhaseDataset
from KCD.Segmentation.Inference import infer_network
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import SparsePCA as PCA

# this script loads the trained models and uses them to generate latent spaces for the same patient
# in CECT and NCCT. It then compares the latent spaces and the output spaces of the models
# to assess the similarity between the two modalities. The nc2ceCT model is trained to match
# the latent spaces of the two modalities, so should hopefully be more similar in latent space than
# the ce model just naively applied to the nc data. We visualise the latent spaces with the first
# 3 principal components of the CE model's latent space on CE data, assuming this is 'the answer'.

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]
    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

if __name__ == '__main__':
    # load model
    ce_model = get_3d_featureoutput_unet(1, 2, 7)
    ce_model.load_state_dict(torch.load('/path/to/kcd_data/Downloads/assessing_latent/ce_model', map_location='cpu'))

    nc2ceCT_model = get_3d_featureoutput_unet(1, 2, 7)
    nc2ceCT_model.load_state_dict(torch.load('/path/to/kcd_data/Downloads/assessing_latent/network_weights', map_location='cpu'))

    ce_model.eval()
    nc2ceCT_model.eval()

    ce_data_path = '/path/to/kcd_data/Downloads/assessing_latent/ce'
    ne_data_path = '/path/to/kcd_data/Downloads/assessing_latent/nc'
    split_path = '/path/to/kcd_data/Downloads/assessing_latent'

    dataset = CrossPhaseDataset(ne_data_path, ce_data_path,split_path, device='cpu', is_train=False,
                                patch_size=128)
    dataset.apply_foldsplit()


    # cect
    #    window=np.array([-278, 279]),
    #    scaling=np.array([55.6, 68])
    # ncct
    #    window=np.array([-120, 240]),
    #    scaling=np.array([60, 60])

    for count,(ne, ce) in enumerate(dataset):
        ne = ne.unsqueeze(0)
        ce = ce.unsqueeze(0)
        ce_from_ne = ne

        # get the latent space
        with torch.no_grad():
            original_nc = nc2ceCT_model(ne)
            latent_nc = original_nc[1][0]
            original_ce = ce_model(ce)
            latent_ce = original_ce[1][0]
            original_nc_from_ce = ce_model(ce_from_ne)
            latent_nc_from_ce = original_nc_from_ce[1][0]

        seg_output = original_nc[0].argmax(dim=1)
        seg_output_from_ce = original_nc_from_ce[0].argmax(dim=1)
        seg_output_ce = original_ce[0].argmax(dim=1)

        #clip latent spaces to -2-2
        extent = 2
        latent_nc = torch.clamp(latent_nc, -1*extent, extent)
        latent_ce = torch.clamp(latent_ce, -1*extent, extent)
        latent_nc_from_ce = torch.clamp(latent_nc_from_ce, -1*extent, extent)

        #normalise latent spaces
        latent_nc = (latent_nc - latent_nc.min()) / (latent_nc.max() - latent_nc.min())
        latent_ce = (latent_ce - latent_ce.min()) / (latent_ce.max() - latent_ce.min())
        latent_nc_from_ce = (latent_nc_from_ce - latent_nc_from_ce.min()) / (latent_nc_from_ce.max() - latent_nc_from_ce.min())

        latent_ce_reshape = latent_ce.view(latent_ce.shape[0], -1).T
        latent_nc_reshape = latent_nc.view(latent_nc.shape[0], -1).T
        latent_nc_from_ce_reshape = latent_nc_from_ce.view(latent_nc_from_ce.shape[0], -1).T

        # pca = PCA(n_components=3)
        # pca.fit(latent_ce_reshape.cpu().detach().numpy())
        # latent_ce_pca = pca.transform(latent_ce_reshape.cpu().detach().numpy())
        # latent_nc_pca = pca.transform(latent_nc_reshape.cpu().detach().numpy())
        # latent_nc_from_ce_pca = pca.transform(latent_nc_from_ce_reshape.cpu().detach().numpy())

        # # reshape the latent spaces to 3D, back to the original shape
        # latent_nc_pca = latent_nc_pca.T.reshape(3,latent_nc.shape[1],latent_nc.shape[2],latent_nc.shape[3])
        # latent_ce_pca = latent_ce_pca.T.reshape(3,latent_ce.shape[1],latent_ce.shape[2],latent_ce.shape[3])
        # latent_nc_from_ce_pca = latent_nc_from_ce_pca.T.reshape(3,latent_nc_from_ce.shape[1],latent_nc_from_ce.shape[2],latent_nc_from_ce.shape[3])

        # set slice index to the slice with the maximum difference between latent_nc_pca and latent_nc_from_ce_pca
        slice = np.argmax(seg_output.sum(dim=(0,2,3)).cpu().detach().numpy())

        # plot a 3 row 2 column set of plots
        # top row is the ce image and the nc image
        # middle row is the ce latent space and the nc latent space
        # bottom row is the nc latent space from the ce model and the difference between the ce and nc latent spaces
        plt.switch_backend('TkAgg')
        # fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        plt.imshow(ne[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # axs[0, 0].imshow(ne[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # #add title
        # axs[0, 0].set_title('NC')
        # axs[0, 0].axis('off')
        # axs[0, 1].imshow(ce[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # axs[0, 1].set_title('CE')
        # axs[0, 1].axis('off')
        # # axs[1, 0].imshow(latent_nc_pca[:, slice, :, :].T)
        # # axs[1, 0].set_title('NC latent')
        # # axs[1, 0].axis('off')
        # # axs[1, 1].imshow(latent_ce_pca[:, slice, :, :].T)
        # # axs[1, 1].set_title('CE latent')
        # # axs[1, 1].axis('off')
        # axs[1, 1].imshow(ne[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # axs[1, 1].imshow(seg_output[:,slice, :, :].T.cpu().detach().numpy(), alpha=0.5)
        # axs[1, 1].set_title('Segmentation Output from NC Data')
        # axs[1, 1].axis('off')
        # axs[1,0].set_title('Segmentation Output from CE Data')
        # axs[1, 0].imshow(ce[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # # axs[1, 0].imshow(ne[0, 0, slice, :, :].T.cpu().detach().numpy(), cmap='gray', vmin=-3, vmax=3)
        # axs[1,0].imshow(seg_output_from_ce[:,slice, :, :].T.cpu().detach().numpy(), alpha=0.5)
        # axs[1,0].axis('off')
        # # axs[1, 0].imshow(seg_output[:,slice, :, :].T.cpu().detach().numpy(), cmap='gray')
        # # axs[1, 0].set_title('Segmentation Output from Domain-Adapted NC Model')
        # # axs[1, 0].axis('off')
        # # axs[1,1].set_title('Segmentation Output from CE Model on CE input')
        # # axs[1,1].imshow(seg_output_ce[:,slice, :, :].T.cpu().detach().numpy(), cmap='gray')
        # # axs[1,1].axis('off')
        plt.tight_layout()

        abs_diff_latent = 0
        abs_diff_ce = 0
        abs_diff_output = 0
        abs_diff_ce_output = 0

        for output in zip(original_nc, original_ce,original_nc_from_ce):
            nc_l,ce_l,ne_from_ce_l = output
            if nc_l.shape[1] == 2 :
                if nc_l.shape[-1]>16:
                    # add the cross entropy difference between the output spaces
                    abs_diff_output += (torch.nn.CrossEntropyLoss()(nc_l, ce_l.argmax(dim=1)) + torch.nn.CrossEntropyLoss()(ce_l, nc_l.argmax(dim=1)))/2
                    abs_diff_ce_output += (torch.nn.CrossEntropyLoss()(ne_from_ce_l, ce_l.argmax(dim=1)) + torch.nn.CrossEntropyLoss()(ce_l, ne_from_ce_l.argmax(dim=1)))/2
            else:
                abs_diff_latent += torch.abs(nc_l - ce_l).mean()
                abs_diff_ce += torch.abs(ne_from_ce_l - ce_l).mean()

        print(f'Absolute difference between CE and NC latent spaces: {abs_diff_latent}')
        print(f'Absolute difference between CE and NC latent spaces from CE model: {abs_diff_ce}')
        print(f'Cross Entropy between CE and NC output : {abs_diff_output}')
        print(f'Cross Entropy between CE and NC output from CE model: {abs_diff_ce_output}')

        # print size of the average weight and bias in each model
        print(f'Average weight size in CE model: {torch.abs(torch.cat([p.view(-1) for p in ce_model.parameters()])).mean()}')
        print(f'Average weight size in NC model: {torch.abs(torch.cat([p.view(-1) for p in nc2ceCT_model.parameters()])).mean()}')
        print(f'Average size of feature output in CE model: {latent_ce.mean()}')
        print(f'Average size of feature output in NC model: {latent_nc.mean()}')

        plt.show(block=True)
