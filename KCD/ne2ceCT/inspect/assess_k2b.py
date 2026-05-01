import torch
import os
os.environ['OV_DATA_BASE'] = '/path/to/kcd_data/Downloads/assess_k2b'
from KCD.ne2ceCT.dataloader_seg import CrossPhaseDataset
from KCD.Segmentation.Inference import infer_network
from KCD.ne2ceCT import loss_function_seg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import SparsePCA as PCA

plt.switch_backend('TkAgg')

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
    ce_model.load_state_dict(torch.load('/path/to/kcd_data/Downloads/assess_k2b/ce_model', map_location='cpu'))

    nc2ceCT_model = get_3d_featureoutput_unet(1, 2, 7)
    nc2ceCT_model.load_state_dict(torch.load('/path/to/kcd_data/Downloads/assess_k2b/network_weights', map_location='cpu'))

    ce_model.eval()
    nc2ceCT_model.eval()

    ce_data_path = '/path/to/kcd_data/Downloads/assess_k2b/ce'
    ne_data_path = '/path/to/kcd_data/Downloads/assess_k2b/nc'
    split_path = '/path/to/kcd_data/Downloads/assess_k2b'
    seg_ce_path = '/path/to/kcd_data/Downloads/assess_k2b/seg/images'


    dataset = CrossPhaseDataset(ne_data_path, ce_data_path,split_path,seg_ce_path, device='cpu', is_train=False,
                                patch_size=128)
    dataset.apply_foldsplit()

    for count,(ne, ce, seg) in enumerate(dataset):
        ne = ne.unsqueeze(0)
        ce = ce.unsqueeze(0)
        print(f'Processing patient {count+1} of {len(dataset)}')
        print(ne.shape, ce.shape, seg.shape)
        print(f'Number of seg voxels: {seg.sum()}')
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

        # set slice index to the slice with the maximum difference between latent_nc_pca and latent_nc_from_ce_pca
        slice = np.argmax(seg.sum(dim=(0,2,3)).cpu().detach().numpy())
        print(f'Slice with maximum difference between latent spaces: {slice}, with {seg_output[0, slice].sum()} segmented voxels')

        # plot a 3 row 3 column set of plots
        # top row is CE image, NC image, CE segmentation
        # middle row is CE latent space, NC latent space, CE segmentation overlayed on the NC image
        # bottom row is CE cancer segmentation, NC cancer segmentation, NC cancer segmentation overlayed on the NC image
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs[0, 0].imshow(ce[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[0, 0].set_title('CE Image')
        axs[0, 1].imshow(ne[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[0, 1].set_title('NC Image')
        axs[0, 2].imshow(seg[0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[0, 2].set_title('CE Segmentation')
        axs[1, 0].imshow(np.rot90(np.flip(latent_ce[0:3, slice, :, :].cpu().detach().numpy().T,1)), cmap='gray')
        axs[1, 0].set_title('CE Latent Space')
        axs[1, 1].imshow(np.rot90(np.flip(latent_nc[0:3, slice, :, :].cpu().detach().numpy().T,1)), cmap='gray')
        axs[1, 1].set_title('NC Latent Space')
        axs[1, 2].imshow(ne[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[1, 2].imshow(seg[0, slice, :, :].cpu().detach().numpy(), cmap='jet', alpha=0.5)
        axs[1, 2].set_title('Kidney Segmentation on NC Image')
        axs[2, 0].imshow(ce[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[2, 0].imshow(seg_output_ce[0, slice, :, :].cpu().detach().numpy(), cmap='jet', alpha=0.5)
        axs[2, 0].set_title('CE Cancer Segmentation')
        axs[2, 1].imshow(ne[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[2, 1].imshow(seg_output_from_ce[0, slice, :, :].cpu().detach().numpy(), cmap='jet', alpha=0.5)
        axs[2, 1].set_title('NC Cancer Segmentation with CE model')
        axs[2, 2].imshow(ne[0, 0, slice, :, :].cpu().detach().numpy(), cmap='gray')
        axs[2, 2].imshow(seg_output[0, slice, :, :].cpu().detach().numpy(), cmap='jet', alpha=0.5)
        axs[2, 2].set_title('NC Cancer Segmentation on NC Image')

        plt.tight_layout()

        abs_diff_latent = 0
        abs_diff_ce = 0
        abs_diff_output = 0
        abs_diff_ce_output = 0
        loss_func = loss_function_seg.PyramidalLatentSimilarityLoss(l2_weight=0, l1_weight=0, cce_weight=1, cos_weight=1,
                                                                    var_loss_weight=0, cov_loss_weight=0, clip_loss_weight=0, out_dim=2)
        print(f'Calculating loss between CE and NC latent spaces')
        loss_nc = loss_func(original_nc, original_ce, seg.unsqueeze(0))
        print(
        )
        print(f'Calculating loss between CE and NC latent spaces from CE model')
        loss_nc_from_ce = loss_func(original_nc_from_ce, original_ce, seg.unsqueeze(0))
        print()

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
        print(f'Loss between NC and CE latent spaces: {loss_nc.item()}')
        print(f'Loss between NC and CE latent spaces from CE model: {loss_nc_from_ce.item()}')

        # print size of the average weight and bias in each model
        print(f'Average weight size in CE model: {torch.abs(torch.cat([p.view(-1) for p in ce_model.parameters()])).mean()}')
        print(f'Average weight size in NC model: {torch.abs(torch.cat([p.view(-1) for p in nc2ceCT_model.parameters()])).mean()}')
        print(f'Average size of feature output in CE model: {latent_ce.mean()}')
        print(f'Average size of feature output in NC model: {latent_nc.mean()}')

        plt.show(block=True)
