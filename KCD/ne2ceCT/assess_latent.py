import torch
import os
os.environ['OV_DATA_BASE'] = '/Users/mcgoug01/Downloads/assessing_latent'
from KCD.ne2ceCT.dataloader import CrossPhaseDataset
from KCD.Segmentation.Inference import infer_network
import matplotlib.pyplot as plt
import numpy as np


def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]
    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

if __name__ == '__main__':
    # load model
    ce_model = get_3d_featureoutput_unet(1, 2, 6)
    ce_model.load_state_dict(torch.load('/Users/mcgoug01/Downloads/assessing_latent/ce_model', map_location='cpu'))

    nc2ceCT_model = get_3d_featureoutput_unet(1, 2, 6)
    nc2ceCT_model.load_state_dict(torch.load('/Users/mcgoug01/Downloads/assessing_latent/ne2ceCT_model', map_location='cpu'))

    ce_model.eval()
    nc2ceCT_model.eval()

    ce_data_path = '/Users/mcgoug01/Downloads/assessing_latent/ce'
    ne_data_path = '/Users/mcgoug01/Downloads/assessing_latent/nc'

    # # display the same slice in the same image from each data path
    # ce = np.load(os.path.join(ce_data_path, os.listdir(ce_data_path)[1]),allow_pickle=True)[0]
    # ne = np.load(os.path.join(ne_data_path, os.listdir(ne_data_path)[1]),allow_pickle=True)[0]
    #
    # plt.switch_backend('TkAgg')
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(ne[50, :, :].T)
    # axs[0].set_title('NC')
    # axs[0].axis('off')
    # axs[1].imshow(ce[50, :, :].T)
    # axs[1].set_title('CE')
    # axs[1].axis('off')
    # plt.show()

    dataset = CrossPhaseDataset(ne_data_path, ce_data_path, device='cpu', is_train=True)
    dataset.apply_foldsplit(split_ratio=1)

    # get an inference sample

    ne, ce = dataset[0]
    ne = ne.unsqueeze(0)
    ce = ce.unsqueeze(0)

    # get the latent space
    with torch.no_grad():
        original_nc = nc2ceCT_model(ne)
        latent_nc = nc2ceCT_model(ne)[1][0]
        original_ce = ce_model(ce)
        latent_ce = original_ce[1][0]
        original_nc_from_ce = ce_model(ne)
        latent_nc_from_ce = original_nc_from_ce[1][0]

    #clip latent spaces to -2-2
    extent = 2
    latent_nc = torch.clamp(latent_nc, -1*extent, extent)
    latent_ce = torch.clamp(latent_ce, -1*extent, extent)
    latent_nc_from_ce = torch.clamp(latent_nc_from_ce, -1*extent, extent)

    #normalise latent spaces
    latent_nc = (latent_nc - latent_nc.min()) / (latent_nc.max() - latent_nc.min())
    latent_ce = (latent_ce - latent_ce.min()) / (latent_ce.max() - latent_ce.min())
    latent_nc_from_ce = (latent_nc_from_ce - latent_nc_from_ce.min()) / (latent_nc_from_ce.max() - latent_nc_from_ce.min())

    channel = 3
    # set slice index to the slice of image with the highest sum of pixel values
    slice = np.argmax(ne[0, 0, :, :, :].sum(dim=1).sum(dim=1).cpu().detach().numpy())

    # plot a 3 row 2 column set of plots
    # top row is the ce image and the nc image
    # middle row is the ce latent space and the nc latent space
    # bottom row is the nc latent space from the ce model and the difference between the ce and nc latent spaces
    plt.switch_backend('TkAgg')
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].imshow(ne[0, 0, slice, :, :].T.cpu().detach().numpy())
    #add title
    axs[0, 0].set_title('NC')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(ce[0, 0, slice, :, :].T.cpu().detach().numpy())
    axs[0, 1].set_title('CE')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(latent_nc[channel:channel+3, slice, :, :].T.cpu().detach().numpy())
    axs[1, 0].set_title('NC latent')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(latent_ce[channel:channel+3, slice, :, :].T.cpu().detach().numpy())
    axs[1, 1].set_title('CE latent')
    axs[1, 1].axis('off')
    axs[2, 0].imshow(latent_nc_from_ce[channel:channel+3, slice, :, :].T.cpu().detach().numpy())
    axs[2, 0].set_title('NC latent from CE model')
    axs[2, 0].axis('off')
    axs[2, 1].imshow((latent_ce - latent_nc)[channel:channel+3, slice, :, :].T.cpu().detach().numpy())
    axs[2, 1].set_title('Difference between CE and NC latent')
    axs[2, 1].axis('off')

    abs_diff_latent = 0
    abs_diff_ce = 0
    abs_diff_output = 0
    abs_diff_ce_output = 0

    for output in zip(original_nc, original_ce,original_nc_from_ce):
        nc_l,ce_l,ne_from_ce_l = output
        if nc_l.shape[1] == 2:
            abs_diff_output += torch.abs(nc_l - ce_l).mean()
            abs_diff_ce_output += torch.abs(ne_from_ce_l - ce_l).mean()
        else:
            abs_diff_latent += torch.abs(nc_l - ce_l).mean()
            abs_diff_ce += torch.abs(ne_from_ce_l - ce_l).mean()

    print(f'Absolute difference between CE and NC latent spaces: {abs_diff_latent}')
    print(f'Absolute difference between CE and NC latent spaces from CE model: {abs_diff_ce}')
    print(f'Absolute difference between CE and NC output spaces: {abs_diff_output}')
    print(f'Absolute difference between CE and NC output spaces from CE model: {abs_diff_ce_output}')

    # print size of the average weight and bias in each model
    print(f'Average weight size in CE model: {torch.abs(torch.cat([p.view(-1) for p in ce_model.parameters()])).mean()}')
    print(f'Average weight size in NC model: {torch.abs(torch.cat([p.view(-1) for p in nc2ceCT_model.parameters()])).mean()}')

    plt.show()
