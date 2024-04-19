import torch
import os
os.environ['OV_DATA_BASE'] ='/rds/project/rds-YVbkxwUTQs4/zl_isc2024_McGough/data_path'
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex

import argparse
from datetime import timedelta

from KCD.ne2ceCT.dataloader import CrossPhaseDataset
from KCD.ne2ceCT.loss_function_reverse import PyramidalLatentSimilarityLoss
from KCD.Segmentation.Inference import infer_network
from KCD.ne2ceCT.lr_scheduler import LinearWarmupExponentialDecayScheduler
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import sys
from datetime import datetime

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

def parse_option():
    parser = argparse.ArgumentParser('ne2ceCT pretraining script', add_help=False)

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--l1_weight', type=float, default=20, help='l1 weight')
    parser.add_argument('--l2_weight', type=float, default=5, help='l2 weight')
    parser.add_argument('--cce_weight', type=float, default=0, help='cce weight')
    parser.add_argument('--cos_weight', type=float, default=0, help='cosine similarity weight')
    parser.add_argument('--var_loss_weight', type=float, default=1, help='variance loss weight')
    parser.add_argument('--cov_loss_weight', type=float, default=0.1, help='covariance loss weight')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr_max', type=float, default=0.00005, help='max learning rate')
    parser.add_argument('--loss_gamma', type=float, default=0.99999, help='gamma for learning rate scheduler')
    parser.add_argument('--out_dim', default=2, type=int, help='output dimension')
    parser.add_argument('--spacing', default=2, type=int, help='spacing')
    parser.add_argument('--workers', default=10, type=int, help='number of workers')


    args = parser.parse_args()

    return args

def main():

    args = parse_option()
    main_worker(args=args)

def main_worker(args):
    dev = 'xpu'
    print('Batch size is:', args.batch_size, 'epochs', args.epochs)

    l1_weight = args.l1_weight
    l2_weight = args.l2_weight
    cce_weight = args.cce_weight
    cos_weight = args.cos_weight
    var_loss_weight = args.var_loss_weight
    cov_loss_weight = args.cov_loss_weight
    lr_max = args.lr_max
    loss_gamma = args.loss_gamma
    batch_size = args.batch_size
    epochs = args.epochs
    out_dim = args.out_dim
    spacing = args.spacing
    workers = args.workers

    dataset_ne_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/coltea_nat/coltea_nat_2/')
    dataset_ce_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/coltea_art/coltea_art_2/')

    assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
    ne2ceCT_path = os.path.join(os.environ['OV_DATA_BASE'], 'ne2ceCT', 'profiling_dawn_coltea',
                                'dawn_profile_l1{}_l2{}_var{}_cov{}_cce{}_{}bs'.format(l1_weight,
                                                                                   l2_weight,
                                                                                   var_loss_weight,
                                                                                   cov_loss_weight,
                                                                                   cce_weight,
                                                                                   batch_size
                                                                                   ))

    log_path = os.path.join(ne2ceCT_path, 'logs.txt')
    if not os.path.exists(ne2ceCT_path):
        os.makedirs(ne2ceCT_path)
    if os.path.exists(log_path):
        os.remove(log_path)

    dataset = CrossPhaseDataset(os.path.join(dataset_ne_path, 'images'), os.path.join(dataset_ce_path, 'images'),
                                device=dev,
                                is_train=True, patches_per_case=100, patch_size=64)
    print(dataset.cases)

    dataset.apply_foldsplit(split_ratio=0.9)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    model_T = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).to(dev)
    model_T.load_state_dict(torch.load(
        '/rds/project/rds-YVbkxwUTQs4/zl_isc2024_McGough/dummy_teacher_model',map_location=dev))
    model_S = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).to(dev)
    model_S.load_state_dict(torch.load(
        '/rds/project/rds-YVbkxwUTQs4/zl_isc2024_McGough/dummy_teacher_model',map_location=dev))

    model_T.eval()
    model_S.train()


    optimizer = torch.optim.Adam(model_S.parameters(), lr=lr_max / 20, betas=(0.95, 0.9), eps=1e-08,
                                 weight_decay=0.000001)

    model_S.train()
    model_S, optimizer = ipex.optimize(model_S, optimizer=optimizer)

    scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=500, lr_max=lr_max,
                                                      decay_rate=loss_gamma)


    loss_func = PyramidalLatentSimilarityLoss(l1_weight=l1_weight, l2_weight=l2_weight,
                                              cce_weight=cce_weight, cos_weight=cos_weight,
                                              var_loss_weight=var_loss_weight, cov_loss_weight=cov_loss_weight,
                                              out_dim=out_dim).to(dev)
    losses = []
    weight = 0.95
    running_loss = None
    running_val_loss = None
    min_val_loss = 1e6
    for epoch in range(epochs):
        dataloader.dataset.is_train = True
        model_S.train()
        for i, (ne, ce) in enumerate(dataloader):
            # forward
            ne = ne.to(dev)
            ce = ce.to(dev)
            with torch.no_grad():
                T = model_T(ce)
            S = model_S(ne)
            loss = loss_func(T, S)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_S.parameters(), 1)
            # update weights
            optimizer.step()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = weight * running_loss + (1 - weight) * loss.item()

            losses.append(running_loss)
            scheduler.step()

        timestring = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        train_string = f'{timestring}: Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}\n'
        with open(log_path, 'a') as f:
            f.write(train_string)

        dataloader.dataset.is_train = False
        with torch.no_grad():
            model_S.eval()
            for i, (ne, ce) in enumerate(dataloader):
                ne = ne.to(dev)
                ce = ce.to(dev)
                T = model_T(ce)
                S = model_S(ne)
                loss = loss_func(T, S)
                if running_val_loss is None:
                    running_val_loss = loss.item()
                else:
                    running_val_loss = weight * running_val_loss + (1 - weight) * loss.item()

        val_string = f'Val Loss: {running_val_loss:.4E}\n'
        with open(log_path, 'a') as f:
            f.write(val_string)
        dataloader.dataset.reset_patch_count()
        if running_val_loss < min_val_loss:
            min_val_loss = running_val_loss
            torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
            print('Model saved')
        with open(log_path, 'a') as f:
                f.write('Model saved\n')

        with open(log_path, 'a') as f:
            f.write('\n')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(losses)
        plt.savefig(os.path.join(ne2ceCT_path, 'losses.png'))


if __name__ == '__main__':
    main()
