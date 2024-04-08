import torch
import os
os.environ['OV_DATA_BASE'] ='/bask/projects/p/phwq4930-renal-canc/data/seg_data'
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import torch.distributed as dist
from datetime import timedelta

from KCD.ne2ceCT.dataloader import CrossPhaseDataset
from KCD.ne2ceCT.loss_function import PyramidalLatentSimilarityLoss
from KCD.Segmentation.Inference import infer_network
from KCD.ne2ceCT.lr_scheduler import LinearWarmupExponentialDecayScheduler
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import sys

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

def parse_option():
    parser = argparse.ArgumentParser('ne2ceCT pretraining script', add_help=False)

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--l1_weight', type=float, default=25, help='l1 weight')
    parser.add_argument('--l2_weight', type=float, default=0, help='l2 weight')
    parser.add_argument('--cce_weight', type=float, default=0, help='cce weight')
    parser.add_argument('--cos_weight', type=float, default=0, help='cosine similarity weight')
    parser.add_argument('--var_loss_weight', type=float, default=0.1, help='variance loss weight')
    parser.add_argument('--cov_loss_weight', type=float, default=0.1, help='covariance loss weight')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr_max', type=float, default=0.00005, help='max learning rate')
    parser.add_argument('--loss_gamma', type=float, default=0.99999, help='gamma for learning rate scheduler')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--distributed', action='store_true', help='start distributed training')
    parser.add_argument('--workers', default=16, type=int, help='number of workers')
    parser.add_argument('--ngpus_per_node', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--out_dim', default=2, type=int, help='output dimension')
    parser.add_argument('--spacing', default=2, type=int, help='spacing')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')

    args = parser.parse_args()
    print('args parsed')

    return args

def main():

    args = parse_option()
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu,args):
    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank,
                                timeout=timedelta(300))
    torch.cuda.set_device(args.gpu)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
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
    world_size = args.world_size

    dataset_ne_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/cropped_tcia_ncct/tcia_cmd_2_ncct/')
    dataset_ce_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/cropped_tcia_cect/tcia_cmd_2_cect/')

    assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
    ne2ceCT_path = os.path.join(os.environ['OV_DATA_BASE'], 'ne2ceCT', 'cropped_tcia_{}_cancer'.format(spacing),
                                'l1{}_l2{}_var{}_cov{}_cce{}_{}lr_{}lg_{}bs_{}ep_{}gpus'.format(l1_weight,
                                                                                   l2_weight,
                                                                                   var_loss_weight,
                                                                                   cov_loss_weight,
                                                                                   cce_weight,
                                                                                   lr_max,
                                                                                   loss_gamma,
                                                                                   batch_size,
                                                                                   epochs,
                                                                                   world_size))
    log_path = os.path.join(ne2ceCT_path, 'logs.txt')
    if args.rank == 0:
        if not os.path.exists(ne2ceCT_path):
            os.makedirs(ne2ceCT_path)
        if os.path.exists(log_path):
            os.remove(log_path)

    dataset = CrossPhaseDataset(os.path.join(dataset_ne_path, 'images'), os.path.join(dataset_ce_path, 'images'),
                                device='cuda',
                                is_train=True, patches_per_case=40, patch_size=64)
    if args.rank == 0:
        print(dataset.cases)

    dataset.apply_foldsplit(split_ratio=0.9)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    model_T = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).cuda(args.gpu)
    model_T.load_state_dict(torch.load(
        '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/kits23_nooverlap/2mm_binary_canceronly/6,3x3x3,32_justcancer/fold_0/network_weights'))
    model_S = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).cuda(args.gpu)
    model_S.load_state_dict(torch.load(
        '/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/kits23_nooverlap/2mm_binary_canceronly/6,3x3x3,32_justcancer/fold_0/network_weights'))

    model_T.eval()
    model_S.train()

    optimizer = torch.optim.Adam(model_S.parameters(), lr=lr_max / 20, betas=(0.95, 0.9), eps=1e-08,
                                 weight_decay=0.0001)
    scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=500, lr_max=lr_max,
                                                      decay_rate=loss_gamma)

    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=loss_gamma)

    loss_func = PyramidalLatentSimilarityLoss(l1_weight=l1_weight, l2_weight=l2_weight,
                                              cce_weight=cce_weight, cos_weight=cos_weight,
                                              var_loss_weight=var_loss_weight, cov_loss_weight=cov_loss_weight,
                                              out_dim=out_dim).cuda(args.gpu)
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
            ne = ne.cuda(args.gpu)
            ce = ce.cuda(args.gpu)
            with torch.no_grad():
                T = model_T(ce)
            S = model_S(ne)
            loss = loss_func(T, S)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_S.parameters(), 1)
            # update weights
            optimizer.step()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = weight * running_loss + (1 - weight) * loss.item()

            if args.rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, len(dataloader),
                                                                         loss.item()))
            # print like above but with line replacement

            losses.append(running_loss)
            scheduler.step()
        if args.rank == 0:
            train_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}\n'
            with open(log_path, 'a') as f:
                f.write(train_string)

        dataloader.dataset.is_train = False
        with torch.no_grad():
            model_S.eval()
            for i, (ne, ce) in enumerate(dataloader):
                ne = ne.cuda(args.gpu)
                ce = ce.cuda(args.gpu)
                T = model_T(ce)
                S = model_S(ne)
                loss = loss_func(T, S)
                if running_val_loss is None:
                    running_val_loss = loss.item()
                else:
                    running_val_loss = weight * running_val_loss + (1 - weight) * loss.item()
        if args.rank == 0:
            val_string = f'Val Loss: {running_val_loss:.4E}\n'
            with open(log_path, 'a') as f:
                f.write(val_string)
        dataloader.dataset.reset_patch_count()
        if args.rank == 0:
            if running_val_loss < min_val_loss:
                min_val_loss = running_val_loss
                torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
                print('Model saved')
                with open(log_path, 'a') as f:
                    f.write('Model saved\n')
        if args.rank == 0:
            with open(log_path, 'a') as f:
                f.write('\n')
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(losses)
            plt.savefig(os.path.join(ne2ceCT_path, 'losses.png'))
        dist.barrier()


if __name__ == '__main__':
    main()
