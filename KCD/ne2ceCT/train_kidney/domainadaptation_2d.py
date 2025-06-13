import numpy as np
import torch
import os
os.environ['OV_DATA_BASE'] ="/rds/project/rds-sDuALntK11g"
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import torch.distributed as dist
from datetime import timedelta
import nibabel as nib

from KCD.ne2ceCT.dataloader_2d import CrossPhaseDataset
from KCD.ne2ceCT.lr_scheduler import LinearWarmupExponentialDecayScheduler
import matplotlib.pyplot as plt
import sys
import gc
import torch.nn as nn


class latent_similarity_loss(torch.nn.Module):
    # this calculates the normalized cosine similarity between two latent vectors
    def __init__(self, weight=1.0):
        super(latent_similarity_loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def normalize(self, z):
        mag = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        return z / mag

    def forward(self, x, y):
        # x and y are the latent vectors
        x = self.normalize(x)
        y = self.normalize(y)
        return 1 - self.cos(x, y)
def collate_fn(batch):
    ne, ce = zip(*batch)
    ne = torch.cat(ne, dim=0)
    ce = torch.cat(ce, dim=0)
    #shuffle order using indexing to prevent batch_norm issues
    shuffle_indices = torch.randperm(ne.shape[0])
    return ne[shuffle_indices], ce[shuffle_indices]


def parse_option():
    parser = argparse.ArgumentParser('ne2ceCT pretraining script', add_help=False)

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
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
    parser.add_argument('--spacing', default=1, type=int, help='spacing')
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
    print('gpu', args.gpu)
    if args.distributed:
        print('distributed',torch.distributed.is_available())
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank,
                                timeout=timedelta(300))
    print('Dist is initialized',torch.distributed.is_initialized(),torch.distributed.is_nccl_available())
    print('Cuda enabled',torch.cuda.is_available())
    torch.cuda.set_device(args.gpu)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.epochs)

    lr_max = args.lr_max
    loss_gamma = args.loss_gamma
    batch_size = args.batch_size
    epochs = args.epochs
    spacing = args.spacing
    workers = args.workers
    world_size = args.world_size

    dataset_ne_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/masked_final_pairedpretrain_nc/2mm_canceronly_customPP/')
    dataset_ce_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/masked_final_pairedpretrain_ce/2mm_canceronly_customPP/')

    assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
    home_path = os.path.join(os.environ['OV_DATA_BASE'], 'ne2ceCT', '2d_featurealignment'.format(spacing))
    ne2ceCT_path = os.path.join(home_path,
                                '{}lr_{}lg_{}bs_{}ep_{}gpus'.format(lr_max,
                                                                                   loss_gamma,
                                                                                   batch_size,
                                                                                   epochs,
                                                                                   world_size))

    opt_save_loc = os.path.join(ne2ceCT_path, 'optimizer_state')

    log_path = os.path.join(ne2ceCT_path, 'logs.txt')
    patch_save_path = os.path.join(ne2ceCT_path, 'patches')
    if args.rank == 0:
        if not os.path.exists(ne2ceCT_path):
            os.makedirs(ne2ceCT_path)
        if os.path.exists(log_path) and not os.path.exists(opt_save_loc):
            os.remove(log_path)
        if os.path.exists(patch_save_path) and not os.path.exists(opt_save_loc):
            # delete all files in the directory
            for file in os.listdir(patch_save_path):
                os.remove(os.path.join(patch_save_path, file))
        elif not os.path.exists(patch_save_path):
            os.makedirs(patch_save_path)


    dataset = CrossPhaseDataset(dataset_ne_path, dataset_ce_path,
                                home_path,
                                device='cuda',
                                is_train=True, patches_per_case=20, patch_size=64)
    if args.rank == 0:
        print(dataset.cases)

    sys.stdout.flush()
    dataset.apply_foldsplit()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    mod_path = '/home/wcm23/rds/rds-early_rcc_ldct-HSfXDTkHyC0/data/seg_data/training_info/masked_kits_add_nooverlap_v2/split_0/fold_0/TileModel/model/TileModel_large_5_10_0.001'
    model_T = torch.load(mod_path, map_location='cuda:{}'.format(args.gpu))
    model_S = torch.load(mod_path, map_location='cuda:{}'.format(args.gpu))
    model_T.eval()
    model_S.train()

    optimizer = torch.optim.Adam(model_S.parameters(), lr=lr_max / 20, betas=(0.95, 0.9), eps=1e-08,
                                 weight_decay=0.000001)
    scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=500, lr_max=lr_max,
                                                      decay_rate=loss_gamma)

    if os.path.exists(opt_save_loc):
        checkpoint = torch.load(opt_save_loc)
        model_S.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        min_val_loss = checkpoint['min_val_loss']
        start_epoch = checkpoint['Epoch']+1
        if args.rank == 0:
            print('Loaded model from checkpoint')

    else:
        train_losses = []
        val_losses = []
        min_val_loss = 1e6
        start_epoch = 0

    loss_func_early = latent_similarity_loss()

    weight = 0.95
    running_loss = None
    print('\nStarting training\n')
    for epoch in range(start_epoch, epochs):
        dataloader.dataset.is_train = True
        model_S.train()
        for i, (ne, ce) in enumerate(dataloader):
            # forward
            ne = ne.cuda(args.gpu)
            ce = ce.cuda(args.gpu)
            with torch.no_grad():
                teacher_features = model_T.features(ce)
            student_features = model_S.features(ne)


            loss = loss_func_early(student_features, teacher_features)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch + 1, epochs, i + 1, len(dataloader), running_loss, scheduler.get_last_lr()[0]))
            # print like above but with line replacement

            train_losses.append(running_loss)
            scheduler.step()
        if args.rank == 0:
            train_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}\n'
            with open(log_path, 'a') as f:
                f.write(train_string)

        gc.collect()
        #clear cuda cache
        torch.cuda.empty_cache()

        dataloader.dataset.is_train = False
        if args.rank == 0:
            with torch.no_grad():
                model_S.eval()
                val_losses_list = []
                for i, (ne, ce) in enumerate(dataloader):
                    ne = ne.cuda(args.gpu)
                    ce = ce.cuda(args.gpu)
                    with torch.no_grad():
                        teacher_features_val = model_T.features(ce)
                        student_features_val = model_S.features(ne)
                    loss = loss_func_early(student_features_val, teacher_features_val)
                    val_losses_list.append(loss.item())
                val_loss_list = np.array(val_losses_list)
                #average each term over all batches
                running_val_loss =val_loss_list.mean(axis=0)
            val_string = f'Val losses at epoch {epoch + 1}: Total Loss: {running_val_loss:.4E}'
            val_losses.append(running_val_loss)
            with open(log_path, 'a') as f:
                f.write(val_string)
            if running_loss < min_val_loss:
                torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
                print('Model saved')
                with open(log_path, 'a') as f:
                    f.write('Model saved\n')

            with open(log_path, 'a') as f:
                f.write('\n')
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(train_losses)
            plt.savefig(os.path.join(ne2ceCT_path, 'train_losses.png'))
            plt.close(fig)

            #plot all val losses
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.array(val_losses), label='Total Loss')
            ax.legend()
            plt.savefig(os.path.join(ne2ceCT_path, 'val_losses.png'))
            plt.close(fig)

            # save optimizer state
            torch.save({'Epoch': epoch,
                        'model_state_dict': model_S.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'min_val_loss': min_val_loss}, opt_save_loc)



        dataloader.dataset.reset_patch_count()
        gc.collect()
        # clear cuda cache
        torch.cuda.empty_cache()
        dist.barrier()



if __name__ == '__main__':
    main()
