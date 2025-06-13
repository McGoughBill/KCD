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

from KCD.ne2ceCT.dataloader_contrastprediction import CrossPhaseDataset
from KCD.ne2ceCT.loss_function_seg import DomainAdaptationLoss as Loss
from KCD.ne2ceCT.lr_scheduler import LinearWarmupExponentialDecayScheduler
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import sys
import gc
from KCD.Segmentation.Inference import infer_network


def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_ClassHead_PowerPool_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

def collate_fn(batch):
    ne, ce, diff = zip(*batch)
    ne = torch.cat(ne, dim=0)
    ce = torch.cat(ce, dim=0)
    diff = torch.cat(diff, dim=0)
    #go through each patch in ne; if 98% of the patch is == -2, remove it
    fg_slices = torch.where(ne != -2, 1, 0)
    fg_slices = fg_slices.sum(dim=(1, 2, 3)) > (ne.shape[1] * ne.shape[2] * ne.shape[3] * 0.02)
    ne = ne[fg_slices]
    ce = ce[fg_slices]
    diff = diff[fg_slices]

    if ne.shape[0] == 0:
        return torch.zeros((1, 1, ne.shape[1], ne.shape[2], ne.shape[3]), device=ne.device).float(), \
                torch.zeros((1, 1, ce.shape[1], ce.shape[2], ce.shape[3]), device=ce.device).float(), \
                torch.zeros((1, 1, diff.shape[1], diff.shape[2], diff.shape[3]), device=diff.device).float()

    #shuffle order using indexing to prevent batch_norm issues
    shuffle_indices = torch.randperm(ne.shape[0])
    return ne[shuffle_indices].unsqueeze(1), ce[shuffle_indices].unsqueeze(1), diff[shuffle_indices].unsqueeze(1)

def parse_option():
    parser = argparse.ArgumentParser('ne2ceCT pretraining script', add_help=False)

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--l2_weight', type=float, default=0, help='l2 weight')
    parser.add_argument('--cce_weight', type=float, default=0, help='cce weight')
    parser.add_argument('--cos_weight', type=float, default=0, help='cosine similarity weight')
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

    l2_weight = args.l2_weight
    cce_weight = args.cce_weight
    cos_weight = args.cos_weight
    lr_max = args.lr_max
    loss_gamma = args.loss_gamma
    batch_size = args.batch_size
    epochs = args.epochs
    out_dim = args.out_dim
    spacing = args.spacing
    workers = args.workers
    world_size = args.world_size



    dataset_ne_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/masked_final_pairedpretrain_nc/2mm_canceronly_customPP/')
    dataset_ce_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed/masked_final_pairedpretrain_ce/2mm_canceronly_customPP/')

    assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
    home_path = os.path.join(os.environ['OV_DATA_BASE'], 'ne2ceCT', 'justchecking_{}_cancer_ckpt_delay'.format(spacing))
    ne2ceCT_path = os.path.join(home_path,
                                'l2-{}_cce{}_cos{}_{}lr_{}lg_{}bs_{}ep_{}gpus'.format(
                                                                                   l2_weight,
                                                                                   cce_weight,
                                                                                   cos_weight,
                                                                                   lr_max,
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers,
                            collate_fn=collate_fn)

    mod_path = '/rds/project/rds-sDuALntK11g/trained_models/masked_kits_add_nooverlap_v2_small/2mm_canceronly_customPP/6_tot_ch_talkative_cls1/fold_0/network_weights'
    model_T = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).cuda(args.gpu)
    model_T.load_state_dict(torch.load(mod_path,map_location='cuda:{}'.format(args.gpu)))
    model_S = get_3d_featureoutput_unet(1, out_dim, 6, filters=32, filters_max=1024).cuda(args.gpu)
    model_S.load_state_dict(torch.load(mod_path,map_location='cuda:{}'.format(args.gpu)))

    model_S = infer_network.UNet_regressivehead_wrapper(model_S, 6, start_filters=32).cuda(args.gpu)

    model_T.eval()
    model_S.train()

    optimizer = torch.optim.Adam(model_S.parameters(), lr=lr_max / 20, betas=(0.95, 0.9), eps=1e-08,
                                 weight_decay=0.000001)
    scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=epochs/4, lr_max=lr_max,
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
        try:
            delay_val_loss = checkpoint['delay_val_loss']
        except:
            delay_val_loss = 1e6
        if args.rank == 0:
            print('Loaded model from checkpoint')

    else:
        train_losses = []
        val_losses = []
        min_val_loss = 1e6
        start_epoch = 0
        delay_val_loss = 1e6

    loss_func_late = Loss(l2_weight=l2_weight,cce_weight=cce_weight, cos_weight=cos_weight).cuda(args.gpu)
    loss_func_early = Loss(l2_weight=0,cce_weight=cce_weight, cos_weight=0).cuda(args.gpu)
    delay = epochs/4

    weight = 0.95
    running_loss = None
    print('\nStarting training\n')
    for epoch in range(start_epoch, epochs):
        dataloader.dataset.is_train = True
        model_S.train()
        for i, (ne, ce, diff) in enumerate(dataloader):
            sys.stdout.flush()
            # forward
            ne = ne.cuda(args.gpu)
            ce = ce.cuda(args.gpu)
            diff = diff.cuda(args.gpu)
            with torch.no_grad():
                T_loglist,_, _, cls_latent = model_T(ce)
            S_loglist, cls_latentpred,predicted_enhancement = model_S(ne)

            if epoch < delay:
                loss_func = loss_func_early
            else:
                loss_func = loss_func_late

            loss,cce_term,mse_term,cls_term = loss_func(S_loglist,cls_latentpred,predicted_enhancement,
                                                        T_loglist,cls_latent,diff)
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, CCE: {:.4f}, CLS: {:.4f}, MSE: {:.4f}, LR: {:.4f}'.format(
                    epoch + 1, epochs, i + 1, len(dataloader), running_loss, cce_term, cls_term, mse_term, scheduler.get_last_lr()[0]))
            # print like above but with line replacement

            train_losses.append(running_loss)
            scheduler.step()
        if args.rank == 0:
            train_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}, CCE: {cce_term:.4E}, CLS: {cls_term:.4E}, MSE: {mse_term:.4E}\n'
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
                for i, (ne, ce,diff) in enumerate(dataloader):
                    ne = ne.cuda(args.gpu)
                    ce = ce.cuda(args.gpu)
                    diff = diff.cuda(args.gpu)
                    with torch.no_grad():
                        T_loglist,_, _, cls_latent = model_T(ce)
                    S_loglist, cls_latentpred, predicted_enhancement = model_S(ne)
                    loss, cce_term, mse_term, cls_term = loss_func(S_loglist, cls_latentpred, predicted_enhancement,
                                                                   T_loglist, cls_latent, diff)
                    val_losses_list.append([loss.item(), cce_term, cls_term, mse_term])
                val_loss_list = np.array(val_losses_list)
                #average each term over all batches
                running_val_losses = val_loss_list.mean(axis=0)
            val_string = f'Val Loss: {running_val_losses[0]:.4E}, CCE: {running_val_losses[1]:.4E}, CLS: {running_val_losses[2]:.4E}, MSE: {running_val_losses[3]:.4E}\n'
            val_losses.append(running_val_losses.tolist())
            with open(log_path, 'a') as f:
                f.write(val_string)
            if (running_val_losses[0] < min_val_loss and epoch < delay) or (running_val_losses[0] < delay_val_loss and epoch >= delay):
                if epoch >= delay:
                    delay_val_loss = running_val_losses[0]
                else:
                    min_val_loss = running_val_losses[0]
                torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
                print('Model saved')
                with open(log_path, 'a') as f:
                    f.write('Model saved\n')

                # save the last image patch, seg output, predicted enhancement, seg label, and diff as nifti files
                epoch_patch_save_path = os.path.join(patch_save_path, 'epoch_{}'.format(epoch))
                if not os.path.exists(epoch_patch_save_path):
                    os.makedirs(epoch_patch_save_path)

                ne = nib.Nifti1Image(ne.cpu().detach().numpy()[-1].squeeze(), np.eye(4))
                ce = nib.Nifti1Image(ce.cpu().detach().numpy()[-1].squeeze(), np.eye(4))

                seg_output = nib.Nifti1Image(torch.argmax(S_loglist[0],dim=1).cpu().detach().numpy().astype(np.int16)[-1].squeeze(), np.eye(4),dtype=np.int16)
                predicted_enhancement = nib.Nifti1Image(predicted_enhancement.cpu().detach().numpy()[-1].squeeze(), np.eye(4))
                seg_label = nib.Nifti1Image(torch.argmax(T_loglist[0],dim=1).cpu().detach().numpy().astype(np.int16)[-1].squeeze(), np.eye(4),dtype=np.int16)
                diff = nib.Nifti1Image(diff.cpu().detach().numpy()[-1].squeeze(), np.eye(4))

                nib.save(ne, os.path.join(epoch_patch_save_path, 'ne.nii.gz'))
                nib.save(ce, os.path.join(epoch_patch_save_path, 'ce.nii.gz'))
                nib.save(seg_output, os.path.join(epoch_patch_save_path, 'seg_output.nii.gz'))
                nib.save(predicted_enhancement, os.path.join(epoch_patch_save_path, 'predicted_enhancement.nii.gz'))
                nib.save(seg_label, os.path.join(epoch_patch_save_path, 'seg_label.nii.gz'))
                nib.save(diff, os.path.join(epoch_patch_save_path, 'diff.nii.gz'))


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
            ax.plot(np.array(val_losses)[:, 0], label='Total Loss')
            ax.plot(np.array(val_losses)[:, 1], label='CCE')
            ax.plot(np.array(val_losses)[:, 2], label='CLS')
            ax.plot(np.array(val_losses)[:, 3], label='MSE')
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
