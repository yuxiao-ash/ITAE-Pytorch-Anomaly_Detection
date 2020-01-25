from data_tools.utils import AverageMeter, evaluate
import time
import datetime
import torch
import mmcv
import os
from tqdm import tqdm


def train_epoch(epoch_i, model, train_loader, criterion, optimizer, cfg, log, train_scheduler):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end_time = time.time()
    for i_batch, (tfs_image, org_image, label) in enumerate(train_loader):

        data_time.update(time.time() - end_time)
        curr_time = datetime.datetime.now()
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        tfs_image = tfs_image.cuda()
        org_image = org_image.cuda()
        restore_image = model(tfs_image)
        cost = criterion(restore_image, org_image)# debug the dtype of label image, need 'long'?
        model.zero_grad()
        cost.backward()
        optimizer.step()

        losses.update(cost.item(), tfs_image.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i_batch % cfg.show_frequent == 0:
            print('{0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {4}'.format(curr_time,
                                  epoch_i,
                                  i_batch + 1,
                                  len(train_loader),
                                  optimizer.param_groups[0]['lr'],
                                  batch_time=batch_time,
                                  data_time=data_time,
                                  loss=losses))
    if (epoch_i + 1) % 20 == 0:
        save_file_path = os.path.join(cfg.work_dir, 'model')
        mmcv.mkdir_or_exist(save_file_path)
        states = {
            'epoch': epoch_i + 1,
            'state_dict': model.state_dict()
        }
        torch.save(states, os.path.join(save_file_path, 'ITAE_model_epoch{}.pth'.format(epoch_i + 1)))

    train_scheduler.step()
    log.log_train(epoch_i + 1, losses.avg, optimizer)


def test_epoch(model, train4val_loader, test_loader, criterion, cfg, log, i=1):

    model.eval()
    t = cfg.transformations
    with torch.no_grad():
        print('evaluate on the train dataset for normal samples L1 error-------')
        nrm_err = torch.zeros(size=(len(train4val_loader.dataset), t), dtype=torch.float32).cuda()
        for i_batch, (tfs_images, org_image, label) in enumerate(tqdm(train4val_loader)):
            b, t, c, h, w = tfs_images.size()#c=1
            tfs_images = tfs_images.view(b*t, c, h, w).cuda()
            org_image = org_image.cuda()# b c h w
            restore_images = model(tfs_images)
            restore_images = restore_images.view(b, t, cfg.nc, h, w)
            for j in range(t):
                err = criterion(restore_images[:, j, :], org_image)# debug the shape of err
                nrm_err[i_batch*cfg.batchsize: i_batch*cfg.batchsize+err.size(0), j] = err.reshape(err.size(0))
        # print((nrm_err.size()))
        nrm_err = torch.mean(nrm_err, dim=0)# shape: t

        print('evaluate on the test dataset for normalized L1 error-------')
        abn_err = torch.zeros(size=(len(test_loader.dataset), t), dtype=torch.float32).cuda()
        abn_tgt = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.long).cuda()
        for i_batch, (tfs_images, org_image, label) in enumerate(tqdm(test_loader)):
            b, t, c, h, w = tfs_images.size()
            tfs_images = tfs_images.view(b*t, c, h, w).cuda()
            org_image = org_image.cuda()# b c h w
            restore_images = model(tfs_images)
            restore_images = restore_images.view(b, t, cfg.nc, h, w)
            abn_tgt[i_batch * cfg.batchsize: i_batch * cfg.batchsize + b] = label.reshape(b)
            for j in range(t):
                err = criterion(restore_images[:, j, :], org_image) / nrm_err[j]
                abn_err[i_batch * cfg.batchsize: i_batch * cfg.batchsize + err.size(0), j] = err.reshape(err.size(0))
        abn_err = torch.mean(abn_err, dim=1)# shape: len(test_loader.dataset)

        auroc = evaluate(abn_tgt, abn_err, metric='roc')

        print('the test dataset AUROC is {}%'.format(auroc * 100))

        if auroc > cfg.best_auroc:
            save_file_path = os.path.join(cfg.work_dir, 'model')
            mmcv.mkdir_or_exist(save_file_path)
            states = {
                'epoch': i,
                'state_dict': model.state_dict()
            }
            torch.save(states, os.path.join(save_file_path, 'best_model.pth'))
            cfg.best_auroc = auroc

    log.log_test(auroc, cfg.best_auroc)