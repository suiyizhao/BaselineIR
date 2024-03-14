import time
import torch
import pyiqa
import random

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from models import NAFNet, NAFNetLocal
from utils import *
from options import TrainOptions
from losses import LossCont, LossLPIPS, LossFFT
from datasets import PairedImgDataset

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = PairedImgDataset(data_source=opt.data_source, mode='train', crop=opt.crop)
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

print('validating data loading...')
val_dataset = PairedImgDataset(data_source=opt.data_source, mode='val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = NAFNet(img_channel=3, 
               width=opt.width, 
               middle_blk_num=opt.middle_blk_num, 
               enc_blk_nums=opt.enc_blk_nums, 
               dec_blk_nums=opt.dec_blk_nums
               ).cuda()

model_val = NAFNetLocal(img_channel=3, 
               width=opt.width, 
               middle_blk_num=opt.middle_blk_num, 
               enc_blk_nums=opt.enc_blk_nums, 
               dec_blk_nums=opt.dec_blk_nums,
               train_size=(1, 3, opt.crop, opt.crop)
               ).cuda()

if opt.data_parallel:
    model = nn.DataParallel(model)
print_para_num(model)

if opt.pretrained is not None:
    model.load_state_dict(torch.load(opt.pretrained))
    print('successfully loading pretrained model.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_lpips = LossLPIPS()
criterion_fft = LossFFT()

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.n_epochs, eta_min=opt.eta_min)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------') \
if not opt.debug else print('---------------------------------------- step 5/5 : debugging... ----------------------------------------------------')
def main():
    
    optimal = [0.]
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        print('Resume from epoch %d' % (start_epoch), optimal)
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch, optimal)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch, optimal)
        
    writer.close()
    
def train(epoch, optimal):
    model.train()
    
    max_iter = len(train_dataloader)
    
    iter_cont_meter = AverageMeter()
    iter_lpips_meter = AverageMeter()
    iter_fft_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (imgs, gts) in enumerate(train_dataloader):
        imgs, gts = imgs.cuda(), gts.cuda()
        cur_batch = imgs.shape[0]
        
        optimizer.zero_grad()
        preds = model(imgs)
        
        loss_cont = criterion_cont(preds, gts)
        loss_lpips = criterion_lpips(preds, gts)
        loss_fft = criterion_fft(preds, gts)
        loss = opt.lambda_cont * loss_cont + opt.lambda_lpips * loss_lpips + opt.lambda_fft * loss_fft
        
        loss.backward()
        optimizer.step()
        
        iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
        iter_lpips_meter.update(loss_lpips.item()*cur_batch, cur_batch)
        iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)
        
        if i == 0:
            save_image(torch.cat((imgs,preds.detach(),gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)
            
        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Loss_cont: {:.4f} Loss_lpips: {:.4f} Loss_fft: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, i + 1, max_iter, iter_cont_meter.average(), iter_lpips_meter.average(), iter_fft_meter.average(), iter_timer.timeit()))
            writer.add_scalar('Loss_cont', iter_cont_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_lpips', iter_lpips_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('Loss_fft', iter_fft_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            
            if opt.debug:
                opt.val_gap = 1
                break
                
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'optimal': optimal}, models_dir + '/latest.pth')
    scheduler.step()
    
def val(epoch, optimal):
    model.eval()
    
    model_val.load_state_dict(model.state_dict())
    model_val.eval()
    
    print(''); print('Validating...', end=' ')
    
    psnr_meter = AverageMeter()
    psnr_meter_val = AverageMeter()
    timer = Timer()
    
    for i, (imgs, gts) in enumerate(val_dataloader):
        imgs, gts = imgs.cuda(), gts.cuda()
        
        with torch.no_grad():
            preds = model(imgs)
            preds_val = model_val(imgs)
        
        preds = torch.clamp(preds, 0, 1)
        preds_val = torch.clamp(preds_val, 0, 1)
        
        psnr_meter.update(get_metrics(preds, gts), imgs.shape[0])
        psnr_meter_val.update(get_metrics(preds_val, gts), imgs.shape[0])
        
        if i == 0:
            if epoch == opt.val_gap:
                save_image(imgs, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_img.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
                save_image(gts, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_gt.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
            save_image(preds_val, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_restored.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
    
    print('Epoch[{:0>4}/{:0>4}] PSNR: {:.4f} PSNR-Local: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, psnr_meter.average(), psnr_meter_val.average(), timer.timeit())); print('')
    
    if optimal[0] < psnr_meter_val.average():
        optimal[0] = psnr_meter_val.average()
        torch.save(model.state_dict(), models_dir + '/optimal_{:.4f}_{:.4f}_epoch_{:0>4}.pth'.format(optimal[0], psnr_meter.average(), epoch))
        
    writer.add_scalar('psnr', psnr_meter_val.average(), epoch)
    
    torch.save(model.state_dict(), models_dir + '/epoch_{:0>4}.pth'.format(epoch))
    
if __name__ == '__main__':
    main()
    
