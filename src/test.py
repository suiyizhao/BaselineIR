import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import NAFNet, NAFNetLocal
from datasets import PairedImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

txt_path = opt.outputs_dir + '/' + opt.experiment + '/test' + '/' + opt.experiment + '.txt'
single_dir = opt.outputs_dir + '/' + opt.experiment + '/test/single'
multiple_dir = opt.outputs_dir + '/' + opt.experiment + '/test/multiple'
clean_dir(single_dir, delete=opt.save_image)
clean_dir(multiple_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('testing data loading...')
test_dataset = PairedImgDataset(data_source=opt.data_source, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{}'.format(len(test_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
if opt.train_crop is None:
    model = NAFNet(img_channel=3, 
               width=opt.width, 
               middle_blk_num=opt.middle_blk_num, 
               enc_blk_nums=opt.enc_blk_nums, 
               dec_blk_nums=opt.dec_blk_nums
               ).cuda()
else:
    model = NAFNetLocal(img_channel=3, 
                   width=opt.width, 
                   middle_blk_num=opt.middle_blk_num, 
                   enc_blk_nums=opt.enc_blk_nums, 
                   dec_blk_nums=opt.dec_blk_nums,
                   train_size=(1, 3, opt.train_crop, opt.train_crop)
                   ).cuda()

print_para_num(model)

model.load_state_dict(torch.load(opt.model_path))
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()
    
    with open(txt_path, 'w') as f:
        for i, (img, gt, name) in enumerate(test_dataloader):
            img, gt = img.cuda(), gt.cuda()

            with torch.no_grad():
                start_time = time.time()
                pred = model(img)
                times = time.time() - start_time

            pred_clip = torch.clamp(pred, 0, 1)

            cur_psnr = get_metrics(pred_clip, gt)
            psnr_meter.update(cur_psnr, 1)
            time_meter.update(times, 1)

            print('Iteration[' + str(i+1) + '/' + str(len(test_dataset)) + ']' + '  Processing image... ' + name[0] + '   PSNR: ' + str(cur_psnr) + '  Time ' + str(times))
            f.write('Iteration[' + str(i+1) + '/' + str(len(test_dataset)) + ']' + '  Processing image... ' + name[0] + '   PSNR: ' + str(cur_psnr) + ' Time: ' + str(times) + '\n')
            
            if opt.save_image:
                save_image(pred_clip, single_dir + '/' + name[0])
                save_image(img, multiple_dir + '/' + name[0].split('.')[0] + '_img.png')
                save_image(pred_clip, multiple_dir + '/' + name[0].split('.')[0] + '_restored.png')
                save_image(gt, multiple_dir + '/' + name[0].split('.')[0] + '_gt.png')
            
        print('Average: PSNR: {:.4f} Time: {:.4f}'.format(psnr_meter.average(), time_meter.average()))
        f.write('Average: PSNR: ' + str(psnr_meter.average()) + ' Time: ' + str(time_meter.average()) + '\n')
    
    print('Successfully save results to: ' + txt_path)
        
if __name__ == '__main__':
    main()
    
