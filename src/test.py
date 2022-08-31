import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import UNet
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
model = UNet().cuda()

model.load_state_dict(torch.load(opt.model_path))
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()
    
    with open(txt_path, 'w') as f:
        for i, (imgs, gts) in enumerate(test_dataloader):
            imgs, gts = imgs.cuda(), gts.cuda()

            with torch.no_grad():
                start_time = time.time()
                preds = model(imgs)
                times = time.time() - start_time

            preds_clip = torch.clamp(preds, 0, 1)

            cur_psnr = get_metrics(preds_clip, gts)
            psnr_meter.update(cur_psnr, 1)
            time_meter.update(times, 1)

            print('Iteration[{:0>4}/{:0>4}] PSNR: {:.4f} Time: {:.4f}'.format(i+1, len(test_dataset), cur_psnr, times))
            f.write('Iteration[' + str(i+1) + '/' + str(len(test_dataset)) + ']' + ' PSNR: ' + str(cur_psnr) + ' Time: ' + str(times) + '\n')
            
            if opt.save_image:
                save_image(preds_clip, single_dir + '/' + str(i).zfill(4) + '.png')
                save_image(imgs, multiple_dir + '/' + str(i).zfill(4) + '_img.png')
                save_image(preds_clip, multiple_dir + '/' + str(i).zfill(4) + '_restored.png')
                save_image(gts, multiple_dir + '/' + str(i).zfill(4) + '_gt.png')
            
        print('Average: PSNR: {:.4f} Time: {:.4f}'.format(psnr_meter.average(), time_meter.average()))
        f.write('Average: PSNR: ' + str(psnr_meter.average()) + ' Time: ' + str(time_meter.average()) + '\n')
    
    print('Successfully save results to: ' + txt_path)
        
if __name__ == '__main__':
    main()
    