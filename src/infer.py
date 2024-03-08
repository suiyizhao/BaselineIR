import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from options import TestOptions
from models import NAFNet, NAFNetLocal
from datasets import SingleImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer'
clean_dir(image_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('inferring data loading...')
infer_dataset = SingleImgDataset(data_source=opt.data_source)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))

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
    
    time_meter = AverageMeter()
    
    for i, (img, name) in enumerate(infer_dataloader):
        img = img.cuda()

        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            times = time.time() - start_time

        pred_clip = torch.clamp(pred, 0, 1)

        time_meter.update(times, 1)
        
        print('Iteration[' + str(i+1) + '/' + str(len(infer_dataset)) + ']' + '  Processing image... ' + name[0] + '  Time ' + str(times))
            
        if opt.save_image:
            save_image(pred_clip, image_dir + '/' + name[0])
    
    print('Average Time: {:.4f}'.format(time_meter.average()))
        
if __name__ == '__main__':
    main()
    
