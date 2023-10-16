import torch
import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--seed", type=int, default=42, help="random seed")
        self.parser.add_argument("--resume", action='store_true', help="if specified, resume the training")
        self.parser.add_argument("--results_dir", type=str, default='../results', help="path of saving models, images, log files")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/5 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        self.parser.add_argument("--train_bs", type=int, default=4, help="size of the training batches (train_bs per GPU)")
        self.parser.add_argument("--val_bs", type=int, default=2, help="size of the validating batches (val_bs per GPU)")
        self.parser.add_argument("--crop", type=int, default=384, help="image size after cropping")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/5 : model defining... ------------------------------------------------
        self.parser.add_argument("--data_parallel", action='store_true', help="if specified, training by data paralleling")
        self.parser.add_argument("--pretrained", type=str, default=None, help="pretrained model path")
        self.parser.add_argument("--num_res", type=int, default=8, help="number of resblocks after each convolution")
        self.parser.add_argument("--base_channel", type=int, default=32, help="number of output channels for first convolution")
        
        # ---------------------------------------- step 4/5 : requisites defining... ------------------------------------------------
        self.parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        self.parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
        
        # ---------------------------------------- step 5/5 : training... ------------------------------------------------
        self.parser.add_argument("--print_gap", type=int, default=40, help="the gap between two print operations, in iteration")
        self.parser.add_argument("--val_gap", type=int, default=40, help="the gap between two validations, also the gap between two saving operation, in epoch")
    
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if opt.data_parallel:
            opt.train_bs = opt.train_bs * torch.cuda.device_count()
            opt.val_bs = torch.cuda.device_count()
            opt.num_workers = opt.num_workers * torch.cuda.device_count()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--outputs_dir", type=str, default='../outputs', help="path of saving images")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/4 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        
        # ---------------------------------------- step 3/4 : model defining... ------------------------------------------------
        self.parser.add_argument("--model_path", type=str, default=None, required=True, help="pretrained model path")
        
        # ---------------------------------------- step 4/4 : testing... ------------------------------------------------
        self.parser.add_argument("--save_image", action='store_true', help="if specified, save image when testing")
        
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')
        
