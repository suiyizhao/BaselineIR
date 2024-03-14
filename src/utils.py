import os
import time
import math
import torch
import random
import shutil
import numpy as np

from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def set_random_seed(seed, deterministic=False):
    
    '''
    function: Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_dir(results_dir, experiment, delete=True):
    
    '''
    prepare needed dirs.
    '''
    
    models_dir = os.path.join(results_dir, experiment, 'models')
    log_dir = os.path.join(results_dir, experiment, 'log')
    train_images_dir = os.path.join(results_dir, experiment, 'images', 'train')
    val_images_dir = os.path.join(results_dir, experiment, 'images', 'val')
    
    clean_dir(models_dir, delete=delete)
    clean_dir(log_dir, delete=delete)
    clean_dir(train_images_dir, delete=delete)
    clean_dir(val_images_dir, delete=delete)
    
    return models_dir, log_dir, train_images_dir, val_images_dir

def clean_dir(path, delete=False, contain=False):
    '''
    if delete is True: if path exist, then delete it's files and folders under it, if not, make it;
    if delete is False: if path not exist, make it.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete:
        delete_under(path, contain=contain)
        
def delete_under(path, contain=False):
    '''
    delete all files and folders under path
    :param path: Folder to be deleted
    :param contain: delete root or not
    '''
    if contain:
        shutil.rmtree(path)
    else:
        del_list = os.listdir(path)
        for f in del_list:
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def print_para_num(model):
    
    '''
    function: print the number of total parameters and trainable parameters
    '''
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inp_shape = (3, 256, 256)
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
    print('GMACs with shape ' + str(inp_shape) + ':', macs)
    print('total parameters: %d' % total_params)
    print('trainable parameters: %d' % total_trainable_params)
     
class AverageMeter(object):
    
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        
    def average(self, auto_reset=False):
        avg = self.sum / self.count
        
        if auto_reset:
            self.reset()
            
        return avg
    
class Timer(object):
    
    """
    Computes the times.
    """
    
    def __init__(self):
        self.last_timestamp = 0. # Can be used between two timeit()
        self.start()

    def start(self):
        self.time_begin = time.time()

    def timeit(self, auto_reset=True):
        
        times = time.time() - self.time_begin
        self.last_timestamp = times
        
        if auto_reset:
            self.start()
                   
        return times

class ETA(object):
    
    """
    Computes the eta.
    """
    
    def __init__(self, n_epochs, n_iterations, print_gap):
        
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.print_gap = print_gap
        
        self.speed = 0. # seconds per iteration
        
    def get_eta(self, cur_epoch, cur_iteration, seconds_per_print):
        elapsed_iterations = ((cur_epoch -1) * self.n_iterations) + cur_iteration - self.print_gap
        elapsed_time = elapsed_iterations * self.speed
        self.speed = (elapsed_time + seconds_per_print) / (elapsed_iterations + self.print_gap)
        eta = ((self.n_epochs * self.n_iterations) - elapsed_iterations) * self.speed
        days = int(eta // (60*60*24))
        hours = int((eta - (days * 60*60*24)) // (60*60))
        minutes = int((eta - (days * 60*60*24) - (hours * 60*60)) // (60))
        seconds = int((eta - (days * 60*60*24) - (hours * 60*60) - (minutes * 60)))
        
        return str(days) + ' days ' + str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
        

def get_metrics(tensor_image1, tensor_image2, psnr_only=True, reduction=False):
    
    '''
    function: given a batch tensor image pair, get the mean or sum psnr and ssim value.
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: two python value, e.g., psnr_value, ssim_value
    '''
    
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Excpetion('a batch tensor image pair should be given!')
        
    numpy_imgs = tensor2img(tensor_image1)
    numpy_gts = tensor2img(tensor_image2)
    psnr_value, ssim_value = 0., 0.
    batch_size = numpy_imgs.shape[0]
    for i in range(batch_size):
        if not psnr_only:
            ssim_value += structural_similarity(numpy_imgs[i],numpy_gts[i], multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        psnr_value += peak_signal_noise_ratio(numpy_imgs[i],numpy_gts[i])
        
    if reduction:
        psnr_value = psnr_value/batch_size
        ssim_value = ssim_value/batch_size
    
    if not psnr_only:  
        return psnr_value, ssim_value
    else:
        return psnr_value

def tensor2img(tensor_image):
    
    '''
    function: transform a tensor image to a numpy image
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: range:[0,255]    type:numpy.uint8         format:[b,h,w,c]  RGB
    '''
    
    tensor_image = tensor_image*255
    tensor_image = tensor_image.permute([0, 2, 3, 1])
    if tensor_image.device != 'cpu':
        tensor_image = tensor_image.cpu()
    numpy_image = np.uint8(tensor_image.numpy())
    return numpy_image

def split_img(x, h_chunk, w_chunk):
    x = torch.cat(x.chunk(h_chunk, dim=2), dim=0)
    x = torch.cat(x.chunk(w_chunk, dim=3), dim=0)
    return x

def cat_img(x, h_chunk, w_chunk):
    x = torch.cat(x.chunk(w_chunk, dim=0), dim=3)
    x = torch.cat(x.chunk(h_chunk, dim=0), dim=2)
    return x
