# BaselineIR
### -Dependencies-
- python 3.6.9
- torch 1.10.1
- torchvision 0.11.2
- tensorboardX 2.1
### -Usage-
#### Prepare dataset:
Please ensure that the data organization matches the [code format for train & test](https://github.com/suiyizhao/BaselineIR/blob/master/src/datasets.py#:~:text=self.img_paths%20%3D%20sorted(glob.glob,%27/%27%20%20%2B%20mode%20%2B%20%27/sharp%27%20%2B%20%27/*/*.*%27)) or the [code format for infer](https://github.com/suiyizhao/BaselineIR/blob/master/src/datasets.py#:~:text=self.img_paths%20%3D%20sorted(glob.glob(data_source%20%2B%20%27/%27%20%2B%20%27test%27%20%2B%20%27/blurry%27%20%2B%20%27/*/*.*%27))).
#### Train:
```
python train.py --data_source /your/dataset/path --experiment your_experiment_name 
```
#### Test:
```
python test.py --data_source /your/dataset/path --model_path /your/model/path --experiment your_experiment_name
```
#### Infer:
```
python infer.py --data_source /your/dataset/path --model_path /your/model/path --experiment your_experiment_name --save_image
```
### -Other functions-
#### Reproducible training
```
# Manually modify the set_random_seed function by setting "deterministic=True"
set_random_seed(opt.seed, deterministic=True)
```
#### Parallel training
```
python train.py --data_source /your/dataset/path --experiment your_experiment_name --data_parallel
```
#### Fine-tuning using pretrained model
```
python train.py --data_source /your/dataset/path --experiment your_experiment_name --pretrained /pretrained/model/path
```
#### Continue training after interruptions
```
python train.py --data_source /your/dataset/path --experiment your_experiment_name --resume
```
