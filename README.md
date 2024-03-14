# BaselineIR
### -Platform-
- python 3.10.8
- torch 2.1.2
- torchvision 0.16.2
- tensorboard 2.15.1

### -Usage-
#### Configure the environment
```
git clone https://github.com/suiyizhao/BaselineIR.git
cd BaselineIR
pip install -r requirements.txt
cd src
```
#### Prepare dataset:
Please ensure that the data organization matches the [code format for train & test](https://github.com/suiyizhao/BaselineIR/blob/master/src/datasets.py#:~:text=self.img_paths%20%3D%20sorted(glob.glob,%27/%27%20%20%2B%20mode%20%2B%20%27/sharp%27%20%2B%20%27/*/*.*%27)) or the [code format for infer](https://github.com/suiyizhao/BaselineIR/blob/master/src/datasets.py#:~:text=self.img_paths%20%3D%20sorted(glob.glob(data_source%20%2B%20%27/%27%20%2B%20%27test%27%20%2B%20%27/blurry%27%20%2B%20%27/*/*.*%27))).
#### Train:
```
python train.py --data_source /your/dataset/path --experiment your_experiment_name
```
#### Test:
```
python test.py --data_source /your/dataset/path --experiment your_experiment_name --model_path /your/model/path --train_crop your_crop_size_in_training --save_image
```
#### Infer:
```
python infer.py --data_source /your/dataset/path --experiment your_experiment_name --model_path /your/model/path --train_crop your_crop_size_in_training --save_image
```
### -Other functions-
#### Debug
```
# During training, it is recommended to debug first to make sure the code is working properly
python train.py --data_source /your/dataset/path --experiment your_experiment_name --debug
```
#### Reproducible training
```
# Manually modify the set_random_seed (utils.py) function by setting "deterministic=True"
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
