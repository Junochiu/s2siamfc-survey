# S2Siamfc - finetuning survey
This repo contain the original code of S2siamfc, and modified versions with several different implementations.
## Installation

## Testing the tracker


## Training the tracker
### S2Siamfc
1. set up dataset path and dataset sequence in ```S2simafc/train.py```
2. check the config in ```s2siamfc/siamfc/siamfc_weight_dropping.py```
3. ```python train.py```
4. model results will be save in ```s2siamfc/checkpoints```

### S2Siamfc + MAML
1. current dataset is set to ImageNetVID in ```S2simafc/maml_train.py   __init()__```
2. check the config for maml training process in ```s2siamfc/maml_train.py```
3. check the config for inner tracker in ```s2siamfc/siamfc/siamfc_maml_weight_dropping.py```
4. ```python maml_train.py```
5. model results will be save in ```s2siamfc/checkpoints```


## Main file structure
```bash
.
├── README.md
├── siamfc...................................main model and implementation
├── got10k...................................testing code with different datasets
├── datasets.................................dataset parsing
├── pretrain.................................placing pretrain models
├── requirements.txt
└── test.py / test.sh / maml_train.py / train.py
 
```
## Basic operations in ./siamfc
### File structure
* Most of the implemetation keypoints  of S2siamfc are in siamfc_weight_dropping (labeling, training steps, tracking steps)
* siamfc_maml_weight_dropping construct the maml inner loop using the maml version tracker(maml_basicstructure.py) and optimizer(inner_loop_optimizer).<br>
* The outer loop implementation can be find in ```../maml_train.py``` where the main flow of maml is constructed.

```bash
.
├── __init__.py
├── ops.py
├── datasets.py
├── losses.py
├── backbones.py.............................followings are code for S2siamfc implementation
├── googlenet.py
├── vgg.py
├── heads.py
├── img_ft.py
├── resnet.py
├── siamfc.py
├── siamfc_weight_dropping.py
├── maml_basicstructure.py...................followings are code for S2siamfc + MAML++ implementation 
├── maml_backbones.py 
├── maml_heads.py
├── siamfc_maml_weight_dropping.py
├── inner_loop_optimizers.py
└── transforms.py
```
