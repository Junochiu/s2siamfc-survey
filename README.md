# S2siamfc - finetuning survey
This repo contain the original code of S2siamfc, and modified versions with several different implements.
## The main structure of the project
```bash
.
├── README.md
├── siamfc
├── checkpoints
├── got10k
├── datasets
├── pretrain
├── results
├── cluster_dict.json
├── clustering.py
├── demo.py
├── eval.py
├── feature_extraction.py
├── feature_gap.pkl
├── inner_loop_optimizers.py
├── maml_train.py
├── requirements.txt
├── seq2neg_dict.json
├── set_up.sh
├── test.py
├── test.sh
├── train.py
├── train_sup.py
├── tree.sh
├── utils
├── vis.sh
└── vis_vot.py
 
```

```bash
.
├── __init__.py
├── backbones.py
├── datasets.py
├── googlenet.py
├── heads.py
├── img_ft.py
├── inner_loop_optimizers.py
├── losses.py
├── maml_backbones.py
├── maml_basicstructure.py
├── maml_heads.py
├── ops.py
├── resnet.py
├── siamfc.py
├── siamfc_linear.py
├── siamfc_maml_weight_dropping.py
├── siamfc_weight_dropping.py
├── ssiamfc.py
├── ssiamfc_onlineft.py
├── ssiamfc_onlineft_linear.py
├── temp.sh
├── transforms.py
└── vgg.py
```
