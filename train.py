from __future__ import absolute_import

import os
from time import localtime, strftime


from datasets.vid import ImageNetVID
from datasets.vot import VOT
#from datasets.coco import Coco
import random

#from siamfc.ssiamfc import TrackerSiamFC 
#from siamfc.siamfc_stn import TrackerSiamFC 
from siamfc.siamfc_weight_dropping import TrackerSiamFC
#from siamfc.siamfc_weight_dropping_maml import TrackerSiamFC


import ipdb
import torch
import numpy as np
torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
np.random.seed(123456) #numpy
random.seed(123456) #random and transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# merge two dict
def merge_two_dicts(x,y):
    seq = []
    for item in x:
        seq.append(item)
    for item in y:
        seq.append(item)
    return seq

if __name__ == '__main__':
    
    save_dir = './checkpoints/'
    save_path = os.path.join(save_dir, 'codecleaning_IV')

# =============================================================================    
#    neg_dir = ['./seq2neg_dict.json', './cluster_dict.json']
#    root_dir = '../dataset/ILSVRC2015'      #Dataset path
#    seqs = ImageNetVID(root_dir, subset=['train'], neg_dir=neg_dir[0])
#    seq_dict = seqs.seq_dict
#    print(len(seqs))
# =============================================================================
    
# =============================================================================
#    root_dir = '../dataset/VOT2018'
#    seqs = VOT(root_dir, subset=['train'])
#    print(len(seqs))
# =============================================================================

# =============================================================================
#     root_dir = 'E:\SiamMask\data\coco'
#     seqs = Coco(root_dir, subset=['train'])
# =============================================================================
    
# =============================================================================
#     save_dir = './checkpoints/eccv_rot_rcrop_mask_0515m0sig_got'
#     root_dir = 'E:/GOT10K'
#     seqs = GOT10k(root_dir, subset='train')
# =============================================================================

# =============================================================================    
    neg_dir = ['./seq2neg_dict.json', './cluster_dict.json']
    root_dir = '../dataset/ILSVRC2015'      #Dataset path
    seqs1 = ImageNetVID(root_dir, subset=['train'], neg_dir=neg_dir[0])
    seq_dict = seqs1.seq_dict
    root_dir = '../dataset/VOT2018'
    seqs2 = VOT(root_dir, subset=['train'])
    seqs = merge_two_dicts(seqs2,seqs1)
    seqs_dict = {"seq":seqs, "seq_dict":seq_dict}
# =============================================================================


# =============================================================================
#    mode = ['supervised', 'self-supervised']
#    tracker = TrackerSiamFC(loss_setting=[0.5, 2.0, 0])
#    tracker.train_over(seqs, supervised=mode[1], save_dir=save_path)
#    print(strftime("%Y-%m-%d %H:%M:%neg_dirS", localtime()))
# =============================================================================
    mode = ['supervised', 'self-supervised']
    net_path = os.path.join('pretrain','eccv_best','siamfc_alexnet_e42.pth')
    tracker = TrackerSiamFC(loss_setting=[0.5, 2.0, 0])
    tracker.train_over(seqs_dict, supervised=mode[1], save_dir=save_path)
    #tracker.meta_train_over(seqs, supervised=mode[1], save_dir=save_path)
    print(strftime("%Y-%m-%d %H:%M:%neg_dirS", localtime()))
# =============================================================================
