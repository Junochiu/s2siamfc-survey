from __future__ import absolute_import
import ipdb
import os
import argparse
from got10k.experiments import *

from siamfc.maml_backbones import AlexNet
from siamfc.maml_heads import SiamFC

# from siamfc.ssiamfc import TrackerSiamFC
# from siamfc.ssiamfc_onlineft import TrackerSiamFC
# from siamfc.siamfc_un_bt import TrackerSiamFC
# from siamfc.siamfc_un_bt_mul import TrackerSiamFC
# from siamfc.siamfc_linear import TrackerSiamFC
from siamfc.siamfc_weight_dropping import TrackerSiamFC, Net

if __name__ == '__main__':
    # =============================================================================
    #     for b in ['05', '15', '20']:
    #         net_path = './checkpoints/b03g%s/siamfc_alexnet_e50.pth'%b
    #         tracker = TrackerSiamFC(net_path=net_path, name='hyper_b03_g%s'%b)
    # =============================================================================
    #    tracker = TrackerSiamFC(net_path=net_path, name='eccv_best_linear')
    # ssiam_base

    '''
    model = Net(
        backbone=AlexNet(args=self.maml_args),
        head=SiamFC(args=self.maml_args))
    self.load_pretrain()

    if torch.cuda.is_available():
        self.state_dict = torch.load(filepath, map_location="cuda:0")
    else:
        self.state_dict = torch.load(filepath, map_location="cpu")
    # model weight parsing
    oldkey = []
    with torch.no_grad():
        self.tmp_model = self.state_dict.copy()
        for key in self.tmp_model.keys():
            key_split = key.split(".")
            if key_split[0] != 'head':
                if key_split[2] == '0':
                    key_split[2] = "conv"
                elif key_split[2] == '1':
                    key_split[2] = "norm_layer"
                str = "."
                new_key = str.join(key_split)
                self.state_dict[new_key] = self.tmp_model[key]
                oldkey.append(key)
        for key in oldkey:
            del self.state_dict[key]
        self.model.load_state_dict(self.state_dict, strict=False)
    '''

# =============================================================================
#         root_dir = 'E:/VID_val_100/Data'
#         VID_exp = ExperimentVID_GOT(root_dir)
#         VID_exp.run(tracker, visualize=False)
#         VID_exp.report([tracker.name])
# =============================================================================

'''
root_dir = '../dataset/VOT2016'
VOT16_exp = ExperimentVOT(root_dir, version=2016, experiments='supervised', read_image=False)
VOT16_exp.run(tracker, visualize=False)
#        p, ious, fail = VOT16_exp.report([tracker.name])
'''
# testing pretrain
# =============================================================================
#for idx in range(30, 49):
#    net_path = './pretrain/eccv_best/siamfc_alexnet_e{}.pth'.format(idx)
#    tracker = TrackerSiamFC(net_path=net_path, name='testing_e{}'.format(idx))
#    root_dir = '../dataset/VOT2018'
#    VOT_exp = ExperimentVOT(root_dir, version=2018, experiments='supervised', read_image=False)
#    VOT_exp.run(tracker, visualize=False)
# =============================================================================

# parsing parameter
parser = argparse.ArgumentParser()
parser.add_argument("--test_name")
parser.add_argument("--testing_frame",type=int)
parser.add_argument("--update_times",type=int)
arg = parser.parse_args()



# testing pretrain with initial update
# =============================================================================
net_path = './pretrain/eccv_best/siamfc_alexnet_e41.pth'
tracker = TrackerSiamFC(net_path=net_path, name=arg.test_name, testing_param=arg)
root_dir = '../dataset/VOT2018'
VOT_exp = ExperimentVOT(root_dir, version=2018, experiments='maml', read_image=False)
VOT_exp.run(tracker, visualize=False)
# =============================================================================

# testing maml trained model
# =============================================================================
#for i in range(0,3):
#    net_path = 'checkpoints/maml_rdn_query_pair_maml_optim_sche/{}.pth'.format(i)
#    tracker = TrackerSiamFC(net_path=net_path, name='pretrain_w_transform_o_resize_maml_optim_sche{}'.format(i))
#    root_dir = '../dataset/VOT2018'
#    VOT_exp = ExperimentVOT(root_dir, version=2018, experiments='maml', read_image=False)
#    VOT_exp.run(tracker, visualize=False)
# =============================================================================
"""
root_dir = 'D:/UDT_pytorch/track/dataset/OTB2015'
OTB_exp = ExperimentOTB(root_dir, version=2015)
OTB_exp.run(tracker, visualize=False)
OTB_exp.report([tracker.name])
"""
#    root_dir = 'D:/UDT_pytorch/track/dataset/OTB2013'
#    OTB_exp = ExperimentOTB(root_dir, version=2013)
#    OTB_exp.run(tracker, visualize=False)
#    OTB_exp.report([tracker.name])
