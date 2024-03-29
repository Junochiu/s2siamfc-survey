from __future__ import absolute_import, division

import numpy as np
import os
import cv2
import json
from torch.utils.data import Dataset
from collections import OrderedDict
import ipdb

__all__ = ['Pair']

class Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1, supervised='supervised', neg=False, img_loader=None, gen_query=False):
        super(Pair, self).__init__()
        self.seqs = seqs['seq']
        print(len(self.seqs))
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.img_loader = img_loader
        self.indices = np.random.permutation(len(self.seqs))
        self.supervised = supervised
        self.rangeup = 10
        self.neg = neg
        self.gen_query = gen_query
        self.seq_dict = seqs['seq_dict']
        '''
        if self.neg:
            self.cluster_dict = json.load(open('./cluster_dict.json'), object_pairs_hook=OrderedDict)
        '''
        self.cluster_dict = json.load(open('./cluster_dict.json'), object_pairs_hook=OrderedDict)

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]
        ''' original code for ISLVRC 2015 
        if self.neg:
            print(self.seqs[index])
            img_files, seq_name, cluster_id = self.seqs[index] #[:2]
        else:
            img_files, seq_name = self.seqs[index] 
        
        '''

        if len(self.seqs[index])>2:
            img_files, seq_name, cluster_id = self.seqs[index] #[:2]
            self.neg = 0.2
        else:
            img_files, seq_name = self.seqs[index]
            self.neg = 0

        if self.supervised == 'self-supervised':
            neg = self.neg and self.neg > np.random.rand()
            if neg:
                #print("-----neg-----")
                random_fid_z = np.random.choice(len(img_files))

                cluster_z_list = self.cluster_dict[str(cluster_id)]
                random_vid_neg = np.random.choice(cluster_z_list)
                
                while random_vid_neg == seq_name:       #in case find the same seq
                    random_vid_neg = np.random.choice(cluster_z_list)
                
                seq_dir_neg, frames_neg, cluster_id_neg = self.seq_dict[random_vid_neg]
                img_files_neg = [os.path.join(seq_dir_neg, '%06d.JPEG' % f) for f in frames_neg]
                
                random_fid_neg = np.random.choice(len(img_files_neg))

# =============================================================================
#                 random_vid_neg = np.random.choice(len(self))
#                 random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
#                 
#                 while random_vid_neg == index:
#                     random_vid_neg = np.random.choice(len(self))
#                     random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
#                 img_files_neg, _, cluster_id_neg = self.seqs[random_vid_neg]
#                 
#                 #hard neg ver
#                 while random_vid_neg == index or cluster_id != cluster_id_neg:
#                     random_vid_neg = np.random.choice(len(self))
#                     random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
#                     img_files_neg, _, cluster_id_neg = self.seqs[random_vid_neg]  
#                     
#                 random_fid_neg = np.random.choice(len(img_files_neg))
# =============================================================================
                
#                random_fid1, random_fid2 = np.random.choice(a=len(img_files), size=2, replace=False)
                z = self.img_loader(img_files[random_fid_z])
                x = self.img_loader(img_files_neg[random_fid_neg])

# =============================================================================
#                 z = cv2.imread(img_files[random_fid_z], cv2.IMREAD_COLOR)
#                 x = cv2.imread(img_files_neg[random_fid_neg], cv2.IMREAD_COLOR)
#                 z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
#                 x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# =============================================================================

                imgz_h, imgz_w, _ = z.shape
                imgx_h, imgx_w, _ = x.shape
#                target_pos = [img_w//2, img_h//2]
#                target_sz = [img_w//6, img_h//6]

                target_sz_z = [imgz_w//np.random.randint(4, 9), imgz_h//np.random.randint(4, 9)]
                target_pos_z = [np.random.randint(target_sz_z[0], (imgz_w-target_sz_z[0])), np.random.randint(target_sz_z[1], (imgz_h-target_sz_z[1]))]

                target_sz_x = [imgx_w//np.random.randint(4, 9), imgx_h//np.random.randint(4, 9)]
                target_pos_x = [np.random.randint(target_sz_x[0], (imgx_w-target_sz_x[0])), np.random.randint(target_sz_x[1], (imgx_h-target_sz_x[1]))]

                ### for training
                box_z = self._cxy_wh_2_bbox(target_pos_z, target_sz_z)
                box_x = self._cxy_wh_2_bbox(target_pos_x, target_sz_x)
                
                ### for testing
                #box_z = [target_pos_z[0],target_pos_z[1],target_sz_z[0],target_sz_z[1]]
                #box_x = [target_pos_x[0],target_pos_x[1],target_sz_x[0],target_sz_x[1]]
                

######### create query set #########
                if self.gen_query:
                    random_fid_query_z = np.random.choice(len(img_files))
                    random_fid_query_x = np.random.choice(len(img_files_neg))

                    query_z = self.img_loader(img_files[random_fid_query_z])
                    query_x = self.img_loader(img_files_neg[random_fid_query_x])

                    imgqz_h, imgqz_w, _ = query_z.shape
                    imgqx_h, imgqx_w, _ = query_x.shape
#                   target_pos = [img_w//2, img_h//2]
#                   target_sz = [img_w//6, img_h//6]

                    target_sz_qz = [imgqz_w//np.random.randint(4, 9), imgqz_h//np.random.randint(4, 9)]
                    target_pos_qz = [np.random.randint(target_sz_qz[0], (imgqz_w-target_sz_qz[0])), np.random.randint(target_sz_qz[1], (imgqz_h-target_sz_qz[1]))]

                    target_sz_qx = [imgqx_w//np.random.randint(4, 9), imgqx_h//np.random.randint(4, 9)]
                    target_pos_qx = [np.random.randint(target_sz_qx[0], (imgqx_w-target_sz_qx[0])), np.random.randint(target_sz_qx[1], (imgqx_h-target_sz_qx[1]))]

                    box_qz = self._cxy_wh_2_bbox(target_pos_qz, target_sz_qz)
                    box_qx = self._cxy_wh_2_bbox(target_pos_qx, target_sz_qx)
                    
                    #while random_fid_neg == random_fid_query:
                    #    random_fid_query = np.random.choice(len(img_files_neg))
                    item = (z,x,box_z,box_x,query_z,query_x,box_qz,box_qx)
                    if self.transforms is not None:
                        item = self.transforms(*item)
                    return item + (neg,)
###### end of create query set ######

                item = (z, x, box_z, box_x)
                if self.transforms is not None:
                    item = self.transforms(*item)
                                
                return item + (neg, )            
            else:
                #print("-----not neg-----")
                random_fid = np.random.choice(range(1,len(img_files)))
# =============================================================================
#                 z = cv2.imread(img_files[random_fid], cv2.IMREAD_COLOR)
#     #            x = cv2.imread(img_files[random_fid], cv2.IMREAD_COLOR)
#                 z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
#     #            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# =============================================================================
                
                z = self.img_loader(img_files[random_fid])      #get RGB img
    
                img_h, img_w, _ = z.shape

                target_sz = [img_w//np.random.randint(4, 9), img_h//np.random.randint(4, 9)]
                target_pos = [np.random.randint(target_sz[0], (img_w-target_sz[0])), np.random.randint(target_sz[1], (img_h-target_sz[1]))]
                
                ### training phase
                box = self._cxy_wh_2_bbox(target_pos, target_sz)
                

                ### for testing
                #box = [target_pos[0],target_pos[1],target_sz[0],target_sz[1]]

######### create query set #########
                if self.gen_query:
                    random_fid_query = np.random.choice(len(img_files))
                    query_z = self.img_loader(img_files[random_fid_query])
                    imgqz_h, imgqz_w, _ = query_z.shape
                    target_sz_qz = [imgqz_w//np.random.randint(4, 9), imgqz_h//np.random.randint(4, 9)]
                    target_pos_qz = [np.random.randint(target_sz_qz[0], (imgqz_w-target_sz_qz[0])), np.random.randint(target_sz_qz[1], (imgqz_h-target_sz_qz[1]))]
                    box_qz = self._cxy_wh_2_bbox(target_pos_qz, target_sz_qz)
                    item = (z,z,box,box,query_z,query_z,box_qz,box_qz)
                    if self.transforms is not None:
                        item = self.transforms(*item)
                    return item + (neg,)

###### end of create query set ######

                item = (z,z,box,box)
                if self.transforms is not None:
                    item = self.transforms(*item)
                       
                return item + (neg, )
        elif self.supervised == 'feature':
            random_fid = np.random.choice(len(img_files))
            
            img_path = img_files[random_fid]
            z = self.img_loader(img_path)      #get RGB img

            img_h, img_w, _ = z.shape

            if self.transforms is not None:
                item = self.transforms(z)
                   
            return [item, img_path, seq_name]
    
    def __len__(self):
        return len(self.indices) * self.pairs_per_seq
    
    def _cxy_wh_2_bbox(self, cxy, wh):
        return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])
'''
    def get_random_pair(self, img, transforms):

        z = self.img_loader(img)  # get RGB img

        img_h, img_w, _ = z.shape

        target_sz = [img_w // np.random.randint(4, 9), img_h // np.random.randint(4, 9)]
        target_pos = [np.random.randint(target_sz[0], (img_w - target_sz[0])),
                      np.random.randint(target_sz[1], (img_h - target_sz[1]))]

        box = self._cxy_wh_2_bbox(target_pos, target_sz)

        item = (z, z, box, box)
        if transforms is not None:
            item = transforms(*item)
        ipdb.set_trace()
        return item
'''

class maml_Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1, supervised='supervised', neg=False, img_loader=None):
        super(maml_Pair, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.img_loader = img_loader
        self.indices = np.random.permutation(len(seqs))
        self.supervised = supervised
        self.rangeup = 10
        self.neg = neg
        if self.neg:
            self.cluster_dict = json.load(open('./cluster_dict.json'), object_pairs_hook=OrderedDict)

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        if self.neg:
            img_files, seq_name, cluster_id = self.seqs[index]  # [:2]
        else:
            img_files, seq_name = self.seqs[index]

        if self.supervised == 'self-supervised':
            neg = self.neg and self.neg > np.random.rand()
            if neg:
                random_fid_z = np.random.choice(len(img_files))

                cluster_z_list = self.cluster_dict[str(cluster_id)]
                random_vid_neg = np.random.choice(cluster_z_list)

                while random_vid_neg == seq_name:  # in case find the same seq
                    random_vid_neg = np.random.choice(cluster_z_list)

                seq_dir_neg, frames_neg, cluster_id_neg = self.seqs.seq_dict[random_vid_neg]
                img_files_neg = [os.path.join(seq_dir_neg, '%06d.JPEG' % f) for f in frames_neg]

                random_fid_neg = np.random.choice(len(img_files_neg))

                # =============================================================================
                #                 random_vid_neg = np.random.choice(len(self))
                #                 random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
                #
                #                 while random_vid_neg == index:
                #                     random_vid_neg = np.random.choice(len(self))
                #                     random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
                #                 img_files_neg, _, cluster_id_neg = self.seqs[random_vid_neg]
                #
                #                 #hard neg ver
                #                 while random_vid_neg == index or cluster_id != cluster_id_neg:
                #                     random_vid_neg = np.random.choice(len(self))
                #                     random_vid_neg = self.indices[random_vid_neg % len(self.indices)]
                #                     img_files_neg, _, cluster_id_neg = self.seqs[random_vid_neg]
                #
                #                 random_fid_neg = np.random.choice(len(img_files_neg))
                # =============================================================================

                #                random_fid1, random_fid2 = np.random.choice(a=len(img_files), size=2, replace=False)
                z = self.img_loader(img_files[random_fid_z])
                x = self.img_loader(img_files_neg[random_fid_neg])

                # =============================================================================
                #                 z = cv2.imread(img_files[random_fid_z], cv2.IMREAD_COLOR)
                #                 x = cv2.imread(img_files_neg[random_fid_neg], cv2.IMREAD_COLOR)
                #                 z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
                #                 x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                # =============================================================================

                imgz_h, imgz_w, _ = z.shape
                imgx_h, imgx_w, _ = x.shape
                #                target_pos = [img_w//2, img_h//2]
                #                target_sz = [img_w//6, img_h//6]

                target_sz_z = [imgz_w // np.random.randint(4, 9), imgz_h // np.random.randint(4, 9)]
                target_pos_z = [np.random.randint(target_sz_z[0], (imgz_w - target_sz_z[0])),
                                np.random.randint(target_sz_z[1], (imgz_h - target_sz_z[1]))]

                target_sz_x = [imgx_w // np.random.randint(4, 9), imgx_h // np.random.randint(4, 9)]
                target_pos_x = [np.random.randint(target_sz_x[0], (imgx_w - target_sz_x[0])),
                                np.random.randint(target_sz_x[1], (imgx_h - target_sz_x[1]))]

                box_z = self._cxy_wh_2_bbox(target_pos_z, target_sz_z)
                box_x = self._cxy_wh_2_bbox(target_pos_x, target_sz_x)

                item = (z, x, box_z, box_x)
                if self.transforms is not None:
                    item = self.transforms(*item)

                return item + (neg,)
            else:
                random_fid = np.random.choice(len(img_files))
                # =============================================================================
                #                 z = cv2.imread(img_files[random_fid], cv2.IMREAD_COLOR)
                #     #            x = cv2.imread(img_files[random_fid], cv2.IMREAD_COLOR)
                #                 z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
                #     #            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                # =============================================================================

                z = self.img_loader(img_files[random_fid])  # get RGB img

                img_h, img_w, _ = z.shape

                target_sz = [img_w // np.random.randint(4, 9), img_h // np.random.randint(4, 9)]
                target_pos = [np.random.randint(target_sz[0], (img_w - target_sz[0])),
                              np.random.randint(target_sz[1], (img_h - target_sz[1]))]

                box = self._cxy_wh_2_bbox(target_pos, target_sz)

                item = (z, z, box, box)
                if self.transforms is not None:
                    item = self.transforms(*item)

                return item + (neg,)
        elif self.supervised == 'feature':
            random_fid = np.random.choice(len(img_files))

            img_path = img_files[random_fid]
            z = self.img_loader(img_path)  # get RGB img

            img_h, img_w, _ = z.shape

            if self.transforms is not None:
                item = self.transforms(z)

            return [item, img_path, seq_name]

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _cxy_wh_2_bbox(self, cxy, wh):
        return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])



