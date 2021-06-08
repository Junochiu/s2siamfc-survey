from __future__ import absolute_import, division, print_function

import random
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
from six import string_types
from PIL import Image

from . import ops
from .maml_backbones import AlexNet, Resnet18, Inception, VGG16
from .maml_heads import SiamFC, SiamFC_1x1_DW
from .losses import AC_BalancedLoss, BalancedLoss, RankLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from .maml_basicstructure import extract_top_level_dict
from utils.lr_helper import build_lr_scheduler
from utils.img_loader import cv2_RGB_loader
from got10k.trackers import Tracker
from utils.average_meter_helper import AverageMeter

__all__ = ['TrackerSiamFC']


class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
        self.gradients = dict()

        def backward_hook_x(grad):
            self.gradients['x'] = grad

        def backward_hook_z(grad):
            self.gradients['z'] = grad

        self.back_x = backward_hook_x
        self.back_z = backward_hook_z

    def forward(self, z, x, params=None, num_step=0):
        param_dict = dict()
        if params is None:
            z = self.backbone(z)
            x = self.backbone(x)
            return self.head(z, x)
        else:
            params = {key:value[0] for key,value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

            z = self.backbone(z, params=param_dict['backbone'], num_step=num_step)
            x = self.backbone(x, params=param_dict['backbone'], num_step=num_step)
            return self.head(z, x, params=param_dict['head'], num_step=num_step)

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            #print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            #print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

class TrackerSiamFC(Tracker):
    def __init__(self, model,maml_args=None, net_path=None, name='SiamFC', loss_setting=[0, 1.5, 0], **kwargs):
        super(TrackerSiamFC, self).__init__(name, True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.maml_args = maml_args

        # setup criterion
        self.criterion = AC_BalancedLoss(pos_thres=loss_setting[0], alpha=loss_setting[1], margin=loss_setting[2])
        #        self.rank_loss = RankLoss()
        #        self.criterion = BalancedLoss()
        self.net = model
    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'backbone': 'Alex',
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 3,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
            'neg': 0.2,
            # loss weighting
            'no_mask': 0.7,
            'masked': 0.15,
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # loader img if path is given
        if isinstance(img, string_types):
            #            img = cv2_RGB_loader(img)
            img = Image.open(img)

        # =============================================================================
        #         self.norm_trans = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         torchvision.transforms.ToTensor()])
        # =============================================================================

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)  # return cv2

        z = cv2.normalize(z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        # =============================================================================
        #         z = self.norm_trans(z).unsqueeze(0).to(self.device)
        # =============================================================================





        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        if isinstance(img, string_types):
            #            img = cv2_RGB_loader(img)
            img = Image.open(img)

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # =============================================================================
        #         x = [self.norm_trans(ops.crop_and_resize(
        #             img, self.center, self.x_sz * f,
        #             out_size=self.cfg.instance_sz,
        #             border_value=self.avg_color)) for f in self.scale_factors]
        #         x = torch.stack(x).to(self.device)
        # =============================================================================

        # responses
        x = self.net.backbone(x)

        responses = self.net.head(self.kernel, x)

        responses = responses.squeeze(1).cpu().numpy()
        # upsample responses and penalize scale changes

        # =============================================================================
        #         print(np.max(responses[1]))
        #         cv2.imwrite("./res.jpg", responses[1]*255)
        #         raise
        # =============================================================================

        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]

        response -= response.min()
        response /= response.sum() + 1e-16

        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
        self.kernel = self.net.backbone(z)

    def maml_init(self, img, box):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # loader img if path is given
        if isinstance(img, string_types):
            #            img = cv2_RGB_loader(img)
            img = Image.open(img)

        # =============================================================================
        #         self.norm_trans = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         torchvision.transforms.ToTensor()])
        # =============================================================================

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)  # return cv2

        z = cv2.normalize(z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
         
        # add transform
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        x,z = transforms.inference_transform(x,z)

        inference_update = 3
        for idx in range(inference_update):
            loss,_ = self.train_step([z,x,[0]])
            self.optimizer.zero_grad()
            loss.backward()
            print("loss update in init = {}".format(loss))
            self.optimizer.step()
        # =============================================================================
        #         z = self.norm_trans(z).unsqueeze(0).to(self.device)
        # =============================================================================
        # set to evaluation mode
        self.net.eval()
        self.kernel = self.net.backbone(z)
    def track(self, img_files, box, visualize=True):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

    def train_step(self, batch, names_weight_copy, phase, num_step, backward=True):
        def get_labels(responses):
            # create labels
            r_b, r_c, r_w, r_h = responses.size()
            # calculate loss

            labels = []
            if all(neg):
                labels = torch.zeros(responses.size()).to(self.device)
            elif not any(neg):
                labels = self._create_labels(responses.size())
            else:
                for n in neg:
                    #            print(n)
                    if n:
                        labels.append(torch.zeros([1, r_w, r_h]).to(self.device))
                    else:
                        labels.append(self._create_label(responses.size()))
                labels = torch.stack(labels)

            return labels

        def get_grad(labels, responses):
            # get the grad from loss with pos idx
            pos_idx = Variable(labels.view(-1).nonzero().squeeze()).to(self.device, non_blocking=self.cuda)

            pred_pos_value = torch.index_select(responses.view(-1), 0, pos_idx).mean()

            self.net.zero_grad()
            pred_pos_value.backward(retain_graph=True)

            grad_z = self.net.gradients['z']

            return grad_z

        def get_adv_mask_img(z, feat_z, grad_z):
            # get the masked image using grad-cam like approach
            b, _, w, h = feat_z.size()

            z_for_dropping = z.clone().detach()
            z_dropping = []
            for bid in range(b):
                z_dropping.append(gradcam_dropping(grad_z[bid], feat_z[bid], z_for_dropping[bid]))
            z_dropping = torch.stack(z_dropping)

            return z_dropping

        def gradcam_dropping(grad, activations, image):
            # cal the saliency region using grad-cam like approach
            k, u, v = grad.size()

            alpha = grad.view(1, k, -1).mean(2)
            weights = alpha.view(1, k, 1, 1)

            h, w = image.size()[1:]  # c, w, h

            atten_map = (weights * activations)
            atten_map = F.relu(atten_map)
            atten_map = F.upsample(atten_map, size=(h, w), mode='bilinear', align_corners=False)

            atten_map_max, atten_map_min = atten_map.max(), atten_map.min()
            atten_map = (atten_map - atten_map_min).div(atten_map_max - atten_map_min)

            atten_map_thres_neg_idx = atten_map < 0.5
            atten_map_thres_pos_idx = atten_map >= 0.5
            atten_map[atten_map_thres_neg_idx] = 0  # keep don't care part
            atten_map[atten_map_thres_pos_idx] = 1  # erase attened part

            high_response_channel = torch.unique(atten_map[0].nonzero()[:, 0]).detach().cpu().numpy()

            avg_color = image.mean([1, 2])
            assert avg_color.size(0) == 3

            candidate = []
            for ele in high_response_channel:
                if atten_map[0][ele].sum() / (w * h) < 0.5:
                    candidate.append(ele)
            if len(candidate) != 0:
                random_idx = np.random.choice(candidate)
            else:
                random_idx = 0

            dropping_mask = atten_map[0][random_idx].bool()

            for channel in range(3):
                #                print(atten_map[0][random_idx].view(-1).nonzero())
                image[channel][dropping_mask] = avg_color[channel]
                image[channel][image[channel] == 0] = avg_color[channel]

            image_dropping = image
            #            image_dropping = image * torch.abs(1 - atten_map[0][random_idx])

            return image_dropping

            # set network mode


        #self.net.train(backward)


        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        if phase == "support":
            q_z = batch[2].to(self.device, non_blocking=self.cuda)
            q_x = batch[3].to(self.device, non_blocking=self.cuda)
        neg = batch[-1]
        random.seed()
        rdn = random.randint(0,100)
        torchvision.utils.save_image(z, './z_{}.png'.format(rdn))
        torchvision.utils.save_image(x, './x_{}.png'.format(rdn))

        param_dict = dict()
        params = {key: value[0] for key, value in names_weight_copy.items()}
        param_dict = extract_top_level_dict(current_dict=params)

        # inference
        feat_z = self.net.backbone(z, num_step=num_step, params=param_dict['backbone'])
        feat_x = self.net.backbone(x, num_step=num_step, params=param_dict['backbone'])

        feat_z.register_hook(self.net.back_z)

        responses = self.net.head(feat_z, feat_x, num_step=num_step, params=param_dict['head'])

        labels = get_labels(responses)

        grad_z = get_grad(labels, responses)

        z_masked_1 = get_adv_mask_img(z, feat_z, grad_z)
        z_masked_2 = get_adv_mask_img(z, feat_z, grad_z)
        #if phase == 'support':
            #query_set = get_adv_mask_img(z, feat_z, grad_z)
            #torchvision.utils.save_image(query_set, './query_set.png')
        #ipdb.set_trace()

        # torchvision.utils.save_image(z_dropping, './test_z_dropping.png')
        #torchvision.utils.save_image(z_masked_1, './test_z_masked_1.png')
        #torchvision.utils.save_image(z_masked_2, './test_z_masked_2.png')

        responses_masked_1 = self.net.forward(z_masked_1, x, num_step=num_step, params=names_weight_copy)
        responses_masked_2 = self.net.forward(z_masked_2, x, num_step=num_step, params=names_weight_copy)

        raw_loss = self.criterion.forward(responses, labels)
        masked_1_loss = self.criterion.forward(responses_masked_1, labels)
        masked_2_loss = self.criterion.forward(responses_masked_2, labels)

        loss_siam = self.cfg.no_mask * raw_loss + self.cfg.masked * masked_1_loss + self.cfg.masked * masked_2_loss
        if phase == 'support':
            return [q_z, q_x, neg], loss_siam, responses
        elif phase == 'support oritemp':
            return [q_z, x, z, neg], loss_siam, responses

        return loss_siam, responses

    def query_step(self,batch, names_weight_copy, phase, num_step, backward=False, original_temp=False):
        def get_labels(responses):
            # create labels
            r_b, r_c, r_w, r_h = responses.size()
            # calculate loss

            labels = []
            if all(neg):
                labels = torch.zeros(responses.size()).to(self.device)
            elif not any(neg):
                labels = self._create_labels(responses.size())
            else:
                for n in neg:
                    #            print(n)
                    if n:
                        labels.append(torch.zeros([1, r_w, r_h]).to(self.device))
                    else:
                        labels.append(self._create_label(responses.size()))
                labels = torch.stack(labels)

            return labels

        self.net.eval()
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        neg = batch[-1]

        #feat_z = self.net.backbone(z, num_step=num_step, params=param_dict['backbone'])
        #feat_x = self.net.backbone(x, num_step=num_step, params=param_dict['backbone'])
        #responses = self.net.head(feat_z, feat_x, num_step=num_step, params=param_dict['head'])

        responses = self.net.forward(z,x,params=names_weight_copy)
        if original_temp:
            original_x = batch[2].to(self.device,non_blocking=self.cuda)
            original_responses = self.net.forward(original_responses,x,param=names_weight_copy)
            labels = get_labels(original_responses)
            loss = self.cfg.no_mask * self.criterion(original_responses,labels) + self.cfg.masked * self.criterion(responses,labels)
        else:
            labels = get_labels(responses)
            loss = self.criterion(responses, labels)

        return loss, responses


    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained', supervised='supervised'):
        avg = AverageMeter()
        # set to train mode

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms, supervised=supervised, img_loader=cv2_RGB_loader, neg=self.cfg.neg)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        ipdb.set_trace()
        end = time.time()
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch

            # loop over dataloader
            ipdb.set_trace()
            for it, batch in enumerate(dataloader):

                #                torchvision.utils.save_image(batch[0][0], 'test0.png')
                #                torchvision.utils.save_image(batch[1][0], 'test1.png')
                #                raise ""

                data_time = time.time() - end

                loss, responses = self.train_step(batch)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                #                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()

                batcn_time = time.time() - end
                end = time.time()
                avg.update(loss=loss, batch_time=batcn_time, data_time=data_time)

                if (it + 1) % 50 == 0:
                    print('Epoch: {} [{}/{}] {:.5f} {:.5f} {:.5f}'.format(
                        epoch + 1, it + 1, len(dataloader), avg.loss, avg.batch_time, avg.data_time))
                    #                    print('Num_high:{:d}'.format(torch.sum(responses.detach()>0.8)))

                    sys.stdout.flush()

            self.lr_scheduler.step(epoch=epoch)

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

        json.dump(self.cfg._asdict(), open(os.path.join(save_dir, 'config.json'), 'w'), indent=4)

    def _create_labels(self, size):
        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels

    def _create_label(self, size):
        # skip if same sized labels already created
        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        _, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        label = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        label = label.reshape((1, h, w))
        #        label = np.tile(label, (n, c, 1, 1))

        # convert to tensors
        self.label = torch.from_numpy(label).to(self.device).float()

        return self.label
