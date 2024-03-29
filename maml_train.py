from __future__ import absolute_import

import torch.nn as nn
import torch.optim as optim
import tqdm
import os
from time import localtime, strftime
from collections import namedtuple
from datasets.vid import ImageNetVID
from inner_loop_optimizers import LSLRGradientDescentLearningRule

from siamfc import ops
from siamfc.maml_backbones import AlexNet, Resnet18, Inception, VGG16
from siamfc.maml_heads import SiamFC, SiamFC_1x1_DW
# from datasets.coco import Coco
import random

# from siamfc.ssiamfc import TrackerSiamFC
# from siamfc.siamfc_stn import TrackerSiamFC
# from siamfc.siamfc_weight_dropping import TrackerSiamFC
from siamfc.siamfc_maml_weight_dropping import TrackerSiamFC, Net
from utils.average_meter_helper import AverageMeter
from siamfc.transforms import SiamFCTransforms
from torch.utils.data import DataLoader
from utils.img_loader import cv2_RGB_loader
from siamfc.datasets import Pair
from torch.utils.tensorboard import SummaryWriter

import ipdb
import torch
import numpy as np
import time

torch.manual_seed(123456)  # cpu
torch.cuda.manual_seed(123456)  # gpu
np.random.seed(123456)  # numpy
random.seed(123456)  # random and transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class maml_trainer(nn.Module):
    def __init__(self):
        super(maml_trainer, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

        # file path and saving initialization
        self.neg_dir = ['./seq2neg_dict.json', './cluster_dict.json']
        self.root_dir = '../dataset/ILSVRC2015'  # Dataset path
        self.save_dir = './checkpoints/maml_rdn_query_pair_maml_optim_sche/'
        self.save_path = os.path.join(self.save_dir, 'S2SiamFC')

        # setting maml config
        self.maml_args = self.parse_maml_args()

        # setting inner tracker
        self.model = Net(
            backbone=AlexNet(args=self.maml_args),
            head=SiamFC(args=self.maml_args))
        self.load_pretrain()
        self.model.to(self.device)
        ops.init_weights(self.model)

        # device setting initialization
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.model = nn.DataParallel(module=self.model)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()
        self.to(self.device)

        self.tracker = TrackerSiamFC(model=self.model, loss_setting=[0.5, 2.0, 0],maml_args=self.maml_args)

        # setting inner tracker config, inner-loop learning rate, and inner loop optimizer 
        self.cfg = self.tracker.cfg
        self.task_learning_rate = self.maml_args.task_learning_rate
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=self.device,
                                                                    init_learning_rate=self.maml_args.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.maml_args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.maml_args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.model.named_parameters()))

        self.to(self.device)
        
        # =============================================================================
        #print("Inner Loop parameters")
        #for key, value in self.inner_loop_optimizer.named_parameters():
        #    print(key, value.shape, value.device)
        # =============================================================================

        # outer loop optimizer, lr scheduler
        self.optimizer = optim.Adam(self.trainable_parameters(), lr=self.maml_args.meta_learning_rate, amsgrad=False)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.maml_args.total_epochs,
                                                            eta_min=self.maml_args.min_learning_rate)
        
        # =============================================================================
        #print("Outer Loop Parameters")
        #for name, param in self.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.shape, param.device, param.requires_grad)
        # =============================================================================

        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)

        # setting training dataset
        self.seqs = ImageNetVID(self.root_dir, subset=['train'], neg_dir=self.neg_dir[0])
        self.current_iter = 0
        self.current_epoch = 0

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.maml_args.number_of_training_steps_per_iter)) * (
                1.0 / self.maml_args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.maml_args.number_of_training_steps_per_iter / self.maml_args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.maml_args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (
                    self.current_epoch * (self.maml_args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.maml_args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.maml_args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)
        return param_dict

    def print_outer_loop_param(self):
        print("Outer Loop Parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model.module.zero_grad(params=names_weights_copy)
        else:
            self.model.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses):
        losses = dict()
        losses['loss'] = torch.mean(torch.stack(total_losses))

        return losses

    def load_pretrain(self):
        # setting pretrain model
        filepath = os.path.join(".", "pretrain", "eccv_best", "siamfc_alexnet_e41.pth")
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

    def parse_maml_args(self, **kwargs):
        # default parameters (can checkout the parameter from maml++ github)
        arg = {
            'norm_layer': "batch_norm",
            'learnable_bn_gamma': True,
            'learnable_bn_beta': True,
            'learnable_per_layer_per_step_inner_loop_learning_rate': True,
            'per_step_bn_statistics': False,
            'number_of_training_steps_per_iter': 5,
            'enable_inner_loop_optimizable_bn_params': True,
            'total_epochs': 20,
            'total_iter_per_epoch': 10000,
            'task_learning_rate': 0.001,
            'total_num_inner_loop_steps': 5,
            'use_second_order': True,
            'use_learnable_learning_rates': True,
            'use_multi_step_loss_optimization': True,
            'multi_step_loss_num_epochs': 10,
            'meta_learning_rate': 0.001,
            'min_learning_rate': 0.0001,
            'num_steps': 5,
            'batches_per_iter': 3
        }

        for key, val in kwargs.items():
            if key in arg:
                arg.update({key: val})
        return namedtuple('Config', arg.keys())(**arg)

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        torch.save(state, f=model_save_dir)

    # =============================================================================
    #def get_training_batches(self, seqs, supervised):
    #    transforms = SiamFCTransforms(
    #        exemplar_sz=self.cfg.exemplar_sz,
    #        instance_sz=self.cfg.instance_sz,
    #        context=self.cfg.context)
    #
    #    # make up random pairs with transform
    #    dataset = Pair(
    #        seqs=seqs,
    #        transforms=transforms, supervised=supervised, img_loader=cv2_RGB_loader, neg=self.cfg.neg)
    #
    #    # setup dataloader
    #    self.dataloader = DataLoader(
    #        dataset,
    #        batch_size=self.cfg.batch_size,
    #        shuffle=True,
    #        num_workers=0,
    #        pin_memory=self.cuda,
    #        drop_last=True)
    # =============================================================================

    @torch.enable_grad()
    def maml_train(self, seqs, val_seqs=None,
                   save_dir='pretrained', supervised='supervised'):

        # ++++++++++++++++++ process be like ++++++++++++++++++
        # run inner loop (support set) + returning query set
        # inner loop update
        # run inner loop (query set)
        # update outter loop
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++


        # load model
        training_phase = True
        self.model.load_state_dict(self.state_dict, strict=False)
        print("==========model loaded==========")
        # initialize settings
        avg = AverageMeter()
        self.model.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        writer = SummaryWriter('runs/maml_rdn_query')

        # setup dataset, both support and query sets
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        # set gen_query to true will get both support and query set
        dataset = Pair(
            seqs=seqs,
            transforms=transforms, supervised=supervised, img_loader=cv2_RGB_loader, neg=self.cfg.neg, gen_query=True)
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.cuda,
            drop_last=True)

        end = time.time()
        while self.current_iter < self.maml_args.total_epochs * self.maml_args.total_iter_per_epoch:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            self.inner_loop_optimizer.zero_grad()
            self.zero_grad()
            total_losses = []
            #self.print_outer_loop_param()
            
            for it, batch in enumerate(dataloader):
                # batch = [support_z, support_x, query_z, query_x]
                query_losses = []
                data_time = time.time() - end
                per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
                names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
                num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

                names_weights_copy = {
                    name.replace('module.', ''): value.unsqueeze(0).repeat(
                        [num_devices] + [1 for i in range(len(value.shape))]) for
                    name, value in names_weights_copy.items()}

                for num_step in range(self.maml_args.num_steps):
                    # go through support set training
                    query_batch, support_loss, responses = self.tracker.train_step(batch,
                                                                                   names_weights_copy, phase='support',
                                                                                   num_step=num_step)  # extract query dataset from here
                    writer.add_scalar('support_loss', support_loss, it * self.maml_args.num_steps + num_step)
                    # update inner loop with support loss
                    names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                      names_weights_copy=names_weights_copy,
                                                                      use_second_order=self.maml_args.use_second_order,
                                                                      current_step_idx=num_step)
                    names_weights_copy = {key: torch.squeeze(value, 0) for key, value in names_weights_copy.items()}
                    
                    # go through query set training
                    if self.maml_args.use_multi_step_loss_optimization and training_phase and self.current_epoch < self.maml_args.multi_step_loss_num_epochs:
                        # get query loss after several steps
                        query_loss, responses = self.tracker.query_step(query_batch,
                                                                        names_weight_copy=names_weights_copy,
                                                                        phase='query', num_step=num_step)
                        writer.add_scalar('query_loss', query_loss, it * self.maml_args.num_steps + num_step)
                        query_losses.append(per_step_loss_importance_vectors[num_step] * query_loss)
                    else:
                        if num_step == (self.maml_args.number_of_training_steps_per_iter - 1):
                            # get query loss after each step
                            query_loss, responses = self.tracker.query_step(query_batch, names_weights_copy,
                                                                            phase='query', num_step=num_step)
                            query_losses.append(query_loss)
                            writer.add_scalar('query_loss', query_loss, it * self.maml_args.num_steps + num_step)
        
                task_losses = torch.sum(torch.stack(query_losses))
                total_losses.append(task_losses)
                if not training_phase:
                    self.model.restore_backup_stats()
                if it % self.maml_args.batches_per_iter == 0:
                    # update outer loop
                    self.optimizer.zero_grad()
                    self.zero_grad()
                    self.model.zero_grad()
                    losses = self.get_across_task_loss_metrics(total_losses=total_losses)
                    for idx, item in enumerate(per_step_loss_importance_vectors):
                        losses['loss_impoertance_vector_{}'.format(idx)] = item.detach().cpu().numpy()
                    loss = losses['loss']
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward() 
                    self.optimizer.step()
                    losses['learning_rate'] = self.lr_scheduler.get_lr()[0]
                    total_losses = []
            # save model
            writer.add_scalar('epoch_loss', loss, self.current_epoch)
            self.save_model(os.path.join(self.save_dir, "{}.pth".format(self.current_epoch)),
                            self.model.state_dict())
            self.current_epoch = self.current_epoch + 1

    def run_experiment(self):
        mode = ['supervised', 'self-supervised']
        # load pretrain
        self.load_pretrain()
        self.maml_train(self.seqs, supervised=mode[1], save_dir=self.save_path)


if __name__ == '__main__':
    # =============================================================================
    #     root_dir = 'E:\SiamMask\data\coco'
    #     seqs = Coco(root_dir, subset=['train'])
    # =============================================================================

    # =============================================================================
    #     save_dir = './checkpoints/eccv_rot_rcrop_mask_0515m0sig_got'
    #     root_dir = 'E:/GOT10K'
    #     seqs = GOT10k(root_dir, subset='train')
    # =============================================================================
    trainer = maml_trainer()
    trainer.run_experiment()

    print(strftime("%Y-%m-%d %H:%M:%neg_dirS", localtime()))
