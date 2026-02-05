import logging
import os
import time
import sys

from loss.DiceLoss import Dice
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_PATH)

from data.CorneaDataLoader import CorneaDataset
from models.SpatialTransformer import SpatialTransformer
from loss.KeypointLoss import loss_func_keypoint_l1
from loss.ImageIntensityLoss import NCC
from utils.ImageTransform import get_homo_coord, transform, matrix_to_flow_2d, cal_affine_matrix
from evaluator.MetricsLandmarks import cal_dis
from utils.Augmentation import generate_random_affine_matrix_2d


class BasicAffineTrainer(nn.Module):
    def __init__(self, run_name='default', path_data_folder='/path_data',
                 fold=0, train_mode='landmark', batch_size=16, n_epochs=2000, learning_rate=1e-5, val_freq=100,
                 loss_weights=(1., 1., 1.), image_size=(384, 384), num_iter_train=1, num_iter_test=None, 
                 prob_steps=(4, 1, 1, 1, 1), device='cuda', **kwargs):
        super(BasicAffineTrainer, self).__init__()
        self.device = device
        self.run_name = run_name
        self.init_lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.test_freq = val_freq
        self.loss_weights = torch.tensor(loss_weights, device=device)
        self.train_mode = train_mode
        self.fold = fold
        self.image_size = image_size
        self.continue_training = False
        self.num_iter_train = num_iter_train if num_iter_train is not None else 1
        self.num_iter_test = self.num_iter_train if num_iter_test is None else num_iter_test
        if prob_steps is None:
            prob_steps = torch.ones((num_iter_train,), device=device) / num_iter_train
        else:
            prob_steps = torch.tensor(prob_steps, device=device)
            prob_steps = prob_steps / torch.sum(prob_steps)
        self.thresh_steps = torch.zeros((num_iter_train + 1,), device=device)
        for i in range(self.num_iter_train):
            self.thresh_steps[i+1] = prob_steps[0:(i+1)].sum()

        # prepare datasets data_backup dataloaders
        out_keys = ['image', 'label', 'mask', 'keypoints', 'num_keypoints', 'matrix']
        train_set = CorneaDataset(path_folder=path_data_folder, mode='train', fold=fold, out_keys=out_keys,
                                  aug_idx_list=[0], contain_origin=True)
        val_set = CorneaDataset(path_folder=path_data_folder, mode='val', fold=fold, out_keys=out_keys,
                                aug_idx_list=[0], contain_origin=True)
        test_set = CorneaDataset(path_folder=path_data_folder, mode='test', fold=fold, out_keys=out_keys,
                                 aug_idx_list=[0], contain_origin=True)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        print('training data: %d, validation data: %d, test data %d' % (len(train_set), len(val_set), len(test_set)))
        # prepare model
        self.stn_linear = SpatialTransformer(size=self.image_size, mode='bilinear').to(device)
        self.stn_nearest = SpatialTransformer(size=self.image_size, mode='nearest').to(device)
        self.affine_predictor = None  # should be defined in child class
        self.optimizer = None  # should be defined in child class
        self.scheduler = None

        # prepare loss functions
        self.loss_func_keypoints = [
            loss_func_keypoint_l1,
        ]
        self.loss_func_image = [
            NCC(),
        ]
        self.loss_func_vessel = [
            # MultiResolutionNCC(win=9),
            Dice(),
        ]

        self.start_ep = 0
        self.best_dis = torch.tensor(200., device=self.device)
        self.to(device)

    def prepare_input(self, data_batch, all_label=False):
        image_moving = data_batch['image_moving'].to(self.device).float()
        image_fixed = data_batch['image_fixed'].to(self.device).float()
        mask_moving = data_batch['mask_moving'].to(self.device).float()
        mask_fixed = data_batch['mask_fixed'].to(self.device).float()
        label_moving = data_batch['label_moving'].to(self.device).float()
        label_fixed = data_batch['label_fixed'].to(self.device).float()
        if not all_label:
            label_moving = label_moving[:, 0:1]
            label_fixed = label_fixed[:, 0:1]
        keypoints_moving = data_batch['keypoints_moving'].to(self.device).float()
        keypoints_fixed = data_batch['keypoints_fixed'].to(self.device).float()
        keypoints_moving, keypoints_fixed = get_homo_coord(keypoints_moving), get_homo_coord(keypoints_fixed)
        num_keypoints = data_batch['num_keypoints'].to(self.device).long()

        return image_moving, image_fixed, mask_moving, mask_fixed, label_moving, label_fixed, \
               keypoints_moving, keypoints_fixed, num_keypoints

    def _random_augment(self, bs, p=0.5):
        # random pick some from the batch to augment
        p_aug = torch.rand(size=(bs,))
        idx_aug = torch.where(torch.less(p_aug, p))[0]

        affine_matrix_random = torch.eye(3, device=self.device).reshape(1, 3, 3).repeat(bs, 1, 1)
        affine_matrix_random[idx_aug] = generate_random_affine_matrix_2d(
            rotation=(-15, 15),
            translation=(-0.3, 0.3, -0.3, 0.3),
            scale=(0.95, 1.05, 0.95, 1.05),
            shear=(-0.1, 0.1, -0.1, 0.1),
            n=idx_aug.shape[0],
            device=self.device
        )

        return affine_matrix_random

    def _apply_transform(self, image, label, mask, keypoints, keypoints_aug, num_keypoints):
        transform_matrix = cal_affine_matrix(keypoints, keypoints_aug, num_keypoints)
        flow = matrix_to_flow_2d(transform_matrix, shape=self.image_size)
        image = self.stn_linear(image, flow)
        label = self.stn_nearest(label, flow)
        mask = self.stn_nearest(mask, flow)
        keypoints = keypoints_aug.clone()
        return image, label, mask, keypoints

    # need to define in child class
    def model_forward(self, image_moving, image_fixed, **kwargs):
        return torch.rand((3, 3))
    
    def model_forward_iteratively(
            self, image_moving, image_fixed, keypoints_moving, keypoints_fixed, num_keypoints, label_moving, 
            if_record_intermediate=False, **kwargs):
        # forward and record intermediate image
        image_moving_warped_list, label_moving_warped_list, keypoints_moving_warped_list, dis_progress_list = [], [], [], []
        transform_matrix = torch.eye(3, device=self.device).unsqueeze(0).repeat(image_fixed.shape[0], 1, 1)
        identity_matrix = torch.eye(3, device=self.device).unsqueeze(0).repeat(image_fixed.shape[0], 1, 1)
        start = time.time()
        for s in range(self.num_iter_test):
            condition = torch.tensor([s, ] * image_fixed.shape[0], device=image_fixed.device)
            flow = matrix_to_flow_2d(transform_matrix, shape=self.image_size)
            tmp_image_moving = self.stn_linear(image_moving, flow)
            tmp_transform_matrix = self.model_forward(tmp_image_moving, image_fixed, condition=condition)

            if s > 0 and ((tmp_transform_matrix - identity_matrix) ** 2).sum() < 1e-6:
                if if_record_intermediate:
                    dis = dis_progress_list[-1].copy()
                    image_moving_warped = image_moving_warped_list[s-1].copy()
                    label_moving_warped = label_moving_warped_list[s-1].copy()
                    keypoints_moving_warped = keypoints_moving_warped_list[s-1].copy()
            else:
                transform_matrix = tmp_transform_matrix.matmul(transform_matrix)

                if if_record_intermediate:
                    # transform the points
                    keypoints_moving_warped = transform(transform_matrix, keypoints_moving, num_keypoints).detach()

                    # warp the image
                    flow_moving = matrix_to_flow_2d(transform_matrix, shape=self.image_size)
                    image_moving_warped = self.stn_linear(image_moving, flow_moving)
                    label_moving_warped = self.stn_nearest(label_moving, flow_moving)

                    # calculate distance
                    dis = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints, 
                                  image_size=self.image_size, mode=['mean'])[0]
                    dis = dis.detach().cpu().numpy()

                    image_moving_warped = image_moving_warped.squeeze().detach().cpu().numpy()
                    label_moving_warped = label_moving_warped.squeeze().detach().cpu().numpy()
                    keypoints_moving_warped = keypoints_moving_warped.squeeze().detach().cpu().numpy()

            # time info
            end = time.time()

            # record the intermediate image and info
            if if_record_intermediate:
                dis_progress_list.append(dis)
                image_moving_warped_list.append(image_moving_warped)
                label_moving_warped_list.append(label_moving_warped)
                keypoints_moving_warped_list.append(keypoints_moving_warped)
            
        return {
            'transform_matrix': transform_matrix,
            'time_per_batch': end - start,
            'dis_progress_list': dis_progress_list,
            'image_moving_warped_list': image_moving_warped_list,
            'label_moving_warped_list': label_moving_warped_list,
            'keypoints_moving_warped_list': keypoints_moving_warped_list
        }

    def train_step(self, ep, loader):
        self.eval()
        self.affine_predictor.train()
        train_loss, train_dis, train_SR, num_recorded = 0., [], 0., 0
        tbar = tqdm(loader, ncols=130)
        for batch_idx, data_batch in enumerate(tbar):
            # transfer data_backup to gpu, tune the formats
            image_moving, image_fixed, mask_moving, mask_fixed, label_moving, label_fixed, \
            keypoints_moving, keypoints_fixed, num_keypoints = self.prepare_input(data_batch)

            with torch.no_grad():
                # randomly augment both fixed and moving images
                random_affine_moving = self._random_augment(bs=image_moving.shape[0], p=0.5)
                random_affine_fixed = self._random_augment(bs=image_fixed.shape[0], p=0.5)

                # random generate iteration step number
                random_prob = torch.rand(size=(image_fixed.shape[0],), device=image_fixed.device)
                random_step = torch.zeros_like(random_prob)
                for i in range(self.num_iter_train):
                    idx = torch.logical_and(torch.ge(random_prob, self.thresh_steps[i]),
                                            torch.less(random_prob, self.thresh_steps[i + 1]))
                    random_step[idx] = i
                random_step = random_step.long()
                for s in range(self.num_iter_train - 1):
                    idx = torch.greater(random_step, s)
                    if idx.sum() == 0:
                        break
                    flow = matrix_to_flow_2d(random_affine_moving[idx], shape=self.image_size)
                    tmp_image_moving = self.stn_linear(image_moving[idx], flow)
                    flow = matrix_to_flow_2d(random_affine_fixed[idx], shape=self.image_size)
                    tmp_image_fixed = self.stn_linear(image_fixed[idx], flow)
                    condition = torch.tensor([s,] * idx.sum(), device=image_fixed.device)
                    random_affine_moving[idx] = self.model_forward(
                        tmp_image_moving, tmp_image_fixed, condition=condition).matmul(random_affine_moving[idx])

                # apply augmentation
                keypoints_moving_aug = transform(random_affine_moving, keypoints_moving, num_keypoints)
                keypoints_fixed_aug = transform(random_affine_fixed, keypoints_fixed, num_keypoints)
                image_moving, label_moving, mask_moving, keypoints_moving = \
                    self._apply_transform(image_moving, label_moving, mask_moving, keypoints_moving, keypoints_moving_aug, num_keypoints)
                image_fixed, label_fixed, mask_fixed, keypoints_fixed = \
                    self._apply_transform(image_fixed, label_fixed, mask_fixed, keypoints_fixed, keypoints_fixed_aug, num_keypoints)

            # model forward
            transform_matrix = self.model_forward(image_moving, image_fixed, condition=random_step)
            transform_matrix_inv = torch.inverse(transform_matrix)

            # transform the coordinates of landmarks
            keypoints_moving_warped = transform(transform_matrix, keypoints_moving, num_keypoints)
            keypoints_fixed_warped = transform(transform_matrix_inv, keypoints_fixed, num_keypoints)

            if self.train_mode not in ['landmark']:
                # warp the images, labels and masks
                flow_moving = matrix_to_flow_2d(transform_matrix, shape=self.image_size)
                image_moving_warped = self.stn_linear(image_moving, flow_moving)
                label_moving_warped = self.stn_linear(label_moving, flow_moving)
                mask_moving_warped = self.stn_nearest(mask_moving, flow_moving)
                mask_moving_warped = torch.logical_and(mask_moving_warped, mask_fixed).float()
                # mask_moving_warped = mask_fixed
                flow_fixed = matrix_to_flow_2d(transform_matrix_inv, shape=self.image_size)
                image_fixed_warped = self.stn_linear(image_fixed, flow_fixed)
                label_fixed_warped = self.stn_linear(label_fixed, flow_fixed)
                mask_fixed_warped = self.stn_nearest(mask_fixed, flow_fixed)
                mask_fixed_warped = torch.logical_and(mask_fixed_warped, mask_moving).float()

            # calculate loss
            loss = torch.tensor(0., device=self.device)
            if self.train_mode in ['landmark']:
                for func in self.loss_func_keypoints:
                    loss += func(keypoints_moving_warped, keypoints_fixed, num_keypoints) * self.loss_weights[0]
                    loss += func(keypoints_fixed_warped, keypoints_moving, num_keypoints) * self.loss_weights[0]
            elif self.train_mode in ['vxm', 'vxm-seg']:
                for func in self.loss_func_image:
                    loss += func(image_moving_warped, image_fixed, mask_moving_warped) * self.loss_weights[1]
                    loss += func(image_fixed_warped, image_moving, mask_fixed_warped) * self.loss_weights[1]
                    # loss += func(image_moving_warped, image_fixed) * self.loss_weights[1]
                    # loss += func(image_fixed_warped, image_moving) * self.loss_weights[1]
                if self.train_mode in ['vxm-seg']:
                    for func in self.loss_func_vessel:
                        loss += func(label_moving_warped, label_fixed, mask_moving_warped) * self.loss_weights[2]
                        loss += func(label_fixed_warped, label_moving, mask_fixed_warped) * self.loss_weights[2]
                        # loss += func(label_moving_warped, label_fixed) * self.loss_weights[2]
                        # loss += func(label_fixed_warped, label_moving) * self.loss_weights[2]

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record training info
            with torch.no_grad():
                train_loss += loss.detach().data * image_fixed.shape[0]
                dis_mean = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints,
                                   image_size=self.image_size, mode=['mean'], reduction='none')[0].detach()
                dis_max = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints,
                                  image_size=self.image_size, mode=['max'], reduction='none')[0].detach()
                train_dis += dis_mean
                train_SR += torch.sum(torch.le(dis_max, 10.)).data
                num_recorded += image_fixed.shape[0]
                current_loss = train_loss / num_recorded
                current_dis_median = torch.median(torch.tensor(train_dis)).data
                current_dis_mean = torch.mean(torch.tensor(train_dis)).data
                current_SR = train_SR / num_recorded
                tbar.set_description('Epoch %d/%d: loss %.4f, median %.4f, mean %.4f, SR %.4f' %
                                     (ep, self.n_epochs, current_loss, current_dis_median, current_dis_mean, current_SR)
                )

        tbar.close()

        return current_dis_median, current_dis_mean, current_SR

    def adjust_lr(self, ep):
        if self.n_epochs / 2 <= ep < self.n_epochs * 3 / 4:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr * 0.5
        elif ep >= self.n_epochs * 3 / 4:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr * 0.5 * 0.5

    def eval_step(self, ep, loader):
        self.eval()
        with torch.no_grad():
            # online test stage
            test_dis, test_dis_before, test_SR, num_recorded = [], [], 0., 0

            for data_batch in loader:
                # transfer data_backup to gpu, tune the formats
                image_moving, image_fixed, mask_moving, mask_fixed, label_moving, label_fixed, \
                keypoints_moving, keypoints_fixed, num_keypoints = self.prepare_input(data_batch)

                # model forward
                num_iter_test = self.num_iter_test
                transform_matrix = torch.eye(3, device=self.device).unsqueeze(0).repeat(image_fixed.shape[0], 1, 1)
                for s in range(num_iter_test):
                    condition = torch.tensor([s,] * image_fixed.shape[0], device=image_fixed.device)
                    flow = matrix_to_flow_2d(transform_matrix, shape=self.image_size)
                    tmp_image_moving = self.stn_linear(image_moving, flow)
                    transform_matrix = self.model_forward(tmp_image_moving, image_fixed, condition=condition).matmul(transform_matrix)

                # transform the points
                keypoints_moving_warped = transform(transform_matrix, keypoints_moving, num_keypoints).detach()

                # record test info
                dis_mean = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints,
                                   image_size=self.image_size, mode=['mean'], reduction='none')[0].detach()
                dis_max = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints,
                                  image_size=self.image_size, mode=['max'], reduction='none')[0].detach()
                dis_before_mean = cal_dis(keypoints_moving, keypoints_fixed, num_keypoints,
                                          image_size=self.image_size, mode=['mean'], reduction='none')[0].detach()
                test_dis += dis_mean
                test_dis_before += dis_before_mean
                test_SR += torch.sum(torch.le(dis_max, 10.)).data
                num_recorded += image_moving.shape[0]

            test_dis_median = torch.median(torch.tensor(test_dis)).data
            test_dis_median_before = torch.median(torch.tensor(test_dis_before)).data
            test_dis_mean = torch.mean(torch.tensor(test_dis)).data
            test_dis_mean_before = torch.mean(torch.tensor(test_dis_before)).data
            test_SR = test_SR / num_recorded
            return test_dis_median, test_dis_median_before, test_dis_mean, test_dis_mean_before, test_SR

    def save_checkpoint(self, ep, ckp_name):
        checkpoint = {
            'epoch': ep,
            'best_dis': self.best_dis,
            'model': self.affine_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.path_result, 'checkpoint_%s.pth' % ckp_name))

    def load_checkpoint(self, path_ckpt, continue_training=False):
        checkpoint = torch.load(path_ckpt, map_location=self.device)
        print('Checkpoint of epoch %d' % checkpoint['epoch'])
        self.affine_predictor.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if continue_training:
            self.start_ep = checkpoint['epoch'] + 1
            self.best_dis = checkpoint['best_dis']
            self.best_dis.to(self.device)
            self.continue_training = continue_training

    def full_training_loop(self):
        # create saving folder
        self.run_name = '%s_ep-%d_lr-%.0e_bs-%d' % \
                        (self.run_name, self.n_epochs, self.init_lr, self.batch_size)
        self.path_result = os.path.join(PROJECT_PATH, 'result/', self.run_name)
        if not os.path.exists(self.path_result):
            os.mkdir(self.path_result)
        self.path_result = '%s/fold_%d' % (self.path_result, self.fold)
        if not os.path.exists(self.path_result):
            os.mkdir(self.path_result)

        # log information
        if not self.continue_training:
            if os.path.exists(os.path.join(self.path_result, 'log.txt')):
                os.remove(os.path.join(self.path_result, 'log.txt'))
        logging.basicConfig(filename=os.path.join(self.path_result, 'log.txt'),
                            level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
        logging.info('Results saved in: %s' % self.path_result)
        print('Results saved in: %s' % self.path_result)

        # training
        for ep in range(self.start_ep, self.n_epochs):
            train_dis_median, train_dis_mean, train_SR = self.train_step(ep=ep, loader=self.train_loader)
            val_dis_median, val_dis_median_before, val_dis_mean, val_dis_mean_before, val_SR = \
                self.eval_step(ep=ep, loader=self.val_loader)
            logging.info('Validation: epoch %d, median(before)=%.3f, mean(before)=%.3f, median=%.3f, mean=%.3f, val SR=%.3f'
                         % (ep, val_dis_median_before, val_dis_mean_before, val_dis_median, val_dis_mean, val_SR))
            print('Validation: epoch %d, median(before)=%.3f, mean(before)=%.3f, median=%.3f, mean=%.3f, val SR=%.3f'
                  % (ep, val_dis_median_before, val_dis_mean_before, val_dis_median, val_dis_mean, val_SR))

            # validation and save model
            if ((ep + 1) % self.test_freq == 0 or (val_dis_median < self.best_dis and ep > self.n_epochs / 3)):
                self.eval()
                test_dis_median, test_dis_median_before, test_dis_mean, test_dis_mean_before, test_SR = \
                    self.eval_step(ep=ep, loader=self.test_loader)
                if (ep + 1) % self.test_freq == 0:
                    self.save_checkpoint(ep=ep, ckp_name='current')
                if val_dis_median < self.best_dis and ep > self.n_epochs / 3:
                    self.best_dis = val_dis_median
                    self.save_checkpoint(ep=ep, ckp_name='best')
                    logging.info('New best model saved, now the best_dis = %.4f, SR = %.4f' % (self.best_dis, val_SR))
                    print('New best model saved, now the best_dis = %.4f, SR = %.4f' % (self.best_dis, val_SR))
                logging.info('Test: epoch %d, median(before)=%.4f, mean(before)=%.4f, median=%.4f, mean=%.4f, val SR=%.4f'
                             % (ep, test_dis_median_before, test_dis_mean_before, test_dis_median, test_dis_mean, test_SR))
                print('Test: epoch %d, median(before)=%.4f, mean(before)=%.4f, median=%.4f, mean=%.4f, val SR=%.4f'
                      % (ep, test_dis_median_before, test_dis_mean_before, test_dis_median, test_dis_mean, test_SR))

            # learning rate decay
            self.adjust_lr(ep)


