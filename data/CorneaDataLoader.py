import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class CorneaDataset(Dataset):
    def __init__(self, path_folder, mode='train', fold=None, out_keys=('image'), aug_idx_list=None, contain_origin=True):
        super(CorneaDataset, self).__init__()
        json_name = [x for x in os.listdir(path_folder) if x[-12:] == 'dataset.json'][0]
        with open(os.path.join(path_folder, json_name), 'r') as f:
            self.dict_dataset = json.load(f)
            f.close()
        self.path_folder = path_folder
        self.mode = mode
        self.out_keys = out_keys
        self.image_size = self.dict_dataset['tensorImageShape']
        self.num_label_class = self.dict_dataset['numLabelClass']
        self.dict_fold = self.dict_dataset if fold is None else self.dict_dataset['fold_%d' % fold]
        if mode in ['train', 'training']:
            mode_name = 'training'
        elif mode in ['val', 'validation']:
            mode_name = 'val'
        elif mode in ['test']:
            mode_name = 'test'
        else:
            mode_name = None
        pair_list = self.dict_fold['registration_%s' % mode_name]
        self.data_name_list = []
        self.num_data = 0
        if aug_idx_list is not None:
            for pair in pair_list:
                name = pair['fixed'][:-6]
                for idx_aug in aug_idx_list:
                    pair_aug = {
                        'fixed': '%s_aug_%d_1.npz' % (name, idx_aug),
                        'moving': '%s_aug_%d_2.npz' % (name, idx_aug),
                    }
                    self.data_name_list.append(pair_aug)
            self.num_data += len(self.data_name_list)
        if contain_origin:
            self.data_name_list += pair_list
            self.num_data += self.dict_fold['numRegistration_%s' % mode_name]

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        name = self.data_name_list[item]
        data_fixed = np.load(os.path.join(self.path_folder, name['fixed']))
        data_moving = np.load(os.path.join(self.path_folder, name['moving']))

        out_dict = {
            'name_fixed': name['fixed'],
            'name_moving': name['moving']
        }
        if 'image' in self.out_keys:
            image_fixed, image_moving = data_fixed['image'], data_moving['image']
            image_fixed = torch.from_numpy(image_fixed).unsqueeze(0)
            image_moving = torch.from_numpy(image_moving).unsqueeze(0)
            out_dict['image_fixed'], out_dict['image_moving'] = image_fixed, image_moving
        if 'label' in self.out_keys:
            label_fixed, label_moving = data_fixed['label'], data_moving['label']
            label_fixed = torch.from_numpy(label_fixed)
            label_moving = torch.from_numpy(label_moving)
            out_dict['label_fixed'], out_dict['label_moving'] = label_fixed, label_moving
        if 'mask' in self.out_keys:
            if 'mask' in data_fixed.keys():
                mask_fixed, mask_moving = data_fixed['mask'], data_moving['mask']
                mask_fixed = torch.from_numpy(mask_fixed).unsqueeze(0)
                mask_moving = torch.from_numpy(mask_moving).unsqueeze(0)
            else:
                mask_fixed = torch.ones((1, *self.image_size), dtype=torch.float32)
                mask_moving = torch.ones((1, *self.image_size), dtype=torch.float32)
            out_dict['mask_fixed'], out_dict['mask_moving'] = mask_fixed, mask_moving
        if 'keypoints' in self.out_keys:
            keypoints_fixed, keypoints_moving = data_fixed['keypoints'], data_moving['keypoints']
            keypoints_fixed = torch.from_numpy(keypoints_fixed)
            keypoints_moving = torch.from_numpy(keypoints_moving)
            out_dict['keypoints_fixed'], out_dict['keypoints_moving'] = keypoints_fixed, keypoints_moving
        if 'num_keypoints' in self.out_keys:
            num_keypoints = data_fixed['num_keypoints']
            num_keypoints = torch.from_numpy(num_keypoints)
            out_dict['num_keypoints'] = num_keypoints
        # if 'matrix' in self.out_keys:
        #     _keypoints_moving = get_homo_coord(keypoints_moving.unsqueeze(0)).squeeze()
        #     _keypoints_fixed = get_homo_coord(keypoints_fixed.unsqueeze(0)).squeeze()
        #     matrix_moving2fixed = cal_affine_matrix(
        #         points_src=_keypoints_moving, points_tgt=_keypoints_fixed, num_keypoints=num_keypoints
        #     )
        #     matrix_fixed2moving = torch.inverse(matrix_moving2fixed)
        #     out_dict['matrix_moving2fixed'] = matrix_moving2fixed
        #     out_dict['matrix_fixed2moving'] = matrix_fixed2moving

        return out_dict



