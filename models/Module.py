import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def para_to_matrix_affine_2d(parameters):
    """
    Convert the affine parameters to matrix
    :param parameters: bs x 6
    :return: matrix: bs x 3 x 3
    """
    translation_x = parameters[:, 0]
    translation_y = parameters[:, 1]
    rotation = parameters[:, 2] * torch.pi
    shearing = parameters[:, 3] * torch.pi
    scaling_x = parameters[:, 4] * 0.5 + 1
    scaling_y = parameters[:, 5] * 0.5 + 1

    mat_translation = torch.eye(3, device=parameters.device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_rotation = torch.eye(3, device=parameters.device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_shearing = torch.eye(3, device=parameters.device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_scaling = torch.eye(3, device=parameters.device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)

    mat_translation[:, 0, 2] = translation_x
    mat_translation[:, 1, 2] = translation_y

    mat_scaling[:, 0, 0] = scaling_x
    mat_scaling[:, 1, 1] = scaling_y

    mat_rotation[:, 0, 0] = torch.cos(rotation)
    mat_rotation[:, 0, 1] = -torch.sin(rotation)
    mat_rotation[:, 1, 0] = torch.sin(rotation)
    mat_rotation[:, 1, 1] = torch.cos(rotation)

    mat_shearing[:, 0, 1] = shearing

    matrix = mat_shearing.matmul(mat_scaling).matmul(mat_rotation).matmul(mat_translation)

    return matrix


def para_to_matrix_affine_3d(parameters):
    """
    Convert the affine parameters to matrix
    :param parameters: bs x 12
    :return: matrix: bs x 4 x 4
    """
    device = parameters.device

    translation_xyz = parameters[:, 0:3]
    rotation_xyz = parameters[:, 3:6] * math.pi
    shearing_xyz = parameters[:, 6:9] * math.pi
    scaling_xyz = 1 + parameters[:, 9:12] * 0.5

    # print('')
    # print('Translation:', translation_xyz)
    # print('Rotation:', rotation_xyz)
    # print('Shearing:', shearing_xyz)
    # print('Scaling:', scaling_xyz)

    mat_translation = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_rotation_x = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_rotation_y = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_rotation_z = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_shearing = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)
    mat_scaling = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(parameters.shape[0], 1, 1)

    mat_translation[:, 0, 3] = translation_xyz[:, 0]
    mat_translation[:, 1, 3] = translation_xyz[:, 1]
    mat_translation[:, 2, 3] = translation_xyz[:, 2]

    mat_scaling[:, 0, 0] = scaling_xyz[:, 0]
    mat_scaling[:, 1, 1] = scaling_xyz[:, 1]
    mat_scaling[:, 2, 2] = scaling_xyz[:, 2]

    mat_rotation_x[:, 1, 1] = torch.cos(rotation_xyz[:, 0])
    mat_rotation_x[:, 1, 2] = -torch.sin(rotation_xyz[:, 0])
    mat_rotation_x[:, 2, 1] = torch.sin(rotation_xyz[:, 0])
    mat_rotation_x[:, 2, 2] = torch.cos(rotation_xyz[:, 0])

    mat_rotation_y[:, 0, 0] = torch.cos(rotation_xyz[:, 1])
    mat_rotation_y[:, 0, 2] = torch.sin(rotation_xyz[:, 1])
    mat_rotation_y[:, 2, 0] = -torch.sin(rotation_xyz[:, 1])
    mat_rotation_y[:, 2, 2] = torch.cos(rotation_xyz[:, 1])

    mat_rotation_y[:, 0, 0] = torch.cos(rotation_xyz[:, 2])
    mat_rotation_y[:, 0, 1] = -torch.sin(rotation_xyz[:, 2])
    mat_rotation_y[:, 1, 0] = torch.sin(rotation_xyz[:, 2])
    mat_rotation_y[:, 1, 1] = torch.cos(rotation_xyz[:, 2])

    mat_rotation = mat_rotation_z.matmul(mat_rotation_y).matmul(mat_rotation_x)

    mat_shearing[:, 0, 1] = shearing_xyz[:, 0]
    mat_shearing[:, 0, 2] = shearing_xyz[:, 1]
    mat_shearing[:, 1, 2] = shearing_xyz[:, 2]

    matrix = mat_shearing.matmul(mat_scaling).matmul(mat_rotation).matmul(mat_translation)

    return matrix


class ConvNormActi(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ndims=2, norm='BatchNorm'):
        super(ConvNormActi, self).__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        BN = getattr(nn, '%s%dd' % (norm, ndims))
        self.conv = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = BN(out_channels)
        self.acti = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, x):
        return self.acti(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_1x1conv=False, ndims=2,
                 norm='BatchNorm'):
        super(ResBlock, self).__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        Norm = getattr(nn, '%s%dd' % (norm, ndims))
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if use_1x1conv:
            self.conv3 = Conv(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.norm1 = Norm(out_channels)
        self.norm2 = Norm(out_channels)
        self.acti1 = nn.LeakyReLU(1e-2, inplace=True)
        self.acti2 = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, X):
        Y = self.acti1(self.norm1(self.conv1(X)))
        Y = self.norm2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.acti2(Y)


class FeatureExtractor(nn.Module):
    def __init__(self, image_size=(384, 384), num_channels=(1, 64, 128), num_blocks=(2, 3), norm='BatchNorm'):
        super(FeatureExtractor, self).__init__()

        self.image_size = image_size
        self.ndims = len(image_size)

        # first layer
        block_list = [
            ConvNormActi(
                num_channels[0], num_channels[1], kernel_size=7, stride=2, padding=3, ndims=self.ndims, norm='BatchNorm'
            ),
        ]
        for _ in range(num_blocks[0] - 1):
            block_list += [
                ConvNormActi(
                    num_channels[1], num_channels[1], kernel_size=3, stride=1, padding=1, ndims=self.ndims, norm='BatchNorm'
                )
            ]

        # residual block
        for i, (in_channel, out_channel) in enumerate(zip(num_channels[1:-1], num_channels[2:])):
            block_list += [
                ResBlock(in_channel, out_channel, stride=2, use_1x1conv=True, ndims=self.ndims, norm=norm),
            ]
            for _ in range(num_blocks[i+1] - 1):
                block_list += [
                    ResBlock(out_channel, out_channel, stride=1, ndims=self.ndims, norm=norm),
                ]
        self.net = nn.Sequential(*block_list)

    def forward(self, x):
        return self.net(x)



