import math

import torch


def generate_random_affine_matrix_2d(
        rotation=(-15, 15),
        translation=(-0.2, 0.2, -0.2, 0.2),
        scale=(0.95, 1.05, 0.95, 1.05),
        shear=(-0.1, 0.1, -0.1, 0.1),
        n=1,
        device='cpu'
):
    rotation = torch.rand(size=(n,)) * (rotation[1] - rotation[0]) + rotation[0]
    rotation = rotation / 180 * torch.pi
    translation_x = torch.rand(size=(n,)) * (translation[1] - translation[0]) + translation[0]
    translation_y = torch.rand(size=(n,)) * (translation[3] - translation[2]) + translation[2]
    scale_x = torch.rand(size=(n,)) * (scale[1] - scale[0]) + scale[0]
    scale_y = torch.rand(size=(n,)) * (scale[3] - scale[2]) + scale[2]
    shear_x = torch.rand(size=(n,)) * (shear[1] - shear[0]) + shear[0]
    shear_y = torch.rand(size=(n,)) * (shear[3] - shear[2]) + shear[2]

    rotation_matrix = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)
    translation_matrix = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)
    scale_matrix = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)
    shear_matrix = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)

    rotation_matrix[:, 0, 0] = torch.cos(rotation)
    rotation_matrix[:, 0, 1] = -torch.sin(rotation)
    rotation_matrix[:, 1, 0] = torch.sin(rotation)
    rotation_matrix[:, 1, 1] = torch.cos(rotation)
    rotation_matrix[:, 2, 2] = 1.

    translation_matrix[:, 0, 0] = 1.
    translation_matrix[:, 1, 1] = 1.
    translation_matrix[:, 2, 2] = 1.
    translation_matrix[:, 0, 0] = 1.
    translation_matrix[:, 0, 2] = translation_x
    translation_matrix[:, 1, 2] = translation_y

    scale_matrix[:, 0, 0] = scale_x
    scale_matrix[:, 1, 1] = scale_y
    scale_matrix[:, 2, 2] = 1.

    shear_matrix[:, 0, 0] = 1.
    shear_matrix[:, 1, 1] = 1.
    shear_matrix[:, 2, 2] = 1.
    shear_matrix[:, 0, 1] = shear_y
    shear_matrix[:, 1, 0] = shear_x

    return translation_matrix.matmul(rotation_matrix).matmul(scale_matrix).matmul(shear_matrix)


def generate_random_affine_matrix_3d(
        rotation=(-15, 15, -15, 15, -15, 15),
        translation=(-0.2, 0.2, -0.2, 0.2, -0.2, 0.2),
        scale=(0.95, 1.05, 0.95, 1.05, 0.95, 1.05),
        shear=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1),
        n=1,
        device='cpu'
):
    rotation_x = torch.rand(size=(n,), device=device) * (rotation[1] - rotation[0]) + rotation[0]
    rotation_x = rotation_x / 180 * math.pi
    rotation_y = torch.rand(size=(n,), device=device) * (rotation[3] - rotation[2]) + rotation[2]
    rotation_y = rotation_y / 180 * math.pi
    rotation_z = torch.rand(size=(n,), device=device) * (rotation[5] - rotation[4]) + rotation[4]
    rotation_z = rotation_z / 180 * math.pi
    translation_x = torch.rand(size=(n,), device=device) * (translation[1] - translation[0]) + translation[0]
    translation_y = torch.rand(size=(n,), device=device) * (translation[3] - translation[2]) + translation[2]
    translation_z = torch.rand(size=(n,), device=device) * (translation[5] - translation[4]) + translation[4]
    scaling_x = torch.rand(size=(n,), device=device) * (scale[1] - scale[0]) + scale[0]
    scaling_y = torch.rand(size=(n,), device=device) * (scale[3] - scale[2]) + scale[2]
    scaling_z = torch.rand(size=(n,), device=device) * (scale[5] - scale[4]) + scale[4]
    shearing_x = torch.rand(size=(n,), device=device) * (shear[1] - shear[0]) + shear[0]
    shearing_y = torch.rand(size=(n,), device=device) * (shear[3] - shear[2]) + shear[2]
    shearing_z = torch.rand(size=(n,), device=device) * (shear[5] - shear[4]) + shear[4]

    mat_translation = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)
    mat_rotation_x = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)
    mat_rotation_y = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)
    mat_rotation_z = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)
    mat_shearing = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)
    mat_scaling = torch.eye(4, device=device, dtype=torch.float).unsqueeze(0).repeat(n, 1, 1)

    mat_translation[:, 0, 3] = translation_x
    mat_translation[:, 1, 3] = translation_y
    mat_translation[:, 2, 3] = translation_z

    mat_scaling[:, 0, 0] = scaling_x
    mat_scaling[:, 1, 1] = scaling_y
    mat_scaling[:, 2, 2] = scaling_z

    mat_rotation_x[:, 1, 1] = torch.cos(rotation_x)
    mat_rotation_x[:, 1, 2] = -torch.sin(rotation_x)
    mat_rotation_x[:, 2, 1] = torch.sin(rotation_x)
    mat_rotation_x[:, 2, 2] = torch.cos(rotation_x)

    mat_rotation_y[:, 0, 0] = torch.cos(rotation_y)
    mat_rotation_y[:, 0, 2] = torch.sin(rotation_y)
    mat_rotation_y[:, 2, 0] = -torch.sin(rotation_y)
    mat_rotation_y[:, 2, 2] = torch.cos(rotation_y)

    mat_rotation_y[:, 0, 0] = torch.cos(rotation_z)
    mat_rotation_y[:, 0, 1] = -torch.sin(rotation_z)
    mat_rotation_y[:, 1, 0] = torch.sin(rotation_z)
    mat_rotation_y[:, 1, 1] = torch.cos(rotation_z)

    mat_rotation = mat_rotation_z.matmul(mat_rotation_y).matmul(mat_rotation_x)

    mat_shearing[:, 0, 1] = shearing_x
    mat_shearing[:, 0, 2] = shearing_y
    mat_shearing[:, 1, 2] = shearing_z

    matrix = mat_shearing.matmul(mat_scaling).matmul(mat_rotation).matmul(mat_translation)

    return matrix


def random_augment(batch_size, p, device):
    p_aug = torch.rand(size=(batch_size,))
    idx_aug = torch.where(torch.less(p_aug, p))[0]

    affine_matrix_random = torch.eye(3, device=device).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    affine_matrix_random[idx_aug] = generate_random_affine_matrix_2d(
        rotation=(-15, 15),
        translation=(-0.3, 0.3, -0.3, 0.3),
        scale=(0.95, 1.05, 0.95, 1.05),
        shear=(-0.1, 0.1, -0.1, 0.1),
        n=idx_aug.shape[0],
        device=device
    )

    return affine_matrix_random

