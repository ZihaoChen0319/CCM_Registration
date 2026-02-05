import torch
import torch.nn.functional as nnf


def get_homo_coord(pts):
    """
    get homogeneous coordinates of input points
    :param pts: bs x 30 x 2
    :return: new_pts: bs x 30 x 3
    """
    homo_part = torch.ones((pts.shape[0], pts.shape[1], 1), device=pts.device).float()
    pts = torch.cat([pts, homo_part], dim=2)
    return pts


def transform(transform_matrix, points, num_points=None):
    """
    transform the points with homogeneous matrix
    :param transform_matrix: bs x 3 x 3
    :param points: bs x 30 x 3
    :param num_points: bs
    :return: pts_tf: bs x 30 x 3
    """
    s = points.shape[1]
    transform_matrix = torch.tile(transform_matrix.unsqueeze(1), (1, s, 1, 1))  # bs x 30 x 3 x 3
    points = points.unsqueeze(3)  # bs x 30 x 3 x 1
    pts_tf = torch.matmul(transform_matrix, points).squeeze(3)  # bs x 30 x 3
    if num_points is not None:
        mask = torch.zeros_like(pts_tf)
        for i in range(len(num_points)):
            mask[i, :num_points[i], :] = 1
    else:
        mask = torch.ones_like(pts_tf)
    mask[:, :, -1] = 1
    return pts_tf * mask


def matrix_to_flow_2d(matrix, shape, if_norm=True):
    """
    create sampling grids from affine matrix
    :param matrix: bs x 3 x 3
    :param shape: size of image
    :return: flow: bs x 2 x shape[0] x shape[1], the translation of each loc
    """
    tf_inv = torch.linalg.pinv(matrix)
    if if_norm:
        x = (torch.arange(start=0, end=shape[0], device=matrix.device) / shape[0] - 0.5) * 2
        y = (torch.arange(start=0, end=shape[1], device=matrix.device) / shape[1] - 0.5) * 2
    else:
        x = torch.arange(start=0, end=shape[0], device=matrix.device)
        y = torch.arange(start=0, end=shape[1], device=matrix.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    pts_matrix = torch.stack([xx, yy], dim=2)
    homo_part = torch.ones((shape[0], shape[1], 1), device=matrix.device)
    pts_matrix = torch.cat([pts_matrix, homo_part], dim=2).unsqueeze(3).unsqueeze(0)

    tf_inv = tf_inv.unsqueeze(1).unsqueeze(2)
    # tf_inv = torch.tile(tf_inv, (1, inshape[0], inshape[1], 1, 1))

    pts_warped = tf_inv.matmul(pts_matrix)
    flow = pts_warped.squeeze(-1) - pts_matrix.squeeze(-1)
    if if_norm:
        flow[:, :, :, 0] = flow[:, :, :, 0] * shape[0] / 2
        flow[:, :, :, 1] = flow[:, :, :, 1] * shape[1] / 2
    flow = flow[:, :, :, :2].permute(0, 3, 1, 2)

    return flow


def matrix_to_flow_3d(matrix, shape, if_norm=True):
    """
    create sampling grids from affine matrix
    :param matrix: bs x 4 x 4
    :param shape: size of image
    :return: flow: bs x 3 x shape[0] x shape[1], the translation of each loc
    """
    device = matrix.device
    tf_inv = torch.linalg.inv(matrix)  # bs x 4 x 4
    if if_norm:
        x = (torch.arange(start=0, end=shape[0], device=device) / shape[0] - 0.5) * 2
        y = (torch.arange(start=0, end=shape[1], device=device) / shape[1] - 0.5) * 2
        z = (torch.arange(start=0, end=shape[2], device=device) / shape[2] - 0.5) * 2
    else:
        x = torch.arange(start=0, end=shape[0], device=device)
        y = torch.arange(start=0, end=shape[1], device=device)
        z = torch.arange(start=0, end=shape[2], device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    pts_matrix = torch.stack([xx, yy, zz], dim=3)  # H x W x D x 3
    homo_part = torch.ones((*shape, 1), device=device)  # H x W x D x 1
    pts_matrix = torch.cat([pts_matrix, homo_part], dim=3).unsqueeze(4).unsqueeze(0)  # 1 x H x W x D x 4 x 1

    tf_inv = tf_inv.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # bs x 1 x 1 x 1 x 4 x 4

    pts_warped = tf_inv.matmul(pts_matrix)  # bs x H x W x D x 4 x 1
    flow = pts_warped.squeeze(-1) - pts_matrix.squeeze(-1)  # bs x H x W x D x 4
    if if_norm:
        flow[..., 0] = flow[..., 0] * shape[0] / 2
        flow[..., 1] = flow[..., 1] * shape[1] / 2
        flow[..., 2] = flow[..., 2] * shape[2] / 2
    flow = flow[..., :3].permute(0, 4, 1, 2, 3)

    return flow


# def matrix_to_flow_3d(matrix, shape, if_norm=True):
#     matrix_inv = torch.linalg.inv(matrix)
#     bs = matrix_inv.shape[0]
#     flow = nnf.affine_grid(matrix_inv[..., :3, :4], [bs, 1, *shape], align_corners=True)
#     return flow


def warp_image(image, flow, mode='bilinear'):
    """
    transform the points with homogeneous matrix
    :param transform_matrix: bs x 3 x 3
    :param image: bs x c x shape[0] x shape[1]
    :param mode: string
    :return: image_warped: same as image
    """
    shape = image.shape[2:]

    grids = torch.meshgrid([torch.arange(0, s) for s in shape])
    grids = torch.stack(grids)
    grids = torch.unsqueeze(grids, 0).type(torch.float32).to(image.device)  # 1 x 2 x shape[0] x shape[1]

    new_locs = grids + flow
    for i in range(len(shape)):
        new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    return nnf.grid_sample(image, new_locs, align_corners=True, mode=mode)


def cal_affine_matrix(points_src, points_tgt, num_keypoints):
    """Calculate the ground truth matrix

    Args:
        points_src: (bs, N, 3), torch.float32
        points_tgt: (bs, N, 3), torch.float32
        num_keypoints: (bs,), torch.int

    Return:
        affine_matrix: (bs, 3, 3), torch.float32

    """
    batch_size = points_src.shape[0]

    _points_src = points_src[..., :].clone()
    _points_tgt = points_tgt[..., :2].clone()
    for i in range(batch_size):
        _points_src[i, num_keypoints[i]:, :] = 0.
        _points_tgt[i, num_keypoints[i]:, :] = 0.

    M = torch.linalg.lstsq(_points_src, _points_tgt)[0]
    affine_matrix = torch.zeros((*M.shape[:-1], 1), device=M.device)
    affine_matrix[..., 2, 0] = 1.
    affine_matrix = torch.cat([M, affine_matrix], dim=-1).transpose(dim0=-2, dim1=-1)

    # _pt_coord_source = points_src[:num_keypoints, :]
    # _pt_coord_target = points_tgt[:num_keypoints, :2]
    # M = torch.linalg.lstsq(_pt_coord_source, _pt_coord_target)[0]
    # affine_matrix = torch.zeros(size=(3, 3), dtype=torch.float32)
    # affine_matrix[:2, :3] = M.T
    # affine_matrix[2, 2] = 1.

    return affine_matrix

