import torch


def loss_func_keypoint_l2(pts_pred, pts_true, num_points):
    """
    :param pts_pred: bs x 30 x 3
    :param pts_true: bs x 30 x 3
    :param num_points: bs
    :return: scalar
    """
    loss = torch.pow(pts_pred - pts_true, 2).sum(dim=2).sum(dim=1).divide(num_points).mean()
    # loss = torch.sqrt(torch.pow(pts_pred / 1536 - pts_true / 1536, 2).sum(dim=2)).sum(dim=1).divide(num_points)
    # loss = loss.mean()
    return loss


def loss_func_keypoint_l1(pts_pred, pts_true, num_points=None, divide_point_num=True):
    """
    :param pts_pred: bs x 30 x 3
    :param pts_true: bs x 30 x 3
    :param num_points: bs
    :return: scalar
    """
    if divide_point_num is True and num_points is not None:
        loss = torch.abs(pts_pred - pts_true).sum(dim=2).sum(dim=1).divide(num_points).mean()
    else:
        loss = torch.abs(pts_pred - pts_true).sum(dim=2).sum(dim=1).mean()
    return loss
