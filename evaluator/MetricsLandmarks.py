import torch
import numpy as np


def cal_dis(pts_pred, pts_true, num_points, image_size=(384, 384), mode=('mean'), reduction='mean'):
    """ Calculate distance between point pairs.

    :param pts_pred: torch.tensor, (bs, 30, 2) or (bs, 30, 3)
    :param pts_true: torch.tensor, (bs, 30, 2) or (bs, 30, 3)
    :param num_points: torch.tensor, (bs,)
    :param image_size: list, (2,)
    :param mode: list, choose a subset from ['mean', 'median', 'max']
    :param reduction: str, choose from ['mean', 'none']
    :return: result -- a list contains the distances
    """

    image_size = torch.tensor(image_size, device=pts_pred.device).unsqueeze(0).unsqueeze(1) / 2
    dis = ((pts_pred[:, :, :2] - pts_true[:, :, :2]).mul(image_size)).pow(2).sum(dim=2).sqrt()
    result = []
    for m in mode:
        if m == 'mean':
            tmp = dis.sum(dim=1).divide(num_points).detach().cpu()
        elif m == 'max':
            tmp = dis.max(dim=1).values.detach().cpu()
        elif m == 'median':
            tmp = dis.clone()
            tmp = [tmp[i, :num_points[i]] for i in range(dis.shape[0])]
            tmp = [tmp[i].median() for i in range(dis.shape[0])]
            tmp = torch.tensor(tmp).detach().cpu()
        else:
            raise ValueError('Please choose mode from [\'mean\', \'median\', \'max\']')

        if reduction == 'mean':
            result.append(tmp.mean())
        elif reduction == 'none':
            result.append(tmp)
        else:
            raise ValueError('Please choose reduction mode from [\'mean\', \'none\']')

    return result


def analyze_results(dataframe):
    """ Analyze the results after getting the dataframe containing distances of all cases

    :param dataframe: the dataframe containing all cases
    :return:
    """
    success_before = np.logical_and(dataframe['Max Before'] <= 10, dataframe['Median Before'] <= 5)
    success_before = np.sum(success_before) / dataframe.shape[0]
    mean_before = dataframe['Mean Before']
    success_after = np.logical_and(dataframe['Max After'] <= 10, dataframe['Median After'] <= 5)
    success_after = np.sum(success_after) / dataframe.shape[0]
    mean_after = dataframe['Mean After']
    print('Success rate: before registration %.4f, after %.4f' % (success_before, success_after))
    print('Mean distance: before registration %.4f+-%.4f, after %.4f+-%.4f' %
          (np.mean(mean_before).item(), np.std(mean_before).item(),
           np.mean(mean_after).item(), np.std(mean_after).item()))
    print('25, 50, 75 percentile distance: before registration %.4f, %.4f, %.4f; after %.4f, %.4f, %.4f' %
          (np.percentile(mean_before, 25), np.percentile(mean_before, 50), np.percentile(mean_before, 75),
           np.percentile(mean_after, 25), np.percentile(mean_after, 50), np.percentile(mean_after, 75),))