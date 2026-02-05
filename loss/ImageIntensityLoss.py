import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_ncc(y_true, y_pred, win=9, mask=None):
    if mask is None:
        Ii = y_true
        Ji = y_pred
    else:
        Ii = y_true * mask
        Ji = y_pred * mask

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    if win is None:
        win = [9] * ndims
    else:
        win = [win] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win], device=Ii.device)
    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    if mask is None:
        return cc.mean(dim=(1, 2, 3))
    else:
        cc = cc * mask
        return cc.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred, mask=None):


        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(y_true.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            win = [9] * ndims
        else:
            win = [self.win] * ndims

        cc = cal_ncc(y_true, y_pred, win=self.win, mask=mask)

        return - cc.mean()


class MultiResolutionNCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3, kernel=3):
        super(MultiResolutionNCC, self).__init__()
        self.num_scale = scale
        self.kernel = kernel
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J, mask=None):
        ndims = len(list(I.size())) - 2
        total_NCC = []
        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J, mask)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())
            avg_pool = getattr(F, 'avg_pool%dd' % ndims)
            I = avg_pool(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = avg_pool(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            if mask is not None:
                mask = avg_pool(mask, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)


class MSE(nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred, mask=None):
        mse = (y_true - y_pred) ** 2
        if mask is None:
            return torch.mean(mse)
        else:
            mse = mse * mask
            return torch.sum(mse) / torch.sum(mask) / mse.shape[1]


class Grad(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def forward(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class MINDSSC(torch.nn.Module):
    def __init__(self, radius: int = 2, dilation: int = 2):
        """
        Implementation of the MIND-SSC loss function
        See http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for the MIND-SSC descriptor. This ignores the center voxel, but compares adjacent voxels of the 6-neighborhood with each other.
        See http://mpheinrich.de/pub/MEDIA_mycopy.pdf for the original MIND loss function.

        Implementation retrieved from
        https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        Annotation/Comments and support for 2d by Steffen Czolbe.

        Parameters:
            radius (int): determines size of patches around members of the 6-neighborhood
            dilation (int): determines spacing of members of the 6-neighborhood from the center voxel
        """
        super(MINDSSC, self).__init__()
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        # for a list of length N of interger-valued pixel-coordinates, return an NxN matrix containing the squred euclidian distance between them:
        # 0: coordinates are the same
        # 1: coordinates are neighbours
        # 2: coordinates are diagonal
        # 4: coordinates are opposide, 2 apart

        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC3d(self, img):
        B, C, H, W, D = img.shape
        assert H > 1 and W > 1 and D > 1, "Use 2d implementation for 2d data_backup"

        # Radius: determines size of patches around members of the 6-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1, 1]
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]  # 12x3 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]  # 12x3 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(img.device)  # shifting-kernels, one channel to 12 channels. Each 3x3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(self.dilation)  # Padding to account for borders.
        rpad2 = nn.ReplicationPad3d(self.radius)

        # compute patch-ssd

        # shift-align all 12 patch pairings, implemented via convolution
        h1 = F.conv3d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = F.conv3d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = F.avg_pool3d(diff, kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]  # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True)  # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(),
                               (mind_var.mean() * 1000).item())  # remove outliers
        mind = mind / mind_var
        mind = torch.exp(-mind)

        return mind

    def MINDSSC2d(self, img):
        # Radius: determines size of patches around members of the 4-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1]
        four_neighbourhood = torch.Tensor([[0, 1],
                                           [1, 0],
                                           [2, 1],
                                           [1, 2]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(four_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = four_neighbourhood.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :]  # 4x2 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = four_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]  # 4x2 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(4, 1, 3, 3).to(img.device)  # shifting-kernels, one channel to 4 channels. Each 3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(4) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        mshift2 = torch.zeros(4, 1, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(4) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1

        rpad1 = nn.ReplicationPad2d(self.dilation)  # Padding to account for borders.
        rpad2 = nn.ReplicationPad2d(self.radius)

        # compute patch-ssd

        # shift-align all 4 patch pairings, implemented via convolution
        h1 = F.conv2d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = F.conv2d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = F.avg_pool2d(diff, kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]  # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True)  # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(),
                               (mind_var.mean() * 1000).item())  # remove outliers
        mind = mind / mind_var
        mind = torch.exp(-mind)

        return mind

    def forward(self, y_pred, y_true, mask=None):
        # Get the MIND-SSC descriptor for each image
        if y_pred.dim() == 4:
            true = self.MINDSSC2d(y_true)
            pred = self.MINDSSC2d(y_pred)
        elif y_pred.dim() == 5:
            true = self.MINDSSC3d(y_true)
            pred = self.MINDSSC3d(y_pred)

        # calulate difference
        mse = (true - pred) ** 2
        if mask is None:
            return torch.mean(mse)
        else:
            mse = mse * mask
            return torch.sum(mse) / torch.sum(mask) / mse.shape[1]

