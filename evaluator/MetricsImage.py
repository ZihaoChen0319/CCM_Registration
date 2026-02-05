import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_mae(img1, img2, mask=None):
	ndims = len(img1.shape) - 2
	dim_range = [*list(range(2, ndims + 2))]
	abs_error = torch.abs(img1 - img2)
	if mask is not None:
		abs_error = abs_error * mask
		score = torch.sum(abs_error, dim=dim_range) / torch.sum(mask, dim=dim_range)
	else:
		score = torch.mean(abs_error, dim=dim_range)
	return score


def compute_rmse(img1, img2, mask=None):
	ndims = len(img1.shape) - 2
	dim_range = [*list(range(2, ndims + 2))]
	square_error = (img1 - img2) ** 2
	if mask is not None:
		square_error = square_error * mask
		score = torch.sum(square_error, dim=dim_range) / torch.sum(mask, dim=dim_range)
	else:
		score = torch.mean(square_error, dim=dim_range)
	score = score ** 0.5
	return score


def compute_psnr(img1, img2, mask=None, data_range=1.0, eps=1e-8):
	ndims = len(img1.shape) - 2
	dim_range = [*list(range(2, ndims + 2))]
	square_error = (img1 - img2) ** 2
	if mask is not None:
		square_error = square_error * mask
		score = torch.sum(square_error, dim=dim_range) / torch.sum(mask, dim=dim_range)
	else:
		score = torch.mean(square_error, dim=dim_range)
	score = 10 * torch.log10(data_range ** 2 / (score + eps))
	return score


def gaussian(window_size, sigma):
	gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
	return gauss / gauss.sum()


def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window


def _ssim(img1, img2, window, window_size, channel, size_average=False):
	ndims = len(img1.shape) - 2

	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(dim=[*list(range(2, ndims + 2))])


class SSIM(torch.nn.Module):
	def __init__(self, window_size=11, size_average=True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compute_ssim(img1, img2, mask=None, window_size=7, size_average=False):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)

	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	if mask is not None:
		img1 = img1 * mask
		img2 = img2 * mask
	return _ssim(img1, img2, window, window_size, channel, size_average)

