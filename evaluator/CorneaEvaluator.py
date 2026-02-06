import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torchmetrics import MeanMetric, MetricCollection
import torch
import time
from thop import profile
import torch.nn as nn

from trainer.SATRNetTrainer import SATRNetTrainerCornea
from models.SpatialTransformer import SpatialTransformer
from loss.KeypointLoss import loss_func_keypoint_l1
from evaluator.MetricsSeg import compute_dice_torch
from evaluator.MetricsImage import compute_mae, compute_rmse, compute_psnr, compute_ssim, MutualInformation, nmi_batch_masked, nmi_batch_masked_torch
from evaluator.MetricsLandmarks import analyze_results, cal_dis
from utils.ImageTransform import transform, matrix_to_flow_2d, cal_affine_matrix


def evaluate_in_Cornea(
        trainer: SATRNetTrainerCornea,
        test_loader,
        if_plot=False,
        device='cuda'
    ):
    trainer.eval()

    # spatial transformer
    stn_linear = SpatialTransformer(size=(*trainer.down_sample_size,), mode='bilinear').to(device)
    stn_nearest = SpatialTransformer(size=(*trainer.down_sample_size,), mode='nearest').to(device)

    # test loop
    test_dis_after = []
    test_dis_before = []
    test_dis_manual = []
    test_name_list = []
    test_succeed_list = []
    dice_after_list = []
    dice_after_list_all = []
    dice_initial_list = []
    dice_initial_list_all = []
    dice_manual_list = []
    dice_manual_list_all = []
    total_time = 0.0
    compute_mi = nmi_batch_masked_torch
    # compute_mi = MutualInformation().eval().to(device)
    metrics_recorder = MetricCollection({
        'mae': MeanMetric(),
        'rmse': MeanMetric(),
        'psnr': MeanMetric(),
        'ssim': MeanMetric(),
        'mi': MeanMetric()
    })

    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = trainer.model_forward(
                image_moving=torch.randn(1, 1, *trainer.down_sample_size).to(device),
                image_fixed=torch.randn(1, 1, *trainer.down_sample_size).to(device),
                condition=torch.tensor([0, ] * 1, device=device)
            )
        torch.cuda.synchronize()

        loss_points, loss_matrix = 0., []
        tbar = tqdm(test_loader, ncols=80)
        for batch_idx, data_batch in enumerate(tbar):
            # transfer data_backup to gpu, tune the formats
            image_moving, image_fixed, mask_moving, mask_fixed, label_moving, label_fixed, \
            keypoints_moving, keypoints_fixed, num_keypoints = trainer.prepare_input(data_batch, crop=trainer.crop, all_label=True)

            # calculate the ground truth transformation matrix
            matrix_moving2fixed = cal_affine_matrix(keypoints_moving, keypoints_fixed, num_keypoints)
            matrix_fixed2moving = cal_affine_matrix(keypoints_fixed, keypoints_moving, num_keypoints)

            # forward and record intermediate image
            output = trainer.model_forward_iteratively(
                image_moving=image_moving, image_fixed=image_fixed, keypoints_moving=keypoints_moving, 
                keypoints_fixed=keypoints_fixed, num_keypoints=num_keypoints, label_moving=label_moving,
                if_record_intermediate=False
               )
            transform_matrix = output['transform_matrix']
            time_per_batch = output['time_per_batch']
            total_time += time_per_batch

            # transform the points
            keypoints_moving_warped = transform(transform_matrix, keypoints_moving, num_keypoints).detach()
            keypoints_moving_optimal = transform(matrix_moving2fixed, keypoints_moving, num_keypoints).detach()

            # warp the images, labels and masks
            flow_moving = matrix_to_flow_2d(transform_matrix, shape=trainer.down_sample_size)
            image_moving_warped = stn_linear(image_moving, flow_moving)
            label_moving_warped = stn_nearest(label_moving.detach(), flow_moving)
            mask_moving_warped = stn_nearest(mask_moving.detach(), flow_moving)
            flow = matrix_to_flow_2d(matrix_moving2fixed, shape=trainer.down_sample_size)
            label_moving_optimal = stn_nearest(label_moving.detach(), flow)
            mask_moving_optimal = stn_nearest(mask_moving.detach(), flow)

            # distance metric
            dis_before = cal_dis(keypoints_moving, keypoints_fixed, num_keypoints, image_size=trainer.image_size,
                                mode=['mean', 'median', 'max'])
            dis_after = cal_dis(keypoints_moving_warped, keypoints_fixed, num_keypoints, image_size=trainer.image_size,
                                mode=['mean', 'median', 'max'])
            dis_optimal = cal_dis(keypoints_moving_optimal, keypoints_fixed, num_keypoints, image_size=trainer.image_size,
                                mode=['mean', 'median', 'max'])

            mask = mask_moving_warped * mask_moving_optimal

            # segmentation metrics
            mean_dice, dice = compute_dice_torch(label_fixed.detach().cpu()[:, 0:1], label_moving.detach().cpu()[:, 0:1])
            dice_initial = mean_dice
            dice_initial_list_current = dice
            label_fixed_masked = (label_fixed * mask_moving_warped).detach().cpu()
            label_moving_warped_masked = (label_moving_warped * mask_moving_warped).detach().cpu()
            mean_dice, dice = compute_dice_torch(label_fixed_masked[:, 0:1], label_moving_warped_masked[:, 0:1])
            dice_after = mean_dice
            dice_after_list_current = dice

            label_fixed_masked = (label_fixed * mask_moving_optimal).detach().cpu()
            label_moving_optimal_masked = (label_moving_optimal * mask_moving_optimal).detach().cpu()
            mean_dice, dice = compute_dice_torch(label_fixed_masked[:, 0:1], label_moving_optimal_masked[:, 0:1])
            dice_manual = mean_dice
            dice_manual_list_current = dice

            # # surface distance metrics
            # hd95_initial, _ = compute_hd95(label_fixed.detach().cpu().numpy(), label_moving.detach().cpu().numpy(), labels=[0])
            # hd95_after, _ = compute_hd95(label_fixed_masked, label_moving_warped_masked, labels=[0])

            # image pixel level metrics
            result_per_case = {
                'mae': compute_mae(image_moving_warped, image_fixed, mask).detach().cpu(),
                'rmse': compute_rmse(image_moving_warped, image_fixed, mask).detach().cpu(),
                'psnr': compute_psnr(image_moving_warped, image_fixed, mask).detach().cpu(),
                'ssim': compute_ssim(image_moving_warped, image_fixed, mask).detach().cpu(),
                'mi': compute_mi(image_moving_warped, image_fixed, mask_moving_optimal).detach().cpu()
            }
            for k, v in result_per_case.items():
                metrics_recorder[k].update(v)

            # record the metrics
            test_name_list.append(' ')
            succeed_flag = 1 if dis_after < dis_before else 0
            test_succeed_list.append(succeed_flag)
            test_dis_after.append(np.array(dis_after))
            test_dis_before.append(np.array(dis_before))
            test_dis_manual.append(np.array(dis_optimal))
            dice_after_list.append(dice_after)
            dice_initial_list.append(dice_initial)
            dice_manual_list.append(dice_manual)
            dice_after_list_all += dice_after_list_current
            dice_initial_list_all += dice_initial_list_current
            dice_manual_list_all += dice_manual_list_current


        # analyze results
        dis_before_list = np.array(test_dis_before)
        dis_after_list = np.array(test_dis_after)
        dis_manual_list = np.array(test_dis_manual)
        df = pd.concat(
            [pd.Series(test_name_list), *[pd.Series(dis_before_list[:, i]) for i in range(dis_before_list.shape[1])],
            *[pd.Series(dis_after_list[:, i]) for i in range(dis_after_list.shape[1])], pd.Series(test_succeed_list)],
            axis=1)
        # df = pd.concat(
        #     [pd.Series(test_name_list), *[pd.Series(dis_before_list[:, i]) for i in range(dis_before_list.shape[1])],
        #     *[pd.Series(dis_manual_list[:, i]) for i in range(dis_manual_list.shape[1])], pd.Series(test_succeed_list)],
        #     axis=1)
        df.columns = ['ID', 'Mean Before', 'Median Before', 'Max Before', 'Mean After', 'Median After', 'Max After',
                    'Succeed Flag']
        analyze_results(df)

        print('Initial: Dice=%.4f+-%.4f, DSC30=%.4f' %
            (np.mean(dice_initial_list).item(), np.std(dice_initial_list).item(), np.percentile(dice_initial_list_all, q=30)))
        print('After: Dice=%.4f+-%.4f, DSC30=%.4f' %
            (np.mean(dice_after_list).item(), np.std(dice_after_list).item(), np.percentile(dice_after_list_all, q=30)))
        print('Manual: Dice=%.4f+-%.4f, DSC30=%.4f' %
            (np.mean(dice_manual_list).item(), np.std(dice_manual_list).item(), np.percentile(dice_manual_list_all, q=30)))
        
        result_metrics = metrics_recorder.compute()
        ordered_header = ['mae', 'rmse', 'psnr', 'ssim', 'mi']
        ordered_values = []
        for key in ordered_header:
            v = result_metrics[key]
            v = v.item() if hasattr(v, "item") else v
            ordered_values.append(f"{v:.4f}")   # format to 4 decimals
        print(*ordered_header, sep=",")
        print(*ordered_values, sep=",")

        # =======================================================
        # Inference Time, Params, FLOPS, Memory,
        # =======================================================
        # Inference Time
        num_images = len(test_loader)
        time_ms = total_time / num_images * 1000  # ms

        # Params, FLOPS
        class IterativeWrapper(nn.Module):
            def __init__(self, trainer: SATRNetTrainerCornea):
                super().__init__()
                self.trainer = trainer

            def forward(self, *inputs):
                return self.trainer.model_forward_iteratively(*inputs)
        iter_model = IterativeWrapper(trainer=trainer).eval()
        inputs = (image_moving, image_fixed, keypoints_moving, keypoints_fixed, num_keypoints, label_moving, False)
        flops, params = profile(iter_model, inputs=inputs, verbose=False)

        # Memory
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = iter_model(*inputs)
        memory = torch.cuda.max_memory_allocated(device) / 1024**3

        print(f"Params: {params/1e6:.4f} M")
        print(f"FLOPs: {flops/1e9:.4f} G")
        print(f"Memory: {memory:.4f} GB")
        print(f"Inference Time: {time_ms:.4f} ms")

        
        # =======================================================
        # Visualization (To Do)
        # =======================================================
        if if_plot and batch_idx < 30:
            pass