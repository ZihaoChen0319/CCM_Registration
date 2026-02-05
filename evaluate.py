import os
import sys
from torch.utils.data import DataLoader
import json

from data.CorneaDataLoader import CorneaDataset
from trainer.SATRNetTrainer import SATRNetTrainerCornea
from evaluator.CorneaEvaluator import evaluate_in_Cornea


PROJECT_PATH = os.path.dirname(__file__)


if __name__ == '__main__':
    # parameters
    if_plot = False
    device = 'cuda:0'
    fold = 0
    path_data_folder = '/path_data'
    path_result_folder = 'model_name'
    path_result = os.path.join('../result', path_result_folder, 'fold_%d' % fold)

    # load the trainer,
    trainer = SATRNetTrainerCornea(
        run_name='test',
        path_data_folder=path_data_folder,
        fold=0, train_mode='vxm', batch_size=1, n_epochs=2000, learning_rate=1e-4, val_freq=100,
        loss_weights=(1., 0., 0., 0.), image_size=[384, 384], crop=None, down_sample_steps=0,
        num_iter_train=1, num_iter_test=1, prob_steps=None,
        norm='BatchNorm', vit_depth_cross=4, vit_depth_self=4, 
        num_channels_extractor=(1, 32, 64, 128, 256),
        num_blocks_extractor=(2, 2, 2, 2),
        device=device)
    path_ckpt = os.path.join(PROJECT_PATH, 'result', path_result_folder, 'fold_%d' % fold, 'checkpoint_best.pth')
    trainer.load_checkpoint(path_ckpt=path_ckpt)
    trainer.eval()

    # data loader
    out_keys = ['image', 'label', 'mask', 'keypoints', 'num_keypoints', 'matrix']
    aug_idx_list_test = [0]
    test_set = CorneaDataset(path_folder=path_data_folder, mode='test', fold=fold, out_keys=out_keys,
                             aug_idx_list=aug_idx_list_test, contain_origin=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    
    evaluate_in_Cornea(
        trainer=trainer,
        test_loader=test_loader,
        if_plot=if_plot,
        device=device
    )

    

    




