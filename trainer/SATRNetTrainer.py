import torch
import torch.optim as optim

from models.AffineTransformer import SATRNetCornea
from models.Module import para_to_matrix_affine_2d
from trainer.CorneaBasicAffineTrainer import BasicAffineTrainer


class SATRNetTrainerCornea(BasicAffineTrainer):
    def __init__(self, run_name='default', path_data_folder='/home/zihao/Drives/F/Data/111_Cornea_npz',
                 fold=0, train_mode='landmark', batch_size=16, n_epochs=3000, learning_rate=1e-5, val_freq=100,
                 loss_weights=(1., 1., 1.,), image_size=(384, 384), crop=None, down_sample_steps=None,
                 num_iter_train=1, num_iter_test=None, prob_steps=None,
                 norm='BatchNorm', vit_depth_self=4, vit_depth_cross=4,
                 num_channels_extractor=(1, 32, 64, 128, 256),
                 num_blocks_extractor=(2, 2, 2, 2),
                 device='cuda', **kwargs):
        super(SATRNetTrainerCornea, self).__init__(
            run_name=run_name, path_data_folder=path_data_folder, fold=fold, train_mode=train_mode,
            batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, val_freq=val_freq,
            loss_weights=loss_weights, image_size=image_size, crop=crop, down_sample_steps=down_sample_steps,
            num_iter_train=num_iter_train, num_iter_test=num_iter_test, prob_steps=prob_steps,
            device=device
        )

        self.affine_predictor = SATRNetCornea(
            image_size=(384, 384),
            num_channels_extractor=num_channels_extractor,
            num_blocks_extractor=num_blocks_extractor,
            num_heads=4, depth_self=vit_depth_self, depth_cross=vit_depth_cross,
            norm=norm
        )
        self.optimizer = optim.AdamW(self.affine_predictor.parameters(), lr=learning_rate, weight_decay=0.01)

        self.to(device)

    def model_forward(self, image_moving, image_fixed, **kwargs):
        para = self.affine_predictor(image_moving, image_fixed, kwargs['condition'])
        return para_to_matrix_affine_2d(para)


