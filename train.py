from trainer.SATRNetTrainer import SATRNetTrainerCornea


if __name__ == '__main__':
    device = 'cuda:0'

    trainer = SATRNetTrainerCornea(
        run_name='default',
        path_data_folder='/path_data',
        fold=0, train_mode='landmark', batch_size=8, n_epochs=1200, learning_rate=1e-4, val_freq=100,
        loss_weights=(1., 0., 0., 0.), image_size=[384, 384], crop=None, down_sample_steps=0,
        num_iter_train=5, num_iter_test=5, prob_steps=[4, 1, 1, 1, 1],
        norm='BatchNorm', vit_depth_self=4, vit_depth_cross=4,
        num_channels_extractor=(1, 32, 64, 128, 256),
        num_blocks_extractor=(2, 2, 2, 2),
        device=device
    )

    trainer.full_training_loop()





