import argparse

'''

    Argparser function for setting attributes in the config class from the CLI.


'''

def get_config_from_args():

    parser = argparse.ArgumentParser(description="CLI config for MINDFeed")

    # Paths, Names, General Configs
    parser.add_argument('--exp_name', type=str, metavar='EXP',
                        help='Experiment name (e.g., LA, BraTS2019, BraTS2024, Pancreas)')
    parser.add_argument('--path_train_dataset', type=str, metavar='TRAIN',
                        help='Path to the training dataset directory')
    parser.add_argument('--path_val_dataset', type=str, metavar='VAL',
                        help='Path to the validation dataset directory')
    parser.add_argument('--save_path', type=str, metavar='SAVEPATH',
                        help='Path to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, metavar='LOAD',
                        help='Load a checkpoint from this path')
    
    # Model Related Configs
    parser.add_argument('--architecture', type=str, metavar='ARCH',
                        help='Model architecture (e.g., unet3D, vnet)')
    parser.add_argument('--num_classes', type=int, metavar='C',
                        help='Number of segmentation classes (including background)')
    parser.add_argument('--num_modalities', type=int, metavar='MOD',
                        help='Number of input image modalities')
    parser.add_argument('--num_decoders', type=int, metavar='BLOCKS',
                        help='Number of decoder blocks')

    # Data Related
    parser.add_argument('--height', type=int, metavar='H', help='Input height')
    parser.add_argument('--width', type=int, metavar='W', help='Input width')
    parser.add_argument('--depth', type=int, metavar='D', help='Input depth')
    parser.add_argument('--patch_size', type=int, nargs=3, metavar=('D', 'H', 'W'),
                        help='Patch size for training (Depth, Height, Width)')

    # Training Hyperparameters
    parser.add_argument('--labeled_num', type=int, metavar='LAB',
                        help='Number of labeled samples to use')
    parser.add_argument('--total_batch_size', type=int, metavar='TB',
                        help='Total batch size (labeled + unlabeled)')
    parser.add_argument('--labeled_batch_size', type=int, metavar='LB',
                        help='Batch size of labeled samples')
    parser.add_argument('--max_iterations', type=int, metavar='ITERMAX',
                        help='Maximum number of training iterations')
    parser.add_argument('--iteration', type=int, metavar='ITERSTART',
                        help='Start training from this iteration')
    parser.add_argument('--learning_rate', type=float, metavar='LR',
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, metavar='M',
                        help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, metavar='L2',
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, metavar='SEED', help='Random seed')
    parser.add_argument('--num_workers', type=int, metavar='WORKERS',
                        help='Number of data loading threads')
    parser.add_argument('--consistency', type=float, metavar='CONS',
                        help='Consistency weight')
    parser.add_argument('--consistency_rampup', type=int, metavar='RAMP',
                        help='Iterations for consistency weight rampup')
    parser.add_argument('--iteration_scaler', type=int, metavar='ITERSCALER',
                        help='Value to scale iterations for sigmoid rampup')
    parser.add_argument('--T', type=int, metavar='T',
                        help='Number of MC Dropout steps')

    # Augmentation Configs
    parser.add_argument('--a', type=float, metavar='UA', help='Lower limit of uniform distribution')
    parser.add_argument('--b', type=float, metavar='UB', help='Upper limit of uniform distribution')
    parser.add_argument('--cutmix_prob', type=float, metavar='CMP', help='Probability of CutMix')
    parser.add_argument('--cutmix_beta', type=float, metavar='CMB', help='Beta distribution for CutMix')

    # Inference Configs
    parser.add_argument('--stride_xy', type=int, metavar='SXY', help='Stride in XY plane for sliding window')
    parser.add_argument('--stride_z', type=int, metavar='SZ', help='Stride in Z axis for sliding window')

    args = parser.parse_args()

    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}

    return config_kwargs