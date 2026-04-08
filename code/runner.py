from dataset import *
from trainer import *
from argparser import get_config_from_args
from config import Config
import numpy as np
import torch
from torchvision.transforms import Compose
import os

'''

    Main runner file.  To train the model, simply run this 
    file (python runner.py).  

    You can also pass CLI arguments to change the attributes
    in the Config class. Use the --help flag to see the
    fields.

'''


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(config: Config):

    '''

        If you are using a different dataset, please change the following 
        to your custom dataset class
        
    '''

    if config.exp_name == "LA":
        train_dataset = LAHeart(base_dir=config.root_dir, transforms=Compose([Augmentations3D(config)]), config=config)
        val_dataset = LAHeart(base_dir=config.root_dir, config=config, transforms=None, is_training=False)
    elif config.exp_name == "BraTS2019":
        train_dataset = BraTS2019(root_dir=config.path_train_dataset, 
                                    transforms=Compose([Augmentations3D(config)]),
                                    height=config.height, 
                                    width=config.width, 
                                    num_slices=config.depth,
                                    config=config,
                                    is_training=True)
        val_dataset = BraTS2019(root_dir=config.path_val_dataset,
                                    transforms=None,
                                    height=config.height, 
                                    width=config.width, 
                                    num_slices=config.depth,
                                    config=config,
                                    is_training=False)
    elif config.exp_name == "BraTS2024":
        train_dataset = BraTS2024(root_dir=config.path_train_dataset, 
                                    transforms=Compose([Augmentations3D(config)]),
                                    height=config.height, 
                                    width=config.width, 
                                    num_slices=config.depth,
                                    config=config,
                                    is_training=True)
        val_dataset = BraTS2024(root_dir=config.path_val_dataset,
                                    transforms=None,
                                    height=config.height, 
                                    width=config.width, 
                                    num_slices=config.depth,
                                    config=config,
                                    is_training=False)
    elif config.exp_name == "Pancreas":
        train_dataset = PancreasNIH(base_dir=config.root_dir, transforms=Compose([Augmentations3D(config)]), config=config)
        val_dataset = PancreasNIH(base_dir=config.root_dir, transforms=None, config=config, is_training=False)
    else:
        print("The experiment name is not recognized. If it is not one of the following:")
        print("[LA, BraTS2019, BraTS2024, Pancreas]")
        print("you will have to write your own dataset class and add it in runner.py")
        return None, None
    
    if config.labeled_ratio < 1.0:

        labeled_indices = list(range(0, config.labeled_num))
        unlabeled_indices = list(range(config.labeled_num, train_dataset.__len__()))
        batch_sampler = TwoStreamBatchSampler(labeled_indices, 
                                            unlabeled_indices, 
                                            config.total_batch_size, 
                                            config.total_batch_size - config.labeled_batch_size)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_sampler=batch_sampler, 
                                                       num_workers=config.num_workers)
    
    else:

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.labeled_batch_size, num_workers=config.num_workers)
        
    validation_dataloader = torch.utils.data.DataLoader(val_dataset,  batch_size=1, num_workers=config.num_workers, shuffle=False)

    return train_dataloader, validation_dataloader
    


if __name__ == "__main__":

    config_kwargs = get_config_from_args()
    config = Config(**config_kwargs)
    config.print_config_state()
    set_seed(config.seed)
    torch.cuda.empty_cache()
    train_dataloader, validation_dataloader = create_dataloaders(config)
    pwd = os.getcwd()
    if not os.path.isdir(f"{pwd}/Results"):
        os.makedirs(f"{pwd}/Results")
        print(f"Results/ directory created at location: {pwd}\n")
    if not os.path.isdir(f"{pwd}/Results/{config.exp_name}_{config.labeled_num}"):
        os.makedirs(f"{pwd}/Results/{config.exp_name}_{config.labeled_num}")
        print(f"Results/{config.exp_name}_{config.labeled_num}/ directory created at location: {pwd}\n")
    trainer = Trainer(config)
    trainer.semi_supervised_training_loop(train_dataloader, validation_dataloader)
