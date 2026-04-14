import torch
import os

class Config:

    '''

        Config class to set the values of different hyperparameters, data paths, etc.
        These values can be set via the command line, or you can change them here.

    '''

    def __init__(self, **kwargs):

        # LA, BraTS2019, BraTS2024, Pancreas
        self.exp_name = "LA"

        # Paths if dataset is explicitly split. If not, set both to None
        if self.exp_name in ["BraTS2019", "BraTS2024"]:
            self.path_train_dataset = f"../data/{self.exp_name}/train/"
            self.path_val_dataset = f"../data/{self.exp_name}/val/"
        else:
            # Root path if lists are provided instead of explicit split 
            # (train.list, test.list, etc.)
            if self.exp_name == 'Pancreas':
                self.root_dir = "../data/Pancreas"
            elif self.exp_name == "LA":
                self.root_dir = "../data/LA2018" 
            self.path_train_dataset = None
            self.path_val_dataset = None

        # unet3D, vnet
        self.architecture = 'vnet'

        # If center cropping is necessary (must be in the dataset class)
        self.height = None
        self.width = None
        self.depth = None

        self.patch_size = (80, 112, 112)  # (D, H, W)
        self.labeled_num = 8
        self.total_batch_size = 4
        self.labeled_batch_size = 2
        self.num_workers = 8
        self.max_iterations = 30000
        self.iteration_scaler = 150
        self.consistency = 0.1
        self.consistency_rampup = 200
        self.num_classes = 2
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.num_modalities = 1
        self.T = 5
        self.num_decoders = 4
        self.seed = 1337

        # Augmentation related configs
        self.a = 0.25
        self.b = 0.75
        self.cutmix_prob = 1.0
        self.cutmix_beta = 4.0

        # Inference related configs
        if self.exp_name == "LA":
            self.stride_xy = 18
            self.stride_z = 4
        elif self.exp_name == "Pancreas":
            self.stride_xy = 16
            self.stride_z = 16
        self.save_path = f"./Results/{self.exp_name}_{self.labeled_num}/best_model.pth"
        self.iteration = None
        self.load_checkpoint = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.path_train_dataset:
            self.num_training = len(os.listdir(self.path_train_dataset))
        elif self.root_dir:
            with open(self.root_dir + "/train.list", 'r') as f:
                image_list = f.readlines()
            self.num_training = len(image_list)
            del image_list

        self.labeled_ratio = self.labeled_num / self.num_training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def print_config_state(self):

        for name, value in self.__dict__.items():
            if value is None and name not in  [
                "iteration", 
                "load_checkpoint", 
                "path_train_dataset", 
                "path_val_dataset", 
                "root_dir",
                "height",
                "width",
                "depth"
            ]:
                print(f"In config.py, the {name} field has not been set! You can add it manually or use the CLI.")
                exit(1)
            else:
                print(f"{name:<20} : {value}")
        print("\n")





        
