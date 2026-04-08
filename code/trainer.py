from validator import *
from networks.unet3D import UNet3D
import torch
from networks.vnet import VNet
from losses import *
from config import Config
from rampup import *
from cutmix import *
from tqdm import tqdm

class Trainer:

    '''
        Trainer class to train the model. 

        Parameters:

            - config : config class instance
    
    '''


    def __init__(self, config: Config):

        self.config = config
        self.load_checkpoint = config.load_checkpoint

        self.best_performance = 0.0

        if self.config.architecture == 'unet3D':
            print("3D UNet Selected...")
            self.model = UNet3D(
                n_classes=self.config.num_classes, 
                in_channels=self.config.num_modalities,
            ).to(self.config.device)
        elif self.config.architecture == 'vnet':
            print("VNet Selected...")
            self.model = VNet(
                n_channels=self.config.num_modalities, 
                n_classes=self.config.num_classes,
            ).to(self.config.device)

        if self.load_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.load_checkpoint, map_location=self.config.device))
            print(f"Model set to checkpoint: {self.load_checkpoint}")

        self.dice_loss = DiceLoss(num_classes=self.config.num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kl_distance = torch.nn.KLDivLoss(reduction="none")

        self.cutmix = CutMix3D(prop=self.config.cutmix_prob, beta=self.config.cutmix_beta)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        self.scaler = torch.amp.GradScaler(self.config.device)

    def compute_mi_map(self, predictions_list):

        avg_pred = torch.stack(predictions_list).mean(dim=0) 
        entropy = -torch.sum(avg_pred * torch.log(avg_pred + 1e-6), dim=1, keepdim=True)
        expected_entropy = torch.stack([
            -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
            for p in predictions_list
        ]).mean(dim=0)
        mi_map = entropy - expected_entropy
        if self.config.num_classes >= 4:
            mi_map = mi_map / math.log(self.config.num_classes)
        mi_map = torch.clamp(mi_map, 0.0, 1.0)
        return mi_map
    

    def semi_supervised_step(self, batch, iteration):

        weak_augmented_inputs, strong_augmented_inputs, augmented_masks = batch

        weak_augmented_inputs = weak_augmented_inputs.to(self.config.device)
        strong_augmented_inputs = strong_augmented_inputs.to(self.config.device)
        augmented_masks = augmented_masks.to(self.config.device)

        labeled_inputs = weak_augmented_inputs[:self.config.labeled_batch_size]
        labeled_masks = augmented_masks[:self.config.labeled_batch_size]
        
        weak_unlabeled_inputs = weak_augmented_inputs[self.config.labeled_batch_size:]
        strong_unlabeled_inputs = strong_augmented_inputs[self.config.labeled_batch_size:]

        consistency_weight = get_current_consistency_weight(iteration // self.config.iteration_scaler, self.config)

        self.model.train()

        with torch.amp.autocast(self.config.device):

            # Supervised Step
            labeled_out = self.model(labeled_inputs.float())
            deep_sup_labeled = labeled_out["deep_supervision"]
            deep_soft_labeled = [torch.softmax(x, dim=1) for x in deep_sup_labeled]

            loss_ce = sum(self.ce_loss(x, labeled_masks.long()) for x in deep_sup_labeled)
            loss_dice = sum(self.dice_loss(x, labeled_masks) for x in deep_soft_labeled)
            supervised_loss = (loss_ce + loss_dice) / (2 * self.config.num_decoders) 

            if self.config.labeled_ratio < 1.0:

                # Unsupervised Step
                with torch.no_grad():
                    unlabeled_outs = [self.model(weak_unlabeled_inputs.float()) for _ in range(self.config.T)]
                    unlabeled_soft = [[torch.softmax(x['deep_supervision'][i], dim=1) for i in range(self.config.num_decoders)] for x in unlabeled_outs]
                    mi_maps = [self.compute_mi_map([x[i] for x in unlabeled_soft]) for i in range(self.config.num_decoders)]
                    weak_ensemble_soft = torch.stack([torch.stack(x).mean(dim = 0) for x in unlabeled_soft]).mean(dim = 0)
                
                inputs = [strong_unlabeled_inputs, weak_ensemble_soft] + mi_maps
                mixed_inputs = self.cutmix(inputs)
                mix_strong_aug = mixed_inputs[0]
                mix_pseudo_labels = mixed_inputs[1]
                mix_mi_maps = mixed_inputs[2:]

                strong_out = self.model(mix_strong_aug.float(), mi_map=mix_mi_maps)
                strong_pred_soft = [torch.softmax(x, dim = 1) for x in strong_out["deep_supervision"]]

                consistency_loss = 0.0
                for i in range(self.config.num_decoders):

                    kl_map = torch.sum(self.kl_distance(strong_pred_soft[i].log(), mix_pseudo_labels) , dim=1, keepdim=True)
                    consistency_loss += torch.mean(kl_map)

                unsupervised_loss = (consistency_loss / self.config.num_decoders)

                loss = supervised_loss + consistency_weight * unsupervised_loss

                print(f"Iteration {iteration:<3} | "
                    f"Supervised Loss: {supervised_loss.item():<18.10f} | "
                    f"Unsupervised Loss: {unsupervised_loss.item():<18.10f} | "
                    f"Loss: {loss.item():<18.10f}"
                )
            
            else:

                loss = supervised_loss

                print(f"Iteration {iteration:<3} | "
                    f"Supervised Loss: {supervised_loss.item():<18.10f}"
                )

           
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            learning_rate = self.config.learning_rate * (1.0 - iteration / self.config.max_iterations) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate


    def semi_supervised_training_loop(self, train_dataloader, validation_dataloader):

        max_epochs = int(self.config.max_iterations // ((self.config.labeled_num) // self.config.labeled_batch_size))

        i = 0
        if self.config.iteration is not None:
            i = self.config.iteration
            
        for _ in tqdm(range(1, max_epochs + 1), ncols=70):
            print(" ")

            for batch in train_dataloader:

                i += 1
                self.semi_supervised_step(batch, i)
                if i % 1000 == 0:

                    '''
                    
                        If using a different dataset, please change the following line to
                        your custom validation function.

                    '''

                    if self.config.exp_name == "BraTS2019":
                        self.best_performance = validation_loop_brats2019(self.model, self.best_performance, validation_dataloader, self.config)
                    elif self.config.exp_name == "BraTS2024":
                        self.best_performance = validation_loop_brats2024(self.model, self.best_performance, validation_dataloader, self.config)
                    else:
                        self.best_performance = validation_loop_binary3D(self.model, self.best_performance, validation_dataloader, self.config)


    

    




            
