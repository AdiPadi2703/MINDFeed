import torch
import nibabel as nib
import numpy as np
import os
import random
from torch.utils.data.sampler import Sampler
import itertools
from randcolor import RandomBrightnessContrast
from config import Config
import h5py

class BraTS2019(torch.utils.data.Dataset):


    '''

        Dataset class for loading and preprocessing the BraTS 2019 dataset.

        Parameters:

        - root_dir : the path to the directory containing the dataset
        - transforms : any transformations to be performed on the input and output
        - height : desired remaining height after cropping
        - width : desired remaining width after cropping
        - num_slices: desired number of slices after cropping
        - config : config class instance
        - is_training : if True, patches are extracted

        Preprocessing Pipeline:

        cropping -> intensity normalization -> z normalization -> stacking modalities -> extract patches -> to tensor
        
    '''


    def __init__(self, root_dir, transforms, height, width, num_slices, config : Config, is_training=True):

        self.root_dir = root_dir
        self.patient_folders = sorted(os.listdir(root_dir))
        self.transforms = transforms
        self.height = height
        self.width = width
        self.num_slices = num_slices
        self.config = config
        self.is_training = is_training


    def __len__(self):

        return len(self.patient_folders)


    def __getitem__(self, idx):

        patient_folder = self.patient_folders[idx]
        patient_folder_path = self.root_dir + patient_folder
        modalities_nifti = os.listdir(patient_folder_path)
        modalities_nifti = sorted(modalities_nifti)
        
        flair, gt_mask, _, t1ce, t2 = [nib.load(patient_folder_path + '/' + nifti_volume).get_fdata() for nifti_volume in modalities_nifti]

        gt_mask[gt_mask == 4.] = 3.

        flair_cropped = self.cropping(flair)
        t1ce_cropped = self.cropping(t1ce)
        t2_cropped = self.cropping(t2)
        gt_cropped = self.cropping(gt_mask)

        flair_intensity_normalized = self.intensity_normalize(flair_cropped)
        t1ce_intensity_normalized = self.intensity_normalize(t1ce_cropped)
        t2_intensity_normalized = self.intensity_normalize(t2_cropped)

        flair_z = self.z_normalize(flair_intensity_normalized)
        t1ce_z = self.z_normalize(t1ce_intensity_normalized)
        t2_z = self.z_normalize(t2_intensity_normalized)

        if self.config.num_modalities == 1:
            input_volume = np.stack([flair_z])
        elif self.config.num_modalities == 2:
            input_volume = np.stack([flair_z, t1ce_z])
        elif self.config.num_modalities == 3:
            input_volume = np.stack([flair_z, t1ce_z, t2_z])

        if self.is_training:
            input_volume, gt_cropped = self.random_crop_3d(input_volume, gt_cropped)
        
        if self.transforms and self.is_training:
            sample = {}
            sample['input'] = input_volume
            sample['label'] = gt_cropped
            strong_input, weak_input, label = self.transforms(sample)
            return strong_input, weak_input, label
        
        input_volume = torch.from_numpy(input_volume).permute(0, 3, 1 ,2)
        output_volume = torch.from_numpy(gt_cropped).permute(2, 0, 1)

        return input_volume, output_volume


    def cropping(self, volume):

        H, W, D = volume.shape
        mid_height = H // 2
        mid_width = W // 2
        mid_slice = D // 2
        cropped_volume = volume[
            mid_height - int(self.height/2) : mid_height + int(self.height/2), 
            mid_width - int(self.width/2) : mid_width + int(self.width/2),
            mid_slice - int(self.num_slices/2) : mid_slice + int(self.num_slices/2)
        ]
        return cropped_volume


    def intensity_normalize(self, volume):

        volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
        return volume_normalized


    def z_normalize(self, volume):

        volume_normalized = (volume - volume.mean()) / volume.std()
        return volume_normalized
    
    def random_crop_3d(self, image, mask):
        D, H, W = image.shape[1:]
        pd, ph, pw = self.config.patch_size

        d = random.randint(0, D - pd)
        h = random.randint(0, H - ph)
        w = random.randint(0, W - pw)

        image_patch = image[:, d:d+pd, h:h+ph, w:w+pw]
        mask_patch = mask[d:d+pd, h:h+ph, w:w+pw]
        return image_patch, mask_patch


#############################################################################################

class BraTS2024(torch.utils.data.Dataset):


    '''

       Dataset class for loading and preprocessing the BraTS-GLI 2024 dataset.

        Parameters:

        - root_dir : the path to the directory containing the dataset
        - transforms : any transformations to be performed on the input and output
        - height : desired remaining height after cropping
        - width : desired remaining width after cropping
        - num_slices: desired number of slices after cropping
        - config : config class instance
        - is_training : is True, patches are extracted

        Preprocessing Pipeline:

        cropping -> intensity normalization -> z normalization -> stacking modalities -> patch extraction -> to tensor
        
    '''


    def __init__(self, root_dir, transforms, height, width, num_slices, config : Config, is_training=True):

        self.root_dir = root_dir
        self.patient_folders = sorted(os.listdir(root_dir))
        self.transforms = transforms
        self.height = height
        self.width = width
        self.num_slices = num_slices
        self.is_training = is_training
        self.config = config

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):

        patient_folder = self.patient_folders[idx]
        patient_folder_path = self.root_dir + patient_folder
        modalities_nifti = os.listdir(patient_folder_path)
        modalities_nifti = sorted(modalities_nifti)
        
        gt_mask, t1ce, _, flair, t2 = [nib.load(patient_folder_path + '/' + nifti_volume).get_fdata() for nifti_volume in modalities_nifti]

        flair_cropped = self.cropping(flair)
        t1ce_cropped = self.cropping(t1ce)
        t2_cropped = self.cropping(t2)
        gt_cropped = self.cropping(gt_mask)

        flair_intensity_normalized = self.intensity_normalize(flair_cropped)
        t1ce_intensity_normalized = self.intensity_normalize(t1ce_cropped)
        t2_intensity_normalized = self.intensity_normalize(t2_cropped)

        flair_z = self.z_normalize(flair_intensity_normalized)
        t1ce_z = self.z_normalize(t1ce_intensity_normalized)
        t2_z = self.z_normalize(t2_intensity_normalized)

        if self.config.num_modalities == 1:
            input_volume = np.stack([t1ce_z])
        elif self.config.num_modalities == 2:
            input_volume = np.stack([flair_z, t1ce_z])
        elif self.config.num_modalities == 3:
            input_volume = np.stack([flair_z, t1ce_z, t2_z])

        if self.is_training:
            input_volume, gt_cropped = self.random_crop_3d(input_volume, gt_cropped)
        
        if self.transforms and self.is_training:
            sample = {}
            sample['input'] = input_volume
            sample['label'] = gt_cropped
            strong_input, weak_input, label = self.transforms(sample)
            return strong_input, weak_input, label
        
        input_volume = torch.from_numpy(input_volume).permute(0, 3, 1 ,2)
        output_volume = torch.from_numpy(gt_cropped).permute(2, 0, 1)

        return input_volume, output_volume
    
    def cropping(self, volume):
        H, W, D = volume.shape
        mid_height = int( H // 2)
        mid_width = int(W // 2)
        mid_slice = int(D / 2)
        cropped_volume = volume[
            mid_height - int(self.height/2) : mid_height + int(self.height/2), 
            mid_width - int(self.width/2) : mid_width + int(self.width/2),
            mid_slice - int(self.num_slices/2) : mid_slice + int(self.num_slices/2)
        ]
        return cropped_volume

    def intensity_normalize(self, volume):
        volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
        return volume_normalized

    def z_normalize(self, volume):
        volume_normalized = (volume - volume.mean()) / volume.std()
        return volume_normalized
    
    def random_crop_3d(self, image, mask):
        D, H, W = image.shape[1:]
        pd, ph, pw = self.config.patch_size

        d = random.randint(0, D - pd)
        h = random.randint(0, H - ph)
        w = random.randint(0, W - pw)

        image_patch = image[:, d:d+pd, h:h+ph, w:w+pw]
        mask_patch = mask[d:d+pd, h:h+ph, w:w+pw]
        return image_patch, mask_patch
    
##############################################################################################

class LAHeart(torch.utils.data.Dataset):

    def __init__(self, base_dir, config, transforms=None, is_training=True):
        self.base_dir = base_dir
        self.sample_list = []
        self.is_training = is_training
        self.config = config
        self.transforms = transforms

        train_path = self.base_dir+'/train.list'
        test_path = self.base_dir+'/test.list'

        if self.is_training:
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        else:
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.is_training:
            image, label = self.random_crop_3d(image, label)
   
        image = np.expand_dims(image, axis=0) 
        if self.is_training and self.transforms:
            sample = {}
            sample['input'] = image
            sample['label'] = label
            weak_input, strong_input, label = self.transforms(sample)
            return weak_input, strong_input, label
        
        image = torch.from_numpy(image).permute(0, 3, 1 ,2)
        label = torch.from_numpy(label).permute(2, 0, 1)

        return image, label
    
    def random_crop_3d(self, image, mask):
        H, W, D = image.shape
        pd, ph, pw = self.config.patch_size

        d = random.randint(0, D - pd)
        h = random.randint(0, H - ph)
        w = random.randint(0, W - pw)

        image_patch = image[h:h+ph, w:w+pw, d:d+pd]
        mask_patch = mask[h:h+ph, w:w+pw, d:d+pd]
        return image_patch, mask_patch
    
##############################################################################################

class PancreasNIH(torch.utils.data.Dataset):

    def __init__(self, base_dir, config, transforms=None, is_training=True):
        self._base_dir = base_dir
        self.sample_list = []
        self.is_training = is_training
        self.config = config
        self.transforms = transforms

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if self.is_training:
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        else:
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/" + image_name + "_norm.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.is_training:
            image, label = self.random_crop_3d(image, label)

        image = np.expand_dims(image, axis=0) 
        if self.is_training and self.transforms:
            sample = {}
            sample['input'] = image
            sample['label'] = label
            weak_input, strong_input, label = self.transforms(sample)
            return weak_input, strong_input, label
    
        image = torch.from_numpy(image).permute(0, 3, 1 ,2)
        label = torch.from_numpy(label).permute(2, 0, 1)

        return image, label
    
    def random_crop_3d(self, image, mask):
        H, W, D = image.shape
        pd, ph, pw = self.config.patch_size

        d = random.randint(0, D - pd)
        h = random.randint(0, H - ph)
        w = random.randint(0, W - pw)

        image_patch = image[h:h+ph, w:w+pw, d:d+pd]
        mask_patch = mask[h:h+ph, w:w+pw, d:d+pd]
        return image_patch, mask_patch

##############################################################################################

class RandomRotFlip3D(object):

    '''

        Randomly rotate and flip the 3D volume and corresponding mask in a sample.

        Parameters:

        - volumes: 3D image ndarray of shape (C, H, W, D)
        - masks: 3D mask ndarray of shape (H, W, D)

    '''
   

    def __call__(self, volume, mask=None):
        
        k = np.random.randint(0, 4)  
        flip_h = np.random.rand() > 0.5
        flip_w = np.random.rand() > 0.5
        flip_d = np.random.rand() > 0.5

        volume = np.rot90(volume, k, axes=(1, 2)).copy()
        if mask is not None:
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        if flip_h:
            volume = np.flip(volume, axis=1).copy() 
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()
        
        if flip_w:
            volume = np.flip(volume, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
                
        if flip_d:
            volume = np.flip(volume, axis=3).copy()
            if mask is not None:
                mask = np.flip(mask, axis=2).copy()

        return volume, mask
    

class Augmentations3D(object):

    def __init__(self, config: Config):
        self.config = config
        self.random_rot_flip = RandomRotFlip3D()
        self.randcolor = RandomBrightnessContrast()

    def __call__(self, sample):

        input, label = sample['input'], sample['label']

        if self.config.exp_name not in ['Pancreas']:
            weak_input, label = self.random_rot_flip(input, label)
        else:
            weak_input = input

        if self.config.exp_name in ['Pancreas', 'LA']:
            strong_input = self.randcolor(weak_input)
        else:
            strong_input = weak_input

        sigma = random.uniform(self.config.a, self.config.b)
        rng = np.random.default_rng()
        strong_input = strong_input + sigma * rng.standard_normal(size=strong_input.shape)
      
        weak_input = torch.from_numpy(weak_input).permute(0, 3, 1 ,2)
        strong_input = torch.from_numpy(strong_input).permute(0, 3, 1 ,2)
        label = torch.from_numpy(label).permute(2, 0, 1)

        return weak_input, strong_input, label


#################################################################################################


'''

    Source for: 
    
        - TwoStreamBatchSampler
        - iterate_once
        - iterate_eternally
        - grouper 
        
    is https://github.com/HiLab-git/SSL4MIS

'''

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

##############################################################################################