# MINDFeed

This is the implementation for the paper titled:<br />
<b>MINDFeed: Mutual Information-Guided Single-Network Consistency Learning for Semi-Supervised 3D Glioma Segmentation</b>
<br />
<br />
<b>Please read this README before getting started.</b>

## Overview

The organization of this repository is as follows:

- The `code` directory contains the source code for the implementation.
- The `data` directory contains the `BraTS2019`, `BraTS2024`, `LA2018` and `Pancreas-CT` directories each of which contain files specifying the splits. You can use these to get the exact split that was used to obtain the results in the paper.

If you are using a <b>different dataset</b>, you can refer the dataset classes in `code/dataset.py` to write your own custom dataset class. You will also have to write your own validation loop (refer `code/validator.py`) based on the classes or regions of interest in your dataset, or use the `validation_loop_binary3D` function if doing binary segmentation.

We used an NVIDIA RTX 4060 8GB GPU for all our experiments. Memory requirements depend on the dataset being used. We recommend having at least 8GB of VRAM to comfortably run all datasets.

## Data

You can use the following links to download the datasets:

- <a href="https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019">BraTS 2019</a> 
- <a href="https://www.synapse.org/Synapse:syn53708249/files/">BraTS-GLI 2024</a>  
- <a href="https://github.com/himashi92/Co-BioNet/tree/main/data">LA 2018</a>  
- <a href="https://github.com/himashi92/Co-BioNet/tree/main/data">NIH Pancreas-CT</a>  

For BraTS 2019 and BraTS-GLI, there is a `script.py` file in their respective directories to create and fill the train, test and val directories as per the split lists.


## Training

To make use of this repository:

- First, clone the repository

```
git clone git@github.com:AdiPadi2703/MINDFeed.git
cd MINDFeed
```

- Next, create an environment.  If you are using `conda` then run the following with your desired environment name in place of `<env-name>`:

```
conda env create -f environment.yml -n <env-name>
conda activate <env-name>
```

- Open the `code/config.py` file and set your desired configurations. You can refer to the `dataset_configs.txt` file to see the configurations used in the paper. <b>This can also be done via the command line.</b> To see the list of fields, simply run

```
python runner.py --help
```

- Once you have configured `code/config.py`, start training with

```
python runner.py
```

## Evaluation

To evaluate the saved checkpoint, you can use the `code/validator.py` file on its own. We have provided an example case on LA 2018, and this example can be followed for other datasets by replacing the dataset class, changing the config parameters, and validator function. Once setup, just run:

```
python validator.py
```

## Citing

If you use our paper or this implementation in your work, please use the following to cite our work:

```


```

We are grateful to the authors of <a href="https://github.com/HiLab-git/SSL4MIS">SSL4MIS</a>, <a href="https://github.com/himashi92/Co-BioNet">Co-BioNet</a>, <a href="https://github.com/WYC-321/MCF">MCF, <a href="https://github.com/ZhenZHAO/AD-MT">AD-MT</a> and <a href="https://github.com/ortonwang/SGRS-Net">SGRS-Net</a> for their implementations. 
