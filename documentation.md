# Repository Documentation <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [Device Utilized](#device-utilized)
- [Folder Structure](#folder-structure)
- [paths.py](#pathspy)
- [pipeline.ipynb](#pipelineipynb)
- [plot\_logs.py](#plot_logspy)
- [README.md](#readmemd)
- [train.py](#trainpy)
- [visualize\_scans.py](#visualize_scanspy)
- [RETOUCH Folder Structure](#retouch-folder-structure)

## Device Utilized
In this section, the specifications of the device used in the experiments performed is explained. All the experiments were done using this device and, therefore, the computation time for each experiment can be compared.

**OS**
- Windows 10 1909

**Motherboard**
- MPG X570 GAMING PRO CARBON WIFI AM4

**RAM**
- 32 GB

**SSD**
- M.2 250GB Samsung 970 EVO plus NVMe
- 250 GB

**HDD**
- SEAGATE 2TB SATA3 256MB
- 2048 GB

**CPU**
- AMD RYZEN 9 3900X AM4
- Cores: 12

**GPU**
- Gigabyte NVIDIA GeForce RTX 3080 10 GB GDDR6X OC-10GD
- 10 GB

## Folder Structure
```bash
D
 ┣ environment # Contains the environment and requirements for this project
 ┃ ┣ environment_tf.yml # File that contains the conda environment created used when Tensorflow was used 
 ┃ ┣ environment_torch.yml # File that contains the conda environment created used when PyTorch was used 
 ┃ ┣ requirements_full_tf.txt # File with the complete requirements, that include 
 ┃ ┃                          # both the packages and the packages installed because of those packages,
 ┃ ┃                          # including Tensorflow
 ┃ ┣ requirements_full_torch.txt # File with the complete requirements, that include 
 ┃ ┃                          # both the packages and the packages installed because of those packages,
 ┃ ┃                          # including PyTorch
 ┃ ┗ requirements.txt # Contains the necessary packages for this project and their respective versions, 
 ┃                    # without including PyTorch or Tensorflow packages, which are installed manually 
 ┃                    # as described in the README.md file
 ┣ imgs # Folder that contains the relevant images output from other functions called in this project
 ┃ ┣ Run1_training_error.png # PNG file which contains the plots of the training and evaluation error in Run1  
 ┃ ┗ ...
 ┣ init # Folder that contains the Python files that are used before training the networks, to prepare the data
 ┃ ┣ __init__.py # Despite not having code in it, marks the folder as a possible library and allows its use in Jupyter
 ┃ ┣ folds_split.py # Has functions that perform k-fold-split on the RETOUCH dataset, according to the project needs
 ┃ ┣ mip_split.py # Using integer programming, attempts to determine the best fold split
 ┃ ┣ patch_extraction.py # Has functions extract the patches later used to train the networks
 ┃ ┗ read_oct.py # Reads and saves the OCT's B-scans so that it can be saved in the user's computer
 ┣ logs # Folder that contains the error logs of the runs
 ┃ ┣ Run1_training_log_epoch.csv # CSV file which saves the training and validation error in each epoch of a run
 ┃ ┃ ...
 ┃ ┗ Run1_training_log_batch.csv # CSV file which saves the training error in each batch of a run
 ┣ models # Folder that contains the PyTorch file of the best models in each run
 ┃ ┣ Run1_UNet_best_model.pth # PyTorch file of the best model in Run1, which performs multi-class fluid segmentation
 ┃ ┃ ...
 ┃ ┗ Run3_UNet_IRF_best_model.pth # PyTorch file of the best model in Run3, which performs binary segmentation on IRF
 ┣ network_functions # Folder that contains the Python files that contain the functions used in training
 ┃ ┣ __init__.py # Despite not having code in it, marks the folder as a possible library and allows its use in Jupyter
 ┃ ┣ dataset.py # Creates the PyTorch Dataset objects that will be used in train, test, and validation of the models
 ┃ ┗ evaluate.py # Function called to evaluate the model in each epoch
 ┣ networks # Folder that contains the Python files that contain the CNNs used in this project
 ┃ ┣ __init__.py # Despite not having code in it, marks the folder as a possible library and allows its use in Jupyter
 ┃ ┣ loss.py # Contains the loss functions that will be used to train and evaluate the models
 ┃ ┗ unet.py # U-Net model in PyTorch
 ┣ splits # Will contain all the train-test splits
 ┃ ┣ competitive_errors_fold0.csv # Contains the errors obtained in fold 0 of the competitive fold split
 ┃ ┃ ...
 ┃ ┣ competitive_errors_fold4.csv # Contains the errors obtained in fold 4 of the competitive fold split
 ┃ ┣ competitive_errors_mean.csv # Contains the mean of the errors obtained in the competitive fold split
 ┃ ┣ competitive_errors_std.csv # Contains the standard deviation of the errors obtained in the competitive fold split
 ┃ ┣ competitive_fold_selection.csv # Contains the fold split obtained using the competitive fold split
 ┃ ┣ factorial_errors_fold0.csv # Contains the errors obtained in fold 0 of the factorial fold split
 ┃ ┃ ...
 ┃ ┣ factorial_errors_fold4.csv # Contains the errors obtained in fold 4 of the factorial fold split
 ┃ ┣ factorial_errors_mean.csv # Contains the mean of the errors obtained in the factorial fold split
 ┃ ┣ factorial_errors_std.csv # Contains the standard deviation of the errors obtained in the factorial fold split
 ┃ ┣ factorial_fold_selection.csv # Contains the fold split obtained using the factorial fold split
 ┃ ┣ folds_selection.xlsx # Excel file where a manual analysis was made to determine which fold split was optimal
 ┃ ┣ generation_test_splits.csv # Contains the index of the volumes that will be used in the testing of the generative models
 ┃ ┣ generation_train_splits.csv # Contains the index of the volumes that will be used in the training of the generative 
 ┃ ┃                             # models
 ┃ ┣ manual_errors_fold0.csv # Contains the errors obtained in fold 0 of the fold split obtained manually
 ┃ ┃ ...
 ┃ ┣ manual_errors_fold4.csv # Contains the errors obtained in fold 4 of the fold split obtained manually
 ┃ ┣ manual_errors_mean.csv # Contains the mean of the errors obtained in the fold split obtained manually
 ┃ ┣ manual_errors_std.csv # Contains the standard deviation of the errors obtained in the fold split obtained manually
 ┃ ┣ manual_fold_selection.csv # Contains the fold split obtained using the fold split obtained manually
 ┃ ┣ segmentation_test_splits.csv # Contains the index of the volumes that will be used in the testing of 
 ┃ ┃                              # the segmentation models
 ┃ ┣ segmentation_train_splits.csv # Contains the index of the volumes that will be used in the training of 
 ┃ ┃                               # the segmentation models
 ┃ ┣ sortedfactorial_errors_fold0.csv # Contains the errors obtained in fold 0 of the factorial fold split, with the volumes  
 ┃ ┃ ...                              # sorted by the quantity of fluid voxels
 ┃ ┣ sortedfactorial_errors_fold4.csv # Contains the errors obtained in fold 4 of the factorial fold split, with the volumes 
 ┃ ┃                                  # sorted by the quantity of fluid voxels
 ┃ ┣ sortedfactorial_errors_mean.csv # Contains the mean of the errors obtained in the factorial fold split, with the volumes 
 ┃ ┃                                 # sorted by the quantity of fluid voxels
 ┃ ┣ sortedfactorial_errors_std.csv # Contains the standard deviation of the errors obtained in the factorial fold split, with
 ┃ ┃                                # the volumes sorted by the quantity of fluid voxels
 ┃ ┣ sortedfactorial_fold_selection.csv # Contains the fold split obtained using the factorial fold split, with the volumes 
 ┃ ┃                                    # sorted by the quantity of fluid voxels
 ┃ ┗ volumes_info.csv # Contains the number of voxels per fluid class of each volume, with their respective vendor and ordered 
 ┃                    # by their index
 ┣ .gitignore # Declares the files that must not be updated to git
 ┣ documentation.md # Project documentation
 ┣ pipeline.ipynb # Project's pipeline code
 ┣ plot_logs.py # Plots the training and validation errors of a run
 ┣ README.md # Front page of the project, used to orient the user
 ┣ train.py # File used to train the networks
 ┗ visualize_scans.py # Simple UI for the user to visualize what is happening to the images in the processing
 ```

## paths.py
File with the absolute paths required for this project. This changes from device to device and when changed, the following line must be ran in the Git Bash console to prevent it from updating in the repository.

```git update-index --no-assume-unchanged paths.py```

Afterwards, no change made to the file will be included in commits.

## pipeline.ipynb
Python notebook that contains the pipeline behind this project, from reading the images and saving, to the training of the networks. When this file is run, it does not need any other changes except some paths and options selected in the beginning. 

## plot_logs.py
Python script that plots and saves the loss in the training and validation of a specific run.

## README.md
Markdown file that introduces and orients the user in this project, introducing where to navigate to better understand it.

## train.py
In this file, the functions that train the implemented neural networks are present.

## visualize_scans.py
File that, when ran, shows an UI that allows the user to select B-scans from the volumes in the training set of the RETOUCH dataset to visualize, showcasing the unaltered slice, the fluid masks, the entropy mask, and the ROI mask. 

## RETOUCH Folder Structure
Explains the folder structure of the RETOUCH dataset in order to compliment the code used in [folds_split.py](./init/folds_split.py) and [read_oct.py](./init/read_oct.py), allowing for better understanding and visualization.

```bash
RETOUCH # Folder of the RETOUCH dataset
 ┣ RETOUCH-TestSet-Cirrus # Folder that contains OCT images obtained using the Cirrus device 
 ┃ ┃                      # that compose the testing set of the RETOUCH dataset
 ┃ ┣ TEST001 # Folder that contains the OCT images of the set 001 used in training
 ┣ ┃ ...
 ┃ ┗ TEST014
 ┃    ┣ oct.mhd # .mhd file that contains the information required to read the .raw OCT file 
 ┃    ┃         # using the SimpleITK library
 ┃    ┗ oct.raw # Raw image file of the OCT volume
 ┣ RETOUCH-TestSet-Spectralis # Folder that contains OCT images obtained using the Spectralis
 ┃                            # device that compose the testing set of the RETOUCH dataset
 ┣ RETOUCH-TestSet-Topcon # Folder that contains OCT images obtained using the Topcon device 
 ┃                        # that compose the testing set of the RETOUCH dataset
 ┣ RETOUCH-TrainingSet-Cirrus # Folder that contains OCT images obtained using the Cirrus device 
 ┃ ┃                          # that compose the training set of the RETOUCH dataset
 ┃ ┣ TRAIN001
 ┣ ┃ ...
 ┃ ┗ TRAIN024
 ┃    ┣ oct.mhd # .mhd file that contains the information required to read the .raw OCT file using 
 ┃    ┃         # the SimpleITK library
 ┃    ┣ oct.raw # Raw image file of the OCT volume
 ┃    ┣ reference.mhd # .mhd file that contains the information required to read the .raw OCT fluid
 ┃    ┃               # masks file using the SimpleITK library
 ┃    ┗ reference.raw # Raw image file of the OCT fluid (IRF, SRF, and PED) masks
 ┣ RETOUCH-TrainingSet-Spectralis # Folder that contains OCT images obtained using the Spectralis
 ┃                                # device that compose the training set of the RETOUCH dataset
 ┗ RETOUCH-TrainingSet-Topcon # Folder that contains OCT images obtained using the Topcon device that
                              # compose the training set of the RETOUCH dataset
 ```
