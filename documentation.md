# Repository Documentation <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [Folder Structure](#folder-structure)
- [pipeline.ipynb](#pipelineipynb)
- [README.md](#readmemd)
- [visualize\_scans.py](#visualize_scanspy)
- [RETOUCH Folder Structure](#retouch-folder-structure)

## Folder Structure
```bash
D
 ┣ environment # Contains the environment and requirements for this project
 ┃ ┣ environment.yml # File that contains the conda environment created
 ┃ ┣ requirements_full.txt # File with the complete requirements, that include 
 ┃ ┃                       # both the packages and the packages installed because of those packages
 ┃ ┗ requirements.txt # Contains the necessary packages for this project and their respective versions
 ┣ init # Folder that contains the Python files that are used before training the networks, to prepare the data
 ┃ ┣ __init__.py # Despite not having code in it, marks the folder as a possible library and allows its use in Jupyter
 ┃ ┣ foldsSplit.py # Has functions that perform k-fold-split on the RETOUCH dataset, according to the project needs
 ┃ ┣ patchExtraction.py # Has functions extract the patches later used to train the networks
 ┃ ┗ readOCT.py # Reads and saves the OCT's B-scans so that it can be saved in the user's computer
 ┣ splits # Will contain all the train-test splits
 ┃ ┣ segmentation_test_splits.csv # Contains the index of the volumes that will be used in the testing of the segmentation models
 ┃ ┗ segmentation_train_splits.csv # Contains the index of the volumes that will be used in the training of the segmentation models
 ┣ .gitignore # Declares the files that must not be updated to git
 ┣ documentation.md # Project documentation
 ┣ pipeline.ipynb # Project's pipeline code
 ┣ README.md # Front page of the project, used to orient the user
 ┗ visualize_scans.py # Simple UI for the user to visualize what is happening to the images in the processing
 ```

## pipeline.ipynb
Python notebook that contains the pipeline behind this project, from reading the images and saving, to the training of the networks. When this file is run, it does not need any other changes except some paths and options selected in the beginning. 

## README.md
Markdown file that introduces and orients the user in this project, introducing where to navigate to better understand it

## visualize_scans.py
File that, when ran, shows an UI that allows the user to select B-scans from the volumes in the training set of the RETOUCH dataset to visualize, showcasing the unaltered slice, the fluid masks, the entropy mask, and the ROI mask. 

## RETOUCH Folder Structure
Explains the folder structure of the RETOUCH dataset in order to compliment the code used in [foldsSplit.py](./init/foldsSplit.py) and [readOCT.py](./init/readOCT.py), allowing for better understanding and visualization.

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
 ┃                            # that compose the training set of the RETOUCH dataset
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
