# Introduction
This repository contains all the code and work done throughout my dissertation, entitled "Characterization of Retinal Fluid in Optical Coherence Tomography". For more informations regarding this theme and its state-of-the-art, please read the full-sized PDF. All the experiments that took place during the dissertation, using this code is also there explained. Some code here used is inspired by the work of Tennakoon et al., 2018 [1]. To understand the code that was implemented, please check the [documentation](documentation.md). To follow the implementation, please consider understanding the [pipeline file](pipeline.ipynb).

# Tensorflow Installation
The code here implemented was ran in a Windows device, requiring some caution with the installation of Tensorflow. This installation was done following the [Tensorflow installation guide](https://www.tensorflow.org/install/pip?hl=en#windows-native_1). This is only required to run the code using the device GPU, since the use of the CPU only requires the base Tensorflow.
1. Install Microsoft Visual C++ Redistributable
- Install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019. Starting with the TensorFlow 2.1.0 version, the  ```msvcp140_1.dll``` file is required from this package (which may not be provided from older redistributable packages). The redistributable comes with Visual Studio 2019 but can be installed separately:
  1. Go to the [Microsoft Visual C++ downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
  2. Scroll down the page to the Visual Studio 2015, 2017 and 2019 section.
  3. Download and install the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 for your platform.
  4. Make sure [long paths are enabled](https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing) on Windows

2. Install Miniconda  
- [Miniconda](https://docs.anaconda.com/miniconda/) is the recommended approach for installing TensorFlow with GPU support. It creates a separate environment to avoid changing any installed software in your system. This is also the easiest way to install the required software especially for the GPU setup.
- Download the [Miniconda Windows Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). Double-click the downloaded file and follow the instructions on the screen.

3. Create a conda environment
- Create a new conda environment named ```tf``` with the following command.
  ```conda create --name tf python=3.9```
- You can deactivate and activate it with the following commands.
  ```conda deactivate```
  ```conda activate tf```
- Make sure it is activated for the rest of the installation.

4. GPU Setup
- You can skip this section if you only run TensorFlow on CPU.
- First install [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) if you have not.
- Then install the CUDA, cuDNN with conda.
  ```conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0```

5. Install TensorFlow
- TensorFlow requires a recent version of pip, so upgrade your pip installation to be sure you're running the latest version.
  ```pip install --upgrade pip```
- Then, install TensorFlow with pip. The version installed was 2.10.1.
  ```pip install "tensorflow<2.11"```

6. Verify the installation
- Verify the CPU setup:
  ```python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"```
- Note: I believe that the first time I did this, an error occured due to the wrong version of NumPy. If so happens, please consider:
  ```pip install "numpy<2"```
- Then, you can run the code again. The NumPy version must be below 2.x so that it is compatible with the Tensorflow version.
- If a tensor is returned, you've installed TensorFlow successfully.
- Verify the GPU setup:
  ```python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```
- If a list of GPU devices is returned, you've installed TensorFlow successfully.

7. Packages installation
- The packages I used in this code and their respective versions are described in [requirements.txt](environment/requirements.txt). To install them, please run the following command, with the conda environment activated:
 ```pip install -r environment/requirements.txt```
- More information on the packages used as a consequence of those installed, you can see: [requirements_full_tf.txt](environment/requirements_full_tf.txt). The environment specifications are available at [environment_tf.yml](environment/environment_tf.yml).

# PyTorch Installation
Another environment was created to support PyTorch. Since the required versions of the CUDA toolkit used in PyTorch and Tensorflow are incompatible (PyTorch latest version requires the CUDA version 11.8 or higher while Tensorflow requires the 11.2 CUDA version). The steps prior to the environment creation (Driver update and Anaconda installation) are the same for the Tensorflow and the PyTorch installation.

1. Create a conda environment
- Create a new conda environment named ```torch``` with the following command.
  ```conda create --name torch python=3.9```
- You can deactivate and activate it with the following commands.
  ```conda deactivate```
  ```conda activate torch```
- Make sure it is activated for the rest of the installation.

2. GPU Setup
- You can skip this section if you only run PyTorch on CPU.
- First install [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) if you have not.
- Then install the CUDA, cuDNN with conda. cuDNN version is not needed to specify since the latest available is good enough.
  ```conda install -c conda-forge cudatoolkit=11.8 cudnn```

3. Install PyTorch
- PyTorch installation was done using conda and the version is obtained according to the CUDA version installed, which was 11.8. The PyTorch version installed was 2.5.1.
  ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

4. Verify the installation
- Run ```python``` on the Conda console. Then, run the following commands:
- ```import torch``` ```x = torch.rand(5, 3)``` ```print(x)```
- A PyTorch tensor is expected to be printed.
- To check the GPU setup:
  ```torch.cuda.is_available()```
- It must return 1.
- ```torch.cuda.current_device()```
- It must return the index of the device that is being used, which in my case is ```0```.
- ```torch.cuda.get_device_name(0)```
- The name of the GPU must be returned in this case.

5. Packages Installation
- The packages I used in this code and their respective versions are described in [requirements.txt](environment/requirements.txt). To install them, please run the following command, with the conda environment activated: ```pip install -r environment/requirements.txt```
- More information on the packages used as a consequence of those installed, you can see: [requirements_full_torch.txt](environment/requirements_full_torch.txt). The environment specifications are available at [environment_torch.yml](environment/environment_torch.yml).

# Storage Requirements
The dataset used for the experiments here described was the RETOUCH dataset. This is a dataset that is composed of OCT volumes with segmented retinal fluid (IRF, SRF, and PED). The training folder includes the segmentation masks while the testing folder does not. The training folder comprises 70 OCT volumes: 24 obtained with Cirrus, 24 obtained with Spectralis, and 22 obtained with Topcon (2 with the T-1000 and 20 with the T-2000 device). 

The Cirrus volume has 128 slices with a size of 512 x 1024 (height x width) pixels, while each Spectralis volume has 49 slices with a size of 512 x 885 pixels. The volumes obtained with the T-1000 devices produce volumes with 64 slices that have a size of 512 x 650 (h x w) pixels, while those obtained with the T-2000 can have 128 or 64 slices with a size of 512 x 885 pixels. 

The Cirrus and Topcon image's pixels are stored in uint8. The Cirrus volumes occupy 64MB, $1024*512*128 = 67 108 864$ binary bytes ($64$ MB in the decimal scale) and so does their respective masks. The Topcon T-2000 volumes occupy 55.3MB, $512*885*128 = 57 999 360$ binary bytes. The volumes obtained with Topcon T-1000 can occupy 40.6MB when containing 128 slices, $512*650*128 = 42 598 400$ binary bytes or 27.6MB ($512*650*64 = 21 299 200$ binary bytes) when containing 64 slices. The metaheader files in the Topcon folders occupy. The Spectralis volumes have 49 slices with 512 x 496 pixels, stored in uint16. Therefore, the space occupied by each volume is $49*512*496*16/8=24 887 296$ binary bytes or 23.7MB in decimal binary. The space occupied by the metaheader files is significantly smaller than those in the images and therefore will not be taken into account in further calculations.

In the training set, there are 12 volumes obtained with the T-1000, 2 of which have 64 slices. The remaining 10 were obtained with the T-2000. Therefore, the total storage required for the training set of the RETOUCH dataset is $(67 108 864*24 + 57 999 360*10 + 42 598 400*10 + 21 299 200*2)*2 + 24*24 887 296 + 24*24 887 296/2 = 6.1$ GiB ou $5.8$ GB.

For each OCT volume, the constituting slices will be saved both in int32 and uint8. The reason int32 is being saved is because it will allow more precise calculations when handling the images. Meanwhile, the uint8 images will be saved to allow a visualization in the folders. This will be done not only for the OCT volumes, but also for the respective segmentation masks and ROI saved. The space occupied by all this images will be:
- $67 108 864*24*(5 + 4) = 14 495 514 624$ bytes or $13.5$ GB  in Cirrus
- $(57 999 360*10 + 42 598 400*10 + 21 299 200*2)*(5 + 4) = 9 437 184 000$ bytes or $8.8$ GB in Topcon
- $(49*512*496*24)*(5 + 4) = 2 687 827 968$ bytes or $2.5$ GB in Spectralis

In case the patches are saved previously to the training, 12 patches per image with size 256 x 128 in uint8 will be saved. The total number of patches saved would be $(24*128+24*49+20*128+2*64)*12*256*128*3 = 8 182 038 528$ bytes or $7.6$ GB.

For 2.5D segmentation, it would be the triple since three patches (from the previous, the current, and the following slices) per slice are being used. Therefore, the total space would be $(24*128+24*49+20*128+2*64)*12*256*128*3*3 = 24 546 115 584$ bytes or $22.86$ GB.

For the intermediate slice generation, the whole RETOUCH dataset is going to be used. There are 14 Cirrus volumes, 14 Spectralis volumes, and 14 Topcon volumes (10 obtained with the T-2000 and 4 obtained with the T-1000). The T-1000 device resulted in scans with varying dimensions: 512 x 650 (in two volumes) and 512 x 480 (in two volumes). The storage required for these volumes can be calculated as $14*1024*512*128 + 14*496*512*49 + 10*885*512*128 + 2*650*512*128 + 2*480*512*128 = 1 841 840 128$ bytes or $1.7$ GB. Each slice is then saved as an individual image in uint8, which requires the same storage capacity ($1.7$ GB).

# References
[1] R. Tennakoon, A. K. Gostar, R. Hoseinnezhad, and A. Bab-Hadiashar, “Retinal fluid segmentation in OCT images using adversarial loss based convolutional neural networks,” in *2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)*, pp. 1436–1440, 2018.