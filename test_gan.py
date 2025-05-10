import torch
from IPython import get_ipython
from numpy import array, ndarray
from os import makedirs
from os.path import exists
from pandas import concat, DataFrame, read_csv
from PIL import Image
from shutil import rmtree
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from typing import List
from networks.gan import Generator
from networks.loss import psnr
from network_functions.dataset import TestDatasetGAN
from networks.pix2pix import Pix2PixGenerator
from networks.unet import UNet
from paths import IMAGES_PATH

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def extract_patches(img: torch.Tensor, patch_size: int=64):
    """
    Function used to split a single image tensor with shape HxW 
    into square patches of with a given height/width in order of 
    left to right and top to bottom
    
    Args:
        img (PyTorch tensor): image that will be patched
        patch_size (int): height/width of the patches that will 
            be extracted. Since we are dealing with 64x64 patches
            the value considered default was 64

    Returns:
        None
    """
    # Gets the height and width of the image
    h, w = img.shape[-2], img.shape[-1]
    # Calculates the height that is necessary to pad
    pad_h = (patch_size - (h % patch_size)) % patch_size
    # Padds the image with zeros at the bottom
    padded_img = torch.nn.functional.pad(img, (0, 0, 0, pad_h))

    # Gets the patches from the padded image
    patches = padded_img.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    # Organizes it to have shape C x H x W, where C is the total 
    # number of patches, H and W correspond to the height and width
    # of the extracted patch
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches

def stitch_patches(patches: List[ndarray],
                    image_shape: List, 
                    patch_size: int=64):
    """
    Reconstructs the full image from the 64x64 patches, 
    by stitching them together side by side

    Args:
        patches (List[NumPy array]): list of NumPy 
            matrices that correspond to the predicted
            patches
        image_shape (List[int, int]): shape of the 
            input image and, therefore, expected shape
        patch_size (int): height/width of the patches that will 
            be extracted. Since we are dealing with 64x64 patches
            the value considered default was 64

    Returns:
        
    """
    # Gets the expected shape of images
    original_h, original_w = image_shape
    # Gets the total number of 
    # patches that make up the image
    num_patches = len(patches)
    # Gets the number of patches in each row
    patches_per_row = original_w // patch_size
    # Gets the total number of rows
    rows = (num_patches + patches_per_row - 1) // patches_per_row

    # Gets the expected height 
    # of the image
    h_padded = rows * patch_size
    # Creates a matrix with the same shape as the 
    # image full of zeros
    full_image = torch.zeros((h_padded, original_w))

    # Iterates through all the patches
    for idx, patch in enumerate(patches):
        if isinstance(patch, ndarray):
            patch = torch.from_numpy(patch)
        if patch.ndim == 3:
            patch = patch.squeeze()
        assert patch.shape == (patch_size, patch_size), \
            f"Patch has unexpected shape: {patch.shape}"
        # Gets the row and column in which 
        # the patch must be placed
        i = idx // patches_per_row
        j = idx % patches_per_row
        # Changes the values that are zeroes to 
        # the ones that compose the patch
        full_image[
            i * patch_size:(i + 1) * patch_size,
            j * patch_size:(j + 1) * patch_size
        ] = patch

    # Slices the image to its original size and 
    # removes the bottom part of the image, 
    # which was previously padded
    return full_image[:original_h, :original_w]

def folds_results_gan(first_run_name: str, iteration: int, k: int=5):
    """
    Function used to compare the results obtained in the k folds,
    calculating the mean and standard deviation of the results 
    obtained for each vendor and class, from the files output 
    from the test function

    Args:
        first_run_name (str): name of the first run of the folds 
            considered. It is expected that the name of the first 
            run is something like "Run001" and the runs of the 
            same iteration but different folds increment one to 
            run number thus being named "Run002", "Run003", and
            "Run004", for k=5, for example
        iteration (int): number of the iteration that comprises 
            the k - 1 runs
        k (int): number of folds used in this iteration

    Return: 
        None
    """

    # Initiates a dictionary that will store 
    # the DataFrames from the different runs
    df_dict = {}
    # Gets the number of the first run
    first_run_index = int(first_run_name[3:])
    # Iterates through the runs corresponding to the folds
    # Starts in the index of the first run and stops k - 1 
    # integers after
    for fold in range(first_run_index, k + first_run_index - 1):
        # Gets the name of the run from the fold number
        # e.g. fold=3 -> run_name="Run003"
        run_name = "Run" + str(fold).zfill(3)
        # Indicates the name of the file that stores the SSIM 
        # and PSNR across all slices, grouped by the device 
        # used to obtain the images
        device_file_name = f".\\results\\{run_name}_device.csv"
        # Indicates the name of the file that stores the SSIM 
        # and PSNR across all slices
        results_file_name = f".\\results\\{run_name}_results.csv"            
        # Reads the DataFrame that handles the data per class
        device_df = read_csv(device_file_name)
        # Reads the DataFrame that handles the data per vendor
        results_df = read_csv(results_file_name, index_col="Vendor")

        # Removes the name of the column that has the table's index
        device_df.index.name = None
        results_df.index.name = None

        # Saves, to the corresponding fold in the 
        # dictionary, the two DataFrames as a 
        # single tuple
        df_dict[fold] = (device_df, results_df)

    # Gets the list of devices
    devices = device_df.index.to_list()
    metrics = device_df.columns.to_list()

    # Initiates the two DataFrames, one for 
    # the results per device and the other 
    # for the overall results
    device_df = DataFrame(columns=metrics)
    results_df = DataFrame(columns=metrics)

    # Iterates through the 
    # devices that exist
    for device in devices:
        # Initiates the values 
        # that will be hold for 
        # each device
        values = []
        # Iterates through the 
        # metrics in the DataFrame 
        for metric in metrics:
            # Creates an empty list with the 
            # results in this specific metric
            results = []
            # Iterates through the results in 
            # multiple folds
            for fold, tuple_df in df_dict.items():
                # Appends the values at these coordinates to the list of results
                results.append(float(tuple_df[1].at[device, metric].split(" ")[0]))

            # Calculates the mean of the 
            # results
            mean = array(results).mean()
            # Calculates the standard 
            # deviation of the results
            std = array(results).std()
            # Organizes the value as a single 
            # string
            value = f"{mean:.2f} ({std:.2f})"
            # Appends the value 
            # to the list of values
            values.append(value)
        # For the same device, appends the list 
        # of results to the DataFrame in the last 
        # row
        device_df.loc[len(device_df)] = values
    # Sets the axis of the DataFrame as the 
    # list of possible devices
    device_df = device_df.set_axis(devices)

    # Saves the device's results in a CSV 
    device_df.to_csv(f".\\results\\Iteration{iteration}_devices_results_gan.csv")

    # Iterates through all the 
    # metrics in the DataFrame
    for metric in metrics:
        # Initiates an empty 
        # a list of results
        # for this metric
        results = []
        # Iterates through the multiple folds
        for fold, tuple_df in df_dict.items():
            # Appends the values of the different folds to a list of results
            results.append(float(tuple_df[0].at[0, metric].split(" ")[0]))

        # Calculates the mean and 
        # standard deviation of the results
        mean = array(results).mean()
        std = array(results).std()
        # Converts the values to a single 
        # string
        value = f"{mean:.2f} ({std:.2f})"

        # Appends the values 
        # to a list
        values.append(value)

        # Inserts the values as the last row of 
        # the results DataFrame
        results_df.loc[len(results_df)] = values
    # Sets the name of the axis as the list 
    # of devices
    results_df = results_df.set_axis(devices)

    # Saves the DataFrame as a CSV file
    results_df.to_csv(f".\\results\\Iteration{iteration}_results_gan.csv")

def test_gan(
        fold_test: int,
        model_name: str, 
        weights_name: str, 
        batch_size: int=1, 
        device_name: str="GPU", 
        final_test: bool=False, 
        gen_model_name: str=None,
        number_of_classes: int=1,
        save_images: bool=True,
        split: str="generation_4_fold_split.csv",
        trained_generator_name: str=None
    ):
    """
    Function used to test all the trained generative models

    Args:
        fold_test (int): number of the fold that will be used 
            in the network testing 
        model_name (str): name of the model desired to test.
            Can only be 'GAN', 'UNet', or 'Pix2Pix'
        weights_name (str): path to the model's weight file
        batch_size (int): size of the batch used in testing
        device_name (str): indicates whether the network will 
            be trained using the CPU or the GPU
        final_test (bool): indicates that the final test will 
            be performed, changing the name of the saved files. 
            Since final testing is rare, the default value is 
            False
        gen_model_name (str): name of the trained generator 
            model used in the Pix2Pix. Can only be 'GAN' or
            'UNet'
        number_of_classes (int): number of classes the 
            output will present
        save_images (bool): flag that indicates whether the 
            predicted images will be saved or not
        split (str): name of the k-fold split file that will be 
            used in this run
        trained_generator_name (str): name of the trained 
            generator weights file
            
    Return:
        None
    """
    # Gets the list of volumes used to test the model
    df = read_csv(f"splits\{split}")
    test_volumes = df[str(fold_test)].dropna().to_list()

    # Checks if the given model name is available
    assert model_name in ["GAN", "UNet", "Pix2Pix"],\
        "Possible model names: 'GAN', 'UNet', or 'Pix2Pix'"

    # Declares the Generator model that will be used to infer
    if model_name == "GAN":
        generator = Generator(number_of_classes)
    elif model_name == "UNet":
        generator = UNet(in_channels=2, num_classes=number_of_classes)

    # Declares the path to the weights
    weights_path = "models\\" + weights_name

    # Checks if the declared device exists
    if device_name not in ["CPU", "GPU"]:
        print("Unrecognized device. Possible devices:")
        print("CPU")
        print("GPU")
    elif (device_name == "GPU"):
        # Checks whether the GPU is available 
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            print("GPU is not available. CPU was selected.")
    elif (device_name=="CPU"):
        device_name = "cpu"
    # Saves the variable device as torch.device 
    device = torch.device(device_name)

    # Assigns the generator to the selected device
    generator = generator.to(device=device)
    # Loads the model's trained weights
    if trained_generator_name == None:
        generator.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    else:
        # Declares the path to the weights of the trained generator
        gen_weights_path = "models\\" + trained_generator_name
        # Loads the weights to the trained generator
        generator.load_state_dict(torch.load(gen_weights_path, weights_only=True, map_location=device))
        # Initiates the Pix2Pix generator
        pix2pix_generator = Pix2PixGenerator(number_of_classes, number_of_classes)
        # Loads the weights to the trained Pix2Pix generator
        pix2pix_generator.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        # Sets the Pix2Pix generator 
        # to evaluation mode
        pix2pix_generator.eval()

    # Sets the generator 
    # to evaluation mode
    generator.eval()

    # Creates the TestDatasetGAN object with the 
    # list of volumes that will be used in training
    test_dataset = TestDatasetGAN(test_volumes=test_volumes)
    # Creates the DataLoader object using the dataset, declaring both the size of the 
    # batch and the number of workers
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12)

    # Extracts the name of the run from 
    # the name of the weights file
    run_name = weights_name.split("_")[0]

    # Initiates the list that will 
    # store the results of the slices 
    slice_results = []

    # Declares the name of the folder in which the images will be saved
    if save_images:
        if final_test: 
            folder_to_save = IMAGES_PATH + f"\\OCT_images\\generation\\predictions\\{run_name}_final\\"
        # In case the folder to save exists, it is deleted and created again
        else:
            folder_to_save = IMAGES_PATH + f"\\OCT_images\\generation\\predictions\\{run_name}\\"
        if exists(folder_to_save):
            rmtree(folder_to_save)
            makedirs(folder_to_save)
        else:
            makedirs(folder_to_save)

    # Informs that no backward propagation will be calculated 
    # because it is an inference, thus reducing memory consumption
    with torch.no_grad():
        # Creates a progress bar to track the progress on testing images
        with tqdm(test_dataloader, total=len(test_dataloader), desc='Testing Model', unit='img', leave=True, position=0) as progress_bar:
            # Iterates through every batch and path 
            # (that compose the batch) in the dataloader
            # In this case, the batches are of size one, 
            # so every batch is handled like a single image
            for batch in test_dataloader:
                # Gets the stack of three images from the DataLoader
                stack, mid_image_name = batch["stack"], batch["image_name"][0]

                # Separates the stack in the previous image, 
                # the middle image (the one we aim to predict), 
                # and following image, allocating them to the GPU 
                prev_imgs = stack[:,0,:,:].to(device=device)
                mid_imgs = stack[:,1,:,:].to(device=device)
                next_imgs = stack[:,2,:,:].to(device=device)

                # Using the trained generator, the previous and 
                # following images are used to generate the middle image
                if gen_model_name == "GAN":
                    gen_imgs_wgenerator = generator(prev_imgs.detach(), next_imgs.detach())
                elif gen_model_name == "UNet":
                    gen_imgs_wgenerator = generator(torch.stack([prev_imgs, next_imgs], dim=1).detach())
                elif gen_model_name is not None:
                    print("Unrecognized generator model name. Available names: 'GAN' or 'UNet'")
                    return

                # Generates the middle image
                # In the case of the GAN, since we are dealing 
                # with 64x64 patches, the original image is 
                # patched in multiple others of this shape
                # so that the model, which was also trained in 
                # patches can generate the intermediate patch
                # Then, these patches are stitched together
                if model_name == "GAN":
                    # Initiates a list that will contain the 
                    # outputted patches resulting from the 
                    # 64x64 patches extracted from the loaded 
                    # image 
                    patches_out = []

                    # Calls a function that returns all the 
                    # 1x64x64 patches that make up the image
                    prev_patches = extract_patches(prev_imgs[0])
                    next_patches = extract_patches(next_imgs[0])

                    # Iterates through all the extracted patches
                    for patch in range(prev_patches.shape[0]):
                        # Converts each image to a BxCxHxW format and assigns it to the GPU 
                        prev_patch = prev_patches[patch].unsqueeze(0).unsqueeze(0).to(device)
                        next_patch = next_patches[patch].unsqueeze(0).unsqueeze(0).to(device)

                        # With the previous and following patch 
                        # generates the intermediate one
                        gen_patch = generator(prev_patch, next_patch)
                        # Appends the patch to the list of patches as 
                        # a NumPy array
                        patches_out.append(gen_patch.squeeze(0).cpu())

                    # Stitches all the resulting patches together forming an 
                    # image with shape 496x512 that can be compared
                    gen_imgs = stitch_patches(patches_out, mid_imgs[0].shape)

                elif model_name == "UNet":
                    gen_imgs = generator(torch.stack([prev_imgs, next_imgs], dim=1).detach().to(device=device))
                elif model_name == "Pix2Pix":
                    gen_imgs = pix2pix_generator(gen_imgs_wgenerator).to(device=device)

                # Calculates the PSNR value
                img_psnr = round(psnr(mid_imgs.to(device=device), gen_imgs.to(device=device)), 3)
                # Removes batch dimension from the generated image
                mid_imgs = mid_imgs.squeeze(0).squeeze(0)
                gen_imgs = gen_imgs.squeeze(0).squeeze(0)
                # Calculates the SSIM value
                img_ssim = round(ssim(
                                mid_imgs.cpu().numpy(), gen_imgs.cpu().numpy(),
                                data_range=1,
                                gaussian_weights=True,
                                use_sample_covariance=False,
                                win_size=11
                            ), 3)

                # Appends the images to the list of results
                slice_results.append([mid_image_name, img_psnr, img_ssim])

                # Saves the images
                if save_images:
                    # The image is passed from the GPU to the CPU in case it 
                    # has previously been assigned to it, then it is converted from 
                    # a PyTorch tensor to a NumPy array and saved using Pillow
                    Image.fromarray(array(gen_imgs.cpu().numpy())).save(
                        str(folder_to_save + mid_image_name[:-5] + "_generated" + ".tiff"))

                # Updates the progress bar
                progress_bar.update(1)

    # Creates the folder results in case 
    # it does not exist yet
    makedirs("results", exist_ok=True)

    # Converts the information into a DataFrame
    slice_df = DataFrame(slice_results, columns=["Slice", "PSNR", "SSIM"])

    # Saves the DataFrame as a CSV file
    if not final_test:
        slice_df.to_csv(f"results/{run_name}_slice.csv", index=False)        
    else:
        slice_df.to_csv(f"results/{run_name}_slice_final.csv", index=False)

    # Adds three columns to the DataFrame separating the information of the slice into vendor, volume, and slice_number
    slice_df[['vendor', 'volume', 'slice_number']] = slice_df['Slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)

    # Reads the CSV files with the information of the 
    # OCT volumes used in training in the original dataset
    train_df = read_csv("splits\\volumes_info.csv")
    train_df["volume_key"] = "TRAIN" + train_df["VolumeNumber"].astype(str).str.zfill(3)

    # Reads the CSV files with the information of the 
    # OCT volumes used in testing in the original dataset
    test_df = read_csv("splits\\volumes_info_test.csv")
    test_df["volume_key"] = "TEST" + test_df["VolumeNumber"].astype(str).str.zfill(3)

    # Concatenates both DataFrames into a single one, keeping only the volume_key and the Device 
    # information
    volume_device_map = concat([train_df, test_df], ignore_index=True)[["volume_key", "Device"]]

    # Merges the dataFrame with 
    # the information with the 
    # one of the slices metrics
    # so that it is possible to
    # group by device 
    slice_df = slice_df.merge(
        volume_device_map,
        how="left",
        left_on="volume",
        right_on="volume_key"
    )

    # Removes the key used to merge
    slice_df = slice_df.drop(columns=["volume_key"])

    # Groups the PSNR and SSIM by the mean of all slices, for each Device
    device_df_mean = slice_df[["Device", "PSNR", "SSIM"]].groupby("Device").mean()
    device_df_mean.index.name = "Device"
    device_df_std = slice_df[["Device", "PSNR", "SSIM"]].groupby("Device").std()
    device_df_std.index.name = "Device"
    resulting_device_df = device_df_mean.astype(str) + " (" + device_df_std.astype(str) + ")"

    # Groups the PSNR and SSIM by the mean of all slices
    slice_df_mean = slice_df[["PSNR", "SSIM"]].mean().to_frame().T
    slice_df_mean.index.name = "AllDevices"
    slice_df_std = slice_df[["PSNR", "SSIM"]].std().to_frame().T
    slice_df_std.index.name = "AllDevices"
    resulting_slice_df = slice_df_mean.astype(str) + " (" + slice_df_std.astype(str) + ")"

    # Saves the DataFrames to a CSV file
    if not final_test:
        resulting_device_df.to_csv(f"results/{run_name}_device.csv")
        resulting_slice_df.to_csv(f"results/{run_name}_results.csv")
    else:
        resulting_device_df.to_csv(f"results/{run_name}_device_final.csv")
        resulting_slice_df.to_csv(f"results/{run_name}_results_final.csv")
