import torch
from IPython import get_ipython
from numpy import array
from os import makedirs
from os.path import exists
from pandas import concat, DataFrame, read_csv
from PIL import Image
from shutil import rmtree
from torch.utils.data import DataLoader
from networks.gan import Generator
from network_functions.dataset import TestDatasetGAN
from paths import IMAGES_PATH

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def test_gan(
        fold_test: int, 
        weights_name: str, 
        batch_size: int=1, 
        device_name: str="GPU", 
        final_test: bool=False, 
        number_of_classes: int=1,
        save_images: bool=True
    ):
    """
    Function used to test the trained GAN models

    Args:
        fold_test (int): number of the fold that will be used 
            in the network testing 
        weights_name (str): path to the model's weight file
        batch_size (int): size of the batch used in testing
        device_name (str): indicates whether the network will 
            be trained using the CPU or the GPU
        final_test (bool): indicates that the final test will be 
            performed, changing the name of the saved files. Since final
            testing is rare, the default value is False 
        number_of_classes (int): number of classes the 
            output will present
        save_images (bool): flag that indicates whether the 
            predicted images will be saved or not
            
    Return:
        None
    """
    # Gets the list of volumes used to test the model
    df = read_csv("splits/generation_5_fold_selection.csv")
    test_volumes = df[str(fold_test)].dropna().to_list()

    # Declares the Generator model that will be used to infer
    generator = Generator(number_of_classes)

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
    generator.load_state_dict(torch.load(weights_name, weights_only=True, map_location=device))
    # Sets the generator 
    # to evaluation mode
    generator.eval()

    # Creates the TestDatasetGAN object with the 
    # list of volumes that will be used in training
    test_dataset = TestDatasetGAN(test_volumes)
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
                stack, mid_image_name = batch["stack"], batch["image_name"]

                # Separates the stack in the previous image, 
                # the middle image (the one we aim to predict), 
                # and following image, allocating them to the GPU 
                prev_imgs = stack[:,0,:,:].to(device=device)
                mid_imgs = stack[:,1,:,:].to(device=device)
                next_imgs = stack[:,2,:,:].to(device=device)

                # Generates the middle image
                gen_imgs = generator(prev_imgs, next_imgs)

                # Calculates the PNSR value
                img_pnsr = round(pnsr(mid_imgs, gen_imgs), 3)
                # Calculates the SSIM value
                img_ssim = round(ssim(mid_imgs, gen_imgs), 3)

                # Appends the images to the list of results
                slice_results.append([mid_image_name, img_pnsr, img_ssim])

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
    slice_df = DataFrame(slice_results, columns=["Slice", "PNSR", "SSIM"])

    # Saves the DataFrame as a CSV file
    if not final_test:
        slice_df.to_csv(f"results/{run_name}_slice_dice.csv", index=False)        
    else:
        slice_df.to_csv(f"results/{run_name}_slice_dice_final.csv", index=False)

    # Adds three columns to the DataFrame separating the information of the slice into vendor, volume, and slice_number
    slice_df[['vendor', 'volume', 'slice_number']] = slice_df['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)

    # Reads the CSV files with the information of the 
    # OCT volumes used in training in the original dataset
    train_df = read_csv("..\\splits\\volumes_info.csv")
    train_df["volume_key"] = "TRAIN" + train_df["VolumeNumber"].astype(str).str.zfill(3)

    # Reads the CSV files with the information of the 
    # OCT volumes used in testing in the original dataset
    test_df = read_csv("..\\splits\\volumes_info_test.csv")
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

    # Groups the PNSR and SSIM by the mean of all slices, for each Device
    device_df_mean = slice_df[["Device", "PNSR", "SSIM"]].groupby("Device").mean()
    device_df_mean.index.name = "Device"
    device_df_std = slice_df[["Device", "PNSR", "SSIM"]].groupby("Device").std()
    device_df_std.index.name = "Device"
    resulting_device_df = device_df_mean.astype(str) + " (" + device_df_std.astype(str) + ")"

    # Groups the PNSR and SSIM by the mean of all slices
    slice_df_mean = slice_df[["Device", "PNSR", "SSIM"]].mean().to_frame().T
    slice_df_mean.index.name = "Device"
    slice_df_std = slice_df[["Device", "PNSR", "SSIM"]].std().to_frame().T
    slice_df_std.index.name = "Device"
    resulting_slice_df = slice_df_mean.astype(str) + " (" + slice_df_std.astype(str) + ")"

    # Saves the DataFrames to a CSV file
    if not final_test:
        slice_df.to_csv(f"results/{run_name}_slice.csv")
        resulting_device_df.to_csv(f"results/{run_name}_device.csv")
        resulting_slice_df.to_csv(f"results/{run_name}_results.csv")
    else:
        slice_df.to_csv(f"results/{run_name}_slice_final.csv")
        resulting_device_df.to_csv(f"results/{run_name}_device_final.csv")
        resulting_slice_df.to_csv(f"results/{run_name}_results_final.csv")
