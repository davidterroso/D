import numpy as np
import pandas as pd
from IPython import get_ipython
from os import walk
from read_oct import load_oct_image, load_oct_mask

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

# Dictionary that matches the spacing of the 
# OCT with the device used to obtain the images
spacing_to_device = {
        0.002: "Cirrus",
        0.0039: "Spectralis",
        0.0026: "T-2000",
        0.0035: "T-1000"
    }

# Dictionary of labels in masks to fluid names
label_to_fluids = {
        0: "Background",
        1: "IRF",
        2: "SRF",
        3: "PED"
    }

def volumes_info(oct_folder: str, vol_set: str="train"):
    """
    Reads all the OCT segmentation masks used in the 
    segmentation task and extracts the number of voxels
    per class
    
    Args:
        oct_folder (str): path to the folder where the 
            OCT scans are located
        vol_set (str): set of data to which the OCT 
            volumes belong. Can only be 'train' or 
            'test' and default is 'train'

    Return:
        None
    """

    # Initiates the variable that will 
    # count the progress of volumes 
    # whose slices are being extracted
    volume = 0

    # Checks if the set is popssible and sets them 
    # to the way the data is handled in the folder
    # Also declares the total number of volumes 
    # and the columns that will be present in the CSV
    if vol_set == "train":
        desired_set = "TrainingSet"
        total_vols = 70 
        columns=["VolumeNumber", "Vendor", 
                 "Background", "IRF", "SRF",
                 "PED", "Device", "SlicesNumber"]
    elif vol_set == "test":
        desired_set = "TestSet"
        total_vols = 42 
        columns=["VolumeNumber", "Vendor", 
            "Device", "SlicesNumber"]
    else:
        print("Not a possible 'vol_set' argument. \
              Please select 'train' or 'test'")
        return

    # Creates the dataframe that will later be converted into a CSV file
    df = pd.DataFrame(columns=columns)

    # Iterates through the folders to read the OCT volumes used in segmentation
    # and saves them both in int32 for better manipulation and in uint8 for
    # visualization
    with tqdm(total=total_vols, desc="Counting Voxels", unit="vol", leave=True, position=0) as progress_bar:
        for (root, _, files) in walk(oct_folder):
            train_or_test = root.split("-")
            if ((len(train_or_test) == 3) and (train_or_test[1] == desired_set)):
                vendor_volume = train_or_test[2].split("""\\""")
                if (len(vendor_volume) == 2):
                    vendor = vendor_volume[0]
                    volume_index = int(vendor_volume[1][-3:])
                    # Creates a progress bar
                    # Iterates through to the subfolders and reads the reference.mhd file to 
                    # extract the images
                    for filename in files:
                        if filename == "reference.mhd" and desired_set == "TrainingSet":
                            file_path = root + """\\""" + filename
                            # Loads the OCT volume
                            img, _, spacing = load_oct_mask(file_path)
                            
                            # Gets the number of voxels per class
                            unique_values, voxels_count = np.unique(img, return_counts=True)

                            # Loads the data to the DataFrame
                            tmp_values = [volume_index, vendor]

                            # Checks all the possible classes in the volume 
                            # and appends them to a temporary list
                            for i in [0, 1, 2, 3]:
                                if i in unique_values:
                                    index = np.where(unique_values == i)
                                    voxel_count = voxels_count[index]
                                    tmp_values.append(voxel_count.item())
                                else:
                                    tmp_values.append(0)

                            # Gets the vendor through the spacing of the OCT volume
                            if vendor == "Cirrus":
                                oct_device = spacing_to_device.get(round(spacing[1],3))
                            else:
                                oct_device = spacing_to_device.get(round(spacing[1],4))

                            # Appends the values to the list 
                            # that will be saved on the DataFrame
                            tmp_values.append(oct_device)
                            tmp_values.append(img.shape[0])

                            # Adds a temporary list to the DataFrame
                            df.loc[volume] = tmp_values
                            volume += 1

                            # Updates the progress bar
                            progress_bar.update(1)
                        # For the testing volumes that 
                        # do not have a reference mask
                        elif filename == "oct.mhd" and desired_set == "TestSet":
                            file_path = root + """\\""" + filename
                            # Loads the OCT volume
                            img, _, spacing = load_oct_image(file_path)

                            # Loads the data to the DataFrame
                            tmp_values = [volume_index, vendor]

                            # Gets the vendor through the spacing of the OCT volume
                            if vendor == "Cirrus":
                                oct_device = spacing_to_device.get(round(spacing[1],3))
                            else:
                                oct_device = spacing_to_device.get(round(spacing[1],4))

                            # Appends the values to the list 
                            # that will be saved on the DataFrame
                            tmp_values.append(oct_device)
                            tmp_values.append(img.shape[0])

                            # Adds a temporary list to the DataFrame
                            df.loc[volume] = tmp_values
                            volume += 1

                            # Updates the progress bar
                            progress_bar.update(1)

    # Saves the DataFrame to a CSV file
    if desired_set == "TrainingSet":
        df.to_csv("..\splits\\volumes_info.csv", index=False)
    elif desired_set == "TestSet":
        df.to_csv("..\splits\\volumes_info_test.csv", index=False)

    print("All voxels have been counted.")
    print("EOF.")

def volumes_resumed_info(file_path: str="..\splits\\volumes_info.csv"):
    """
    Reads the CSV file "volumes_info.csv" and calculates the total 
    number of voxels and their mean per vendor
    
    Args:
        file_path (str): path to the "volumes_info.csv" CSV file

    Return:
        None
    """
    # Reads as a DataFrame file the 
    # split desired to analyse
    df = pd.read_csv(file_path)
    # Calculates the mean and standard deviation of each fluid for each vendor
    resulting_df_mean = df.groupby("Vendor").mean().drop("VolumeNumber", axis=1).round(2)
    resulting_df_std = df.groupby("Vendor").std().drop("VolumeNumber", axis=1).round(2)
    # Combines both tables in a single table that is organized by having the mean followed 
    # by the standard deviation in brackets (e.g. mean (std))
    resulting_df = resulting_df_mean.astype(str) + " (" + resulting_df_std.astype(str) + ")"
    # Saves the DataFrame as a CSV file
    resulting_df.to_csv("..\splits\\volumes_mean_std.csv", index=True)

    # Calculates the sum of each fluid for each vendor
    resulting_df_sum = df.groupby("Vendor").sum().drop("VolumeNumber", axis=1).round(2)
    # Saves the DataFrame as a CSV file
    resulting_df_sum.to_csv("..\splits\\volumes_sum.csv", index=True)

volumes_info("D:\RETOUCH")