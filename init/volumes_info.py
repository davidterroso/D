import numpy as np
from IPython import get_ipython
from os import walk
from pandas import DataFrame
from read_oct import load_oct_mask

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

# Dictionary of labels in masks to fluid names
label_to_fluids = {
        0: "Background",
        1: "IRF",
        2: "SRF",
        3: "PED"
    }

def volumes_info(oct_folder: str):
    """
    Reads all the OCT segmentation masks used in the segmentation 
    task and extracts the number of voxels per class
    
    Args:
        oct_folder (str): path to the folder where the OCT scans
            are located

    Return:
        None
    """

    # Initiates the variable that will 
    # count the progress of volumes 
    # whose slices are being extracted
    volume = 0

    # Indicates the name of the columns in the DataFrame
    columns=["VolumeNumber", "Vendor", "IRF", "SRF", "PED"]
    # Creates the dataframe that will later be converted into a CSV file
    df = DataFrame(columns=columns)

    values = []

    # Iterates through the folders to read the OCT volumes used in segmentation
    # and saves them both in int32 for better manipulation and in uint8 for
    # visualization
    with tqdm(total=70, desc="Counting Voxels", unit="vol", leave=True, position=0) as progress_bar:
        for (root, _, files) in walk(oct_folder):
            train_or_test = root.split("-")
            if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
                vendor_volume = train_or_test[2].split("""\\""")
                if len(vendor_volume) == 2:
                    vendor = vendor_volume[0]
                    volume_index = int(vendor_volume[1][-3:])
                    # Creates a progress bar
                    # Iterates through to the subfolders and reads the reference.mhd file to 
                    # extract the images
                    for filename in files:
                        if filename == "reference.mhd":
                            file_path = root + """\\""" + filename
                            # Loads the OCT volume
                            img, _, _ = load_oct_mask(file_path)
                            
                            # Gets the number of voxels per class
                            unique_values, voxels_count = np.unique(img, return_counts=True)

                            # Loads the data to the DataFrame
                            tmp_values = [volume_index, vendor]

                            # Checks all the possible classes in the volume 
                            # and appends them to a temporary list
                            for i in [1, 2, 3]:
                                if i in unique_values:
                                    index = np.where(unique_values == i)
                                    voxel_count = voxels_count[index]
                                    tmp_values.append(voxel_count.item())
                                else:
                                    tmp_values.append(0)

                            # Adds a temporary list to the DataFrame
                            df.loc[volume] = tmp_values
                            volume += 1

                            # Updates the progress bar
                            progress_bar.update(1)

    # Saves the DataFrame to a CSV file
    df.to_csv("..\splits\\volumes_info.csv", index=False)

    print("All voxels have been counted.")
    print("EOF.")

if __name__ == "__main__":
    volumes_info(oct_folder="D:\RETOUCH")
