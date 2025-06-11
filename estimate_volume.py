import numpy as np
import os
import pandas as pd
from IPython import get_ipython
from os import fsdecode, fsencode, listdir, makedirs
from skimage.io import imread
from init.read_oct import load_oct_image
from paths import RETOUCH_PATH

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def estimate_volume(folder_path: str):
    """
    In this function, the path to the folder that contains 
    the predicted masks by the model is indicated. The 
    images are read and, by loading the metadata of the 
    images, the volume of each fluid in each OCT volume is
    estimated. The predicted images might have been resized
    for inference and, therefore, the metadata must also be 
    handled accordingly.

    Args:
        folder_path (str): path to the folder with the 
            predicted masks
    
    Returns:
        None
    """
    # Creates the DataFrame that will store the information 
    # of all slices
    slices_df = pd.DataFrame(columns=["Set", "Vendor", "VolumeNumber", "SliceNumber","IRF", "SRF", "PED"])
    # Encodes the path of the folder 
    # in bytes
    directory = fsencode(folder_path)

    # Creates a progress bar for the fluid calculations
    with tqdm(listdir(directory), total=len(listdir(directory)), desc='Calculating Fluid', unit='img', leave=True, position=0) as progress_bar:
        # Iterates through the images in the folder
        for file in listdir(directory):
            # Reads the number of the slice
            slice_num = fsdecode(file).split("_")[-2]
            # Reads the predicted mask
            img = imread(str(folder_path + fsdecode(file)))
            # Gets the vendor and the volume
            vendor_vol = fsdecode(file).split("_")[:2]
            # Indicates the path to the oct.mhd file that contains the volume metadata
            # As of now, this is only to test with a few images, so every set will be of training
            belonging_set = "TrainingSet"
            vol_path = RETOUCH_PATH + f"\\RETOUCH-{belonging_set}-{vendor_vol[0]}\\{vendor_vol[1]}\\oct.mhd"
            # Reads the OCT volume that contains the slice 
            # that is being iterated and gets the origin 
            # coordinates, the spacing, and the volume as a
            # NumPy array
            vol, origin, spacing = load_oct_image(vol_path)
            
            # The spacing is re-ordered into Width x Height x Depth
            spacing = np.array((spacing[2], spacing[1], spacing[0]))
            # Counts the total number of voxels in the array 
            # for each class
            fluids, counts = np.unique(img, return_counts=True)
            # Removes the counts of background voxels
            fluids = fluids[1:]
            counts = counts[1:]
            
            # In case the image has been reshaped, seen by 
            # the image not being of the height of the 
            # smallest images (496), the spacing is adjusted 
            # to match the real dimensions of the image 
            # having in mind the reshaping factor
            if vol.shape[1] != 496:
                spacing[0] = vol.shape[1] / 496 * spacing[0]
            
            # The real area of each voxel is calculated by 
            # the height of the voxel times its width
            voxel_area = spacing[0] * spacing[1]

            # The volume of the voxel is calculated by multiplying 
            # the area of each voxel by the distance between slices
            # in the non-edge slices and half the distance between 
            # slices in the edge slices (first and last slice)
            if int(slice_num) == 0 or int(slice_num) == vol.shape[0]:
                voxel_volume = voxel_area * 0.5 * spacing[2]
            else:
                voxel_volume = voxel_area * spacing[2]
            # The volume of each fluid in the slice is calculated 
            # by the multiplication of the voxel volume by the 
            # number of voxels present in the fluid mask
            fluid_volumes = (counts * voxel_volume)
            fluid_volumes_results = np.zeros(3)
            for i, label in enumerate(fluids):
                if 1 <= label <= 3:
                    fluid_volumes_results[label - 1] = fluid_volumes[i]
            # Appends the results of the slice to the DataFrame 
            slices_df.loc[len(slices_df)] = [belonging_set, vendor_vol[0], vendor_vol[1], slice_num] + fluid_volumes_results.tolist()

            # Updates the progress bar
            progress_bar.update(1)

    # Groups the the DataFrame according to the OCT volume and the vendor
    volumes_df = slices_df.drop("SliceNumber", axis=1).groupby(["Set","VolumeNumber"], as_index=False).agg({
        "Vendor": "first",
        "IRF": "sum",
        "SRF": "sum",
        "PED": "sum"
    })
    vendors_df = slices_df.drop(["SliceNumber", "Set", "VolumeNumber"], axis=1).groupby("Vendor", as_index=False).sum()
    # Creates the folder in which the files 
    # will be saved
    makedirs("fluid_volumes", exist_ok=True)
    # Get the last folder name for output file naming
    folder_name = os.path.basename(os.path.normpath(folder_path))
    # Saves the DataFrames in CSV files
    volumes_df.to_csv(f".\\fluid_volumes\\{folder_name}_volumes.csv", index=False)
    vendors_df.to_csv(f".\\fluid_volumes\\{folder_name}_vendors.csv", index=False)

estimate_volume(folder_path=r"D:\DavidTerroso\Images\OCT_images\segmentation\predictions\Run076111114_resized_final_retouch_probability_masks\\")
