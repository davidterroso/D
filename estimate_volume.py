import numpy as np
from os import fsdecode, fsencode, listdir
from skimage.io import imread
from init.read_oct import load_oct_image
from paths import RETOUCH_PATH

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
    # Encodes the path of the folder 
    # in bytes
    directory = fsencode(folder_path)
    # Iterates through the images in the folder
    for file in listdir(directory):
        # Reads the number of the slice
        slice_num = fsdecode(file).split("_")[-1][:-5]
        # Reads the predicted mask
        img = imread(str(folder_path + fsdecode(file)))
        # Gets the vendor and the volume
        vendor_vol = fsdecode(file).split("_")[:2]
        # Indicates the path to the oct.mhd file that contains the volume metadata
        # As of now, this is only to test with a few images, so every set will be of training
        vol_path = RETOUCH_PATH + f"\\RETOUCH-TrainingSet-{vendor_vol[0]}\\{vendor_vol[1]}\\oct.mhd"
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
        
        # Handles the differences in shape of the input slices
        
        return


estimate_volume("D:\D\OCT_images\segmentation\masks\int8\\")