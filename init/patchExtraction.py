from os import walk, makedirs
from os.path import isfile, exists
from shutil import rmtree
from PIL import Image
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk, binary_closing
from skimage.filters.rank import entropy
import numpy as np

def createROIMask(slice, mask, threshold, save_location, save_location_to_view):
    """
    Creates and saves the ROI mask

    Args:
        slice (np.array): contains the B-scan from where the ROI is going to 
        be extracted
        mask (np.array): contains the fluid mask of the corresponding B-scan
        threshold (float): threshold of entropy after which the region is kept
        save_location (str): path to where the image is going to be saved in int8
        save_location_to_view (str): path to where the image is going to be saved 
        in uint8 for visualization
    
    Return:
        None
    """
    slice = entropy(slice, disk(15))
    slice = slice / (np.max(slice) + 1e-16)
    slice = np.asarray(slice > threshold, dtype=np.int8)
    slice = np.bitwise_or(slice, mask)
    selem = disk(55)
    slice = binary_closing(slice, footprint=selem)

    h, w = slice.shape
    rnge = list()
    for x in range(0, w):
        col = slice[:, x]
        col = np.nonzero(col)[0]
        if len(col) > 0:
            y_min = np.min(col)
            y_max = np.max(col)
            rnge.append(int((float(y_max) - y_min)/h*100.))
            slice[y_min:y_max, x] = 1

    slice_to_view = (slice * 255).astype(np.uint8)

    slice = Image.fromarray(slice, mode='L')
    slice.save(save_location)
    slice_to_view = Image.fromarray(slice_to_view, mode='L')
    slice_to_view.save(save_location_to_view)

def extractROIMasks(oct_path, folder_path, threshold):
    """
    Responsible for iterating through the folders and extracting the ROI patches

    Args:
        oct_path (str): path to where the RETOUCH dataset is stored
        folder_path (str): path to where the images are being saved
        threshold (float): threshold of entropy after which the region is kept

    Return:
        None
    """
    images_path = folder_path + "\\OCT_images\\segmentation\\slices\\int32\\"
    masks_path = folder_path + "\\OCT_images\\segmentation\\masks\\int8\\"
    save_folder_int8 = folder_path + "\\OCT_images\\segmentation\\roi\\int8\\"
    save_folder_uint8 = folder_path + "\\OCT_images\\segmentation\\roi\\uint8\\"

    # In case the folder to save the images does not exist, it is created
    if not (exists(save_folder_int8) and exists(save_folder_uint8)):
        makedirs(save_folder_int8)
        makedirs(save_folder_uint8)
    else:
        rmtree(save_folder_int8)
        makedirs(save_folder_int8)
        rmtree(save_folder_uint8)
        makedirs(save_folder_uint8)

    for (root, _, files) in walk(oct_path):
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume_name = vendor_volume[1]
                volume_path = images_path + vendor + "_" + volume_name
                volume_masks_path = masks_path + vendor + "_" + volume_name
                slice_num = 0
                slice_path = volume_path + "_" + str(slice_num).zfill(3) + ".tiff"
                mask_path = volume_masks_path + "_" + str(slice_num).zfill(3) + ".tiff"   
                         
                while isfile(slice_path):
                    OCT_slice = imread(slice_path)
                    OCT_slice = img_as_float(OCT_slice.astype(np.float32) / 128. - 1.)
                    OCT_slice_mask = imread(mask_path)
                    OCT_slice_mask = OCT_slice_mask.astype(np.int8)

                    save_name = save_folder_int8 + vendor + "_" + volume_name + "_" + str(slice_num).zfill(3) + ".tiff"
                    save_name_to_view = save_folder_uint8 + vendor + "_" + volume_name + "_" + str(slice_num).zfill(3) + ".tiff"

                    createROIMask(OCT_slice, OCT_slice_mask, threshold, save_location=save_name, save_location_to_view=save_name_to_view)

                    slice_num += 1
                    slice_path = volume_path + "_" + str(slice_num).zfill(3) + ".tiff"
                    mask_path = volume_masks_path + "_" + str(slice_num).zfill(3) + ".tiff"

def extractPatches(folder_path, patch_shape, n_pos, n_neg, pos, neg):
    """
    Extract the patches from the OCT scans

    Args:
        folders_path (str): Path indicating where the images are stored
        patch_shape (tuple): Shape of the patches to extract (height, width)
        n_pos (int): Number of patches from the ROI to extract
        n_neg (int): Number of patches outside the ROI to extract
        pos (int): Intensity indicating a positive region on the ROI mask
        neg (int): Intensity indicating a negative region on the ROI mask

    Return:
        None
    """

    images_path = folder_path + "\\OCT_images\\segmentation\\slices\\int32\\"
    masks_path = folder_path + "\\OCT_images\\segmentation\\masks\\int8\\"
    ROI_path = folder_path + "\\OCT_images\\segmentation\\roi\\int8\\"
    save_patches_path_int32 = folder_path + "\\OCT_images\\segmentation\\patches\\slices\\int32\\"
    save_patches_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\slices\\uint8\\"
    save_patches_masks_path_int8 = folder_path + "\\OCT_images\\segmentation\\patches\\masks\\int8\\"
    save_patches_masks_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\masks\\uint8\\"

    i = 0
    for (root, _, files) in walk(images_path):
        for slice in files:
            slice_path = root + slice
            ROI_mask_path = ROI_path + slice
            mask_path = ROI_path + slice 
            slice = imread(slice_path)
            roi = imread(ROI_mask_path)
            mask = imread(mask_path)

            img_height = slice.shape[0]
            print(roi.min())

            i += 1

            # Escape of the for loop since the number of images, masks, 
            # and ROI is different due to computational power
            if i == 7:
                return 0


if __name__ == "__main__":
    extractPatches(folder_path="D:\D", patch_shape=(256,128), n_pos=12, n_neg=2, pos=1, neg=0)