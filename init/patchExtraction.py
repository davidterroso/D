from os import walk, makedirs
from os.path import isfile, exists
from shutil import rmtree
from PIL import Image
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk, binary_closing
from skimage.filters.rank import entropy
import nutsml.imageutil as ni
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

def extractPatchCenters(roi_mask, patch_shape, npos, pos, neg):
    """
    Extracts the center of the patches from the B-scan and the 
    respective ROI mask

    Args: 
        roi_mask (np.array int8): the region of interst mask
        patch_shape: shape of the resulting patch
        npos: number of positive patches that are going to be
        extracted
        pos: intensity value corresponding to the values that 
        are inside the ROI mask
        neg: intensity value corresponding to the values that 
        are outside the ROI mask

    Return:
        (list[list]): List that contains all the centers of 
        the positive patches that are going to be extracted
    """

    PYINX = 0
    PXINX = 1
    h, w = roi_mask.shape

    # The x coordinates will be separated by 10 along the x axis,
    # between the points from which it is possible to extract patches 
    x = range(int(patch_shape[PXINX] / 2), w - int(patch_shape[PXINX] / 2), 10)
    # The selected x are shuffled
    x_samples = np.random.choice(x, npos * 2, replace=False)
    np.random.shuffle(x_samples)
    c_hold = []
    # Iterates through the possible x index
    for x in x_samples:
        nz = np.nonzero(roi_mask[:, x] == pos)[0]
        # In case there are pixels belonging to the ROI
        if len(nz) > 0:
            # The y coordinate is calculated to be near the medial 
            # position of the mask in that value of x, with a variance of 10
            y = int(float(np.min(nz)) + (
                (float(np.max(nz)) - float(np.min(nz))) / 2.) + np.random.uniform()) + np.random.randint(-10, 10)
            # In case the y value is located below the medial point of the patch and
            # it is impossible to extract patches due to the boundaries of the image, 
            # it is relocated to the minimum possible value where it is still possible
            # to extract patches
            if (y - patch_shape[PYINX] / 2) < 1:
                y = int(patch_shape[PYINX] / 2) + 1
            # In case the y value is located above the medial point of the patch and
            # it is impossible to extract patches due to the boundaries of the image, 
            # it is relocated to the maximum possible value where it is still possible
            # to extract patches
            elif (y + patch_shape[PYINX] / 2) >= h:
                y = h - int(patch_shape[PYINX] / 2) - 1
            # Results are appended to the list of centers
            c_hold.append([y, x])

            # After all the number of patches extracted from the image corresponds to 
            # the determined number, no more patches are extracted
        if len(c_hold) >= npos:
            break

    return c_hold

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

    SHAPE_MULT = {1024: 2., 496: 1., 650: 0.004 / 0.0035, 885: 0.004 / 0.0026}

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
            npshape = (int(patch_shape[0] * SHAPE_MULT[img_height]), patch_shape[1])
            patch_centers = extractPatchCenters(roi_mask=roi, patch_shape=npshape, npos=n_pos, pos=pos, neg=neg)
            it1 = ni.sample_patch_centers(roi, pshape=patch_shape, npos=int(float(n_pos)*.2), nneg=0, pos=pos, neg=neg)
            for r, c, l in it1:
                patch_centers.append([r, c])
            print(patch_centers)






            # Escape of the for loop since the number of images, masks, 
            # and ROI is different due to computational power
            i += 1
            if i == 1:
                return 0


if __name__ == "__main__":
    extractPatches(folder_path="D:\D", patch_shape=(256,128), n_pos=12, n_neg=2, pos=1, neg=0)