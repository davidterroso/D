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
