import SimpleITK as sitk
import numpy as np
from os import walk, makedirs
from os.path import exists
from shutil import rmtree
from PIL import Image

def int32_to_uint8(image):
    """
    Receives an int32 NumPy array that represents an image and transforms it into 
    uint8 so that it can be visualized by the PC image viewer
    Args:
        image (NumPy int32 array): slice of an OCT scan
    Return:
        (NumPy uint8 array): slice of an OCT scan
    """
    return (255 * (image - image.min())/(image.max() - image.min())).astype(np.uint8)

# Inspired by utils/mhd.py file from Tennakoon et al., 2018 work

def load_oct_image(filename):
    """
    Loads an .mhd OCT volume using Simple ITK library
    Args:
        filename: name of the image to be loaded
    Return: 
        (NumPy int32 array) int32 3D image with voxels range 0-255
        (NumPy array) the origin of the scan
        (NumPy array) the scan spacing
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle 
    # the dimensions to get axis in the order z,y,x
    oct_scan = sitk.GetArrayFromImage(itkimage)
    oct_scan = oct_scan.astype(np.int32)
    oct_scan_ret = np.zeros(oct_scan.shape, dtype=np.int32)

    if 'Cirrus' in filename:
        # Range 0-255
        oct_scan_ret = oct_scan.astype(np.int32)
    elif 'Spectralis' in filename:
        # Range 0-2**16
        oct_scan_ret = (oct_scan.astype(np.float32) / (2 ** 16) * 255.).astype(np.int32)
    elif 'Topcon' in filename:
        # Range 0-255
        oct_scan_ret = oct_scan.astype(np.int32)

    # Read the origin of the oct_scan, will be used to convert the 
    # coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan_ret, origin, spacing

def load_oct_mask(filename):
    """
    Loads an .mhd OCT fluid mask volume using Simple ITK library
    Args:
        filename (str): name of the image to be loaded
    Return: 
        (NumPy int8 array) int32 3D image with the fluid masks
        (NumPy array) the origin of the scan
        (NumPy array) the scan spacing
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the 
    # dimensions to get axis in the order z,y,x
    oct_scan = sitk.GetArrayFromImage(itkimage)
    oct_scan = oct_scan.astype(np.int8)
    # Read the origin of the oct_scan, will be used to convert the 
    # coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan, origin, spacing

def save_segmentation_oct_as_tiff(oct_folder, save_folder):
    """
    Reads all the OCT volumes used in the segmentation task
    and saves them as a tiff image in int32 (that will be used) 
    and uint8 (for visualization)
    
    Args:
        oct_folder (str): path to the folder where the OCT scans
        are located
        save_folder (str): path to where the images are going to
        be stored

    Return:
        None
    """
    # In case the folder to save the images does not exist, it is created
    save_name_uint8 = save_folder + "\\OCT_images\\segmentation\\slices\\uint8\\"
    save_name_int32 = save_folder + "\\OCT_images\\segmentation\\slices\\int32\\"
    if not (exists(save_name_int32) and exists(save_name_uint8)):
        makedirs(save_name_int32)
        makedirs(save_name_uint8)
    else:
        rmtree(save_name_int32)
        makedirs(save_name_int32)
        rmtree(save_name_uint8)
        makedirs(save_name_uint8)

    # Iterates through the folders to read the OCT volumes used in segmentation
    # and saves them both in int32 for better manipulation and in uint8 for
    # visualization
    for (root, _, files) in walk(oct_folder):
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume_name = vendor_volume[1]
                save_name_uint8_tmp = save_name_uint8 + vendor + "_" + volume_name
                save_name_int32_tmp = save_name_int32 + vendor + "_" + volume_name
                # Iterates through to the subfolders and reads the oct.mhd file to 
                # extract the images
                for filename in files:
                    if filename == "oct.mhd":
                        file_path = root + """\\""" + filename
                        img, _, _ = load_oct_image(file_path)
                        num_slices = img.shape[0]
                        # Iterates through the slices to save each slice with 
                        # an identifiable name, both in uint8 for visualization
                        # and int32 for better future manipulation 
                        for slice_num in range(num_slices):
                            im_slice = img[slice_num,:,:]
                            # Normalizes the image to uint8 so that it can be 
                            # visualized in the computer
                            im_slice_uint8 = int32_to_uint8(im_slice)

                            # Saves image in int32
                            image = Image.fromarray(im_slice)
                            save_name_slice = save_name_int32_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)

                            # Saves image in uint8
                            image = Image.fromarray(im_slice_uint8)
                            save_name_slice = save_name_uint8_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)

def save_segmentation_mask_as_tiff(oct_folder, save_folder):
    """
    Reads all the OCT segmentation masks used in the segmentation 
    task and saves them as a tiff image in int8 (that will be 
    used) and uint8 (for visualization)
    
    Args:
        oct_folder (str): path to the folder where the OCT scans
        are located
        save_folder (str): path to where the images are going to
        be stored

    Return:
        None
    """

    # In case the folder to save the images does not exist, it is created
    save_name_uint8 = save_folder + "\\OCT_images\\segmentation\\masks\\uint8\\"
    save_name_int8 = save_folder + "\\OCT_images\\segmentation\\masks\\int8\\"
    if not (exists(save_name_int8) and exists(save_name_uint8)):
        makedirs(save_name_int8)
        makedirs(save_name_uint8)
    else:
        rmtree(save_name_int8)
        makedirs(save_name_int8)
        rmtree(save_name_uint8)
        makedirs(save_name_uint8)

    # Iterates through the folders to read the OCT volumes used in segmentation
    # and saves them both in int32 for better manipulation and in uint8 for
    # visualization
    for (root, _, files) in walk(oct_folder):
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume_name = vendor_volume[1]
                save_name_uint8_tmp = save_name_uint8 + vendor + "_" + volume_name
                save_name_int8_tmp = save_name_int8 + vendor + "_" + volume_name

                # Iterates through to the subfolders and reads the reference.mhd file to 
                # extract the images
                for filename in files:
                    if filename == "reference.mhd":
                        file_path = root + """\\""" + filename
                        img, _, _ = load_oct_mask(file_path)
                        num_slices = img.shape[0]

                        # Iterates through the slices to save each slice with an identifiable name,
                        # both in uint8 for visualization and int32 for better future manipulation 
                        for slice_num in range(num_slices):
                            im_slice = img[slice_num,:,:]

                            # Normalize the masks so that they can be visualized
                            im_slice_uint8 = (np.round(255 * (im_slice / 3))).astype(np.uint8)

                            # Saves image in int32
                            image = Image.fromarray(im_slice)
                            save_name_slice = save_name_int8_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)

                            # Saves image in uint8
                            image = Image.fromarray(im_slice_uint8)
                            save_name_slice = save_name_uint8_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)

def save_generation_oct_as_tiff(oct_folder, save_folder):
    """
    Reads all the OCT volumes used in the generation task
    and saves them as a tiff image in int32 (that will be used) 
    and uint8 (for visualization)
    
    Args:
        oct_folder (str): path to the folder where the OCT scans
        are located
        save_folder (str): path to where the images are going to
        be stored

    Return:
        None
    """
    # In case the folder to save the images does not exist, it is created
    save_name_int8 = save_folder + "\\OCT_images\\generation\\uint8\\"
    save_name_int32 = save_folder + "\\OCT_images\\generation\\int32\\"
    if not (exists(save_name_int32) and exists(save_name_int8)):
        makedirs(save_name_int32)
        makedirs(save_name_int8)
    else:
        rmtree(save_name_int32)
        makedirs(save_name_int32)
        rmtree(save_name_int8)
        makedirs(save_name_int8)

        # Iterates through the folders to read the OCT volumes used in segmentation
        # and saves them both in int32 for better manipulation and in uint8 for
        # visualization
    for (root, _, files) in walk(oct_folder):
        train_or_test = root.split("-")
        if (len(train_or_test) == 3):
            vendor_volume = train_or_test[2].split("""\\""")
            if len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume_name = vendor_volume[1]
                save_name_int8_tmp = save_name_int8 + vendor + "_" + volume_name
                save_name_int32_tmp = save_name_int32 + vendor + "_" + volume_name
                # Iterates through to the subfolders and reads the oct.mhd file to 
                # extract the images
                for filename in files:
                    if filename == "oct.mhd":
                        file_path = root + """\\""" + filename
                        img, _, _ = load_oct_image(file_path)
                        num_slices = img.shape[0]
                        # Iterates through the slices to save each slice with an identifiable name, both in uint8 for visualization
                        # and int32 for better future manipulation          
                        for slice_num in range(num_slices):
                            im_slice = img[slice_num,:,:]
                            # Normalizes the image to uint8 so that it can be visualized in the computer
                            im_slice_int8 = int32_to_uint8(im_slice)

                            # Saves image in int32
                            image = Image.fromarray(im_slice)
                            save_name_slice = save_name_int32_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)

                            # Saves image in uint8
                            image = Image.fromarray(im_slice_int8)
                            save_name_slice = save_name_int8_tmp + "_" + str(slice_num).zfill(3) + '.tiff'
                            image.save(save_name_slice)
