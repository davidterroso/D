import SimpleITK as sitk
import numpy as np
from os import walk

# Inspired on utils/mhd.py file from Tennakoon et al., 2018 work

def load_oct_image(filename):
    """
    Loads an .mhd OCT volume using Simple ITK library
    Args:
        filename: name of the image to be loaded
    Return: 
        (Numpy int32 array) int32 3D image with voxels range 0-255
        (Numpy array) the origin of the scan
        (Numpy array) the scan spacing
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle 
    # the dimensions to get axis in the order z,y,x
    oct_scan = sitk.GetArrayFromImage(itkimage)
    oct_scan = oct_scan.astype(np.int32)
    oct_scan_ret = np.zeros(oct_scan.shape, dtype=np.int32)

    if 'Cirrus' in filename:
        # range 0-255
        oct_scan_ret = oct_scan.astype(np.int32)
    elif 'Spectralis' in filename:
        # range 0-2**16
        oct_scan_ret = (oct_scan.astype(np.float32) / (2 ** 16) * 255.).astype(np.int32)
    elif 'Topcon' in filename:
        # range 0-255
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
        (Numpy int8 array) int32 3D image with the fluid masks
        (Numpy array) the origin of the scan
        (Numpy array) the scan spacing
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

def save_all_oct_as_tiff(oct_folder, save_folder):
    """
    Reads all the OCT volumes and saves them as a tiff image in 
    int32 (that will be used) and int8 (for visualization)
    
    Args:
        oct_folder (str): path to the folder where the OCT scans
        are located
        save_folder (str): path to where the images are going to
        be stored

    Return:
        None
    """

    for (root, _, files) in walk(oct_folder):
        i=0
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume_name = vendor_volume[1]
                save_name = vendor + "_" + volume_name
                for filename in files:
                    if filename == "oct.mhd":
                        file_path = root + """\\""" + filename
                        img, _, _ = load_oct_image(file_path)
                        num_slices = img.shape[0]
                        print(num_slices)
                        print(img.shape)
                         
                        # i += 1
        
        if i == 1:
            break

if __name__ == "__main__":
    save_all_oct_as_tiff(oct_folder="D:\RETOUCH", save_folder="")
    print("EOF.")
