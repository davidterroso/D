import SimpleITK as sitk
import numpy as np

# Inspired on utils/mhd.py file from Tennakoon et al., 2018 work

def load_oct_image(filename):
    """
    Loads an .mhd OCT volume using Simple ITK library
    Args:
        filename: name of the image to be loaded
    Return: 
        int32 3D image with voxels range 0-255
        the origin of the scan
        the scan spacing
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
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

    # Read the origin of the oct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan_ret, origin, spacing

def load_oct_mask(filename):
    """
    Loads an .mhd OCT fluid mask volume using Simple ITK library
    Args:
        filename: name of the image to be loaded
    Return: 
        int32 3D image with the fluid masks
        the origin of the scan
        the scan spacing
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    oct_scan = sitk.GetArrayFromImage(itkimage)
    oct_scan = oct_scan.astype(np.int8)
    # Read the origin of the oct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan, origin, spacing