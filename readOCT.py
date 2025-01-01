import SimpleITK as sitk
import numpy as np

# Inspired on utils/mhd.py file from Tennakoon et al., 2018 work

def load_oct_image(filename):
    """
    loads an .mhd file using simple_itk
    :param filename: name of the image to be loaded
    :return: int32 3D image with voxels range 0-255
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.astype(np.int32)
    ct_scan_ret = np.zeros(ct_scan.shape, dtype=np.int32)

    if 'Cirrus' in filename:
        # range 0-255
        ct_scan_ret = ct_scan.astype(np.int32)
    elif 'Spectralis' in filename:
        # range 0-2**16
        ct_scan_ret = (ct_scan.astype(np.float32) / (2 ** 16) * 255.).astype(np.int32)
    elif 'Topcon' in filename:
        # range 0-255
        ct_scan_ret = ct_scan.astype(np.int32)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan_ret, origin, spacing

def load_oct_mask(filename):
    """
    loads an .mhd file using simple_itk
    :param filename: 
    :return: 
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.astype(np.int8)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing