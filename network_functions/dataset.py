import torch
from numpy import any, expand_dims, max, min, nonzero, stack, sum, int8, uint8, ndarray
from numpy.random import random_sample
from os import listdir, remove
from os.path import exists
from paths import IMAGES_PATH
from skimage.io import imread
from skimage.transform import resize
from torchvision.transforms.v2 import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, InterpolationMode
from torch.utils.data import Dataset

# Dictionary for the patch multiplication depending on the height of the image
SHAPE_MULT = {1024: 2., 496: 1., 650: 0.004 / 0.0035, 885: 0.004 / 0.0026}

def handle_test_images(scan: ndarray, mask: ndarray, roi: ndarray, patch_shape: tuple):
    """
    Function that handles the images shapes in test images

    Args:
        scan (NumPy array): OCT B-scan to handle in channels 
            last
        mask (NumPy array): fluid ground-truth mask to handle
        roi (NumPy array): ROI mask useful to extract the 
            center location
        patch_shape (tuple(int, int)): shape of the output 
            images

    Return:
        scan (NumPy array): OCT B-scan in shape (256, 512)
        mask (NumPy array): fluid ground-truth mask in shape
            (256, 512) 

    """
    # Gets the height of the image
    img_height = scan.shape[0]

    # Gets the new patch shape
    new_patch_shape = (int(patch_shape[0] * SHAPE_MULT[img_height]), patch_shape[1])

    # Declares the ROI mask as int8
    roimask = roi.astype(int8)
    # Calculates the number of rows that 
    # have non-zero values in the ROI mask
    nx_y = nonzero(sum(roimask, axis=1))[0]

    # In case there are rows with ROI
    if len(nx_y) > 0:
        # Locates the lowest and highest row with ROI
        miny, maxy = float(min(nx_y)), float(max(nx_y))
        # Declares the row between these two rows as 
        # the vertical center of the new patch 
        y = int(miny + (maxy-miny)/2.)
        # In case this new coordinate is above 
        # the middle of the patch, the patch 
        # vertical center is set to the middle 
        # of the image 
        if not y > int(float(new_patch_shape[0])/2.):
            y = int(float(new_patch_shape[0])/2.)
    # In case there are no rows with ROI, it 
    # is set to the middle of the image
    else:
        y = int(float(new_patch_shape[0])/2)

    # The patches are extracted from the scan
    img_patch = scan[int(y - new_patch_shape[0] // 2):int(y + new_patch_shape[0] // 2), :, :]
    mask_patch = mask[int(y - new_patch_shape[0] // 2):int(y + new_patch_shape[0] // 2), :]

    # In case the image height is not 496, it is converted to the bigger and desired shape
    if img_height != 496:
        scan = resize(img_patch.astype(uint8), patch_shape, order=0, preserve_range=True).astype("uint8")
        mask = resize(mask_patch.astype(uint8), patch_shape, order=0, preserve_range=True).astype("uint8")
    else:
        scan = img_patch
        mask = mask_patch

    # Returns the reshaped scan and mask
    return scan, mask

def drop_patches(prob: float, volumes_list: list, model: str):
    """
    Randomly drops a percentage of extracted patches whose slice does
    not present fluid

    Args:
        prob (float): fraction of patches from each slice that will be 
            dropped in case there is no fluid in the slice
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in training
        model (str): name of the model used

    Return: None
    """
    # The path to the patches is dependent on the model selected
    if model == "2.5D":
        images_path = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
    else:
        images_path = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"

    # Declares the path where the true masks are located
    masks_path = IMAGES_PATH + "\\OCT_images\\segmentation\\masks\\int8\\"

    # Declares the path of the patches
    patches_slice_path = images_path + "slices\\"
    patches_mask_path = images_path + "masks\\"
    patches_roi_path = images_path + "roi\\"
    
    # Iterates through the ground truth masks
    for mask in listdir(masks_path):
        # Separates the name of the mask 
        # to obtain its information 
        mask_name_parts = mask.split("_")

        # Only considers the slices that are in the list 
        # of volumes that will be used in training
        if int(mask_name_parts[1][-3:]) in volumes_list:
            # Reads the mask
            fluid_mask = imread(str(masks_path + mask))

            # Checks if the slice has any of 
            # the three fluids
            irf_exists = any(fluid_mask==1)
            srf_exists = any(fluid_mask==2)
            ped_exists = any(fluid_mask==3)

            # In case there is no fluid, randomly eliminates patches
            if not (irf_exists or srf_exists or ped_exists):
                # Iterates through the patches extracted
                for patch in listdir(patches_slice_path):
                    # Extracts the information from the patches
                    patch_name_parts = patch.split("_")
                    # Checks if the name of the vendor, the volume, and the
                    # slice are the same
                    # In case the model is the 2.5D only considers the center 
                    # slice and eliminates the slices to it associated afterwards
                    if (((patch_name_parts[0] == mask_name_parts[0]) and \
                        (patch_name_parts[1] == mask_name_parts[1]) and \
                        (patch_name_parts[2] == mask_name_parts[2][:3]))
                        and (model != "2.5D" or len(patch_name_parts) == 6)):

                        # Randomly eliminates the indicated percentage 
                        # of patches
                        if (float(random_sample()) < prob):
                            remove(str(patches_slice_path + patch))
                            remove(str(patches_mask_path + patch))
                            remove(str(patches_roi_path + patch))

                            # In case it is the 2.5D segmentation model
                            # also deletes the associated before and 
                            # following patch  
                            if model == "2.5D":
                                # Deletes the previous slices
                                remove(str(patches_slice_path + patch[:-5] + "_before.tiff"))
                                remove(str(patches_mask_path + patch[:-5] + "_before.tiff"))
                                remove(str(patches_roi_path + patch[:-5] + "_before.tiff"))

                                # Deletes the following slices
                                remove(str(patches_slice_path + patch[:-5] + "_after.tiff"))
                                remove(str(patches_mask_path + patch[:-5] + "_after.tiff"))
                                remove(str(patches_roi_path + patch[:-5] + "_after.tiff"))

def patches_from_volumes(volumes_list: list, model: str):
    """
    Used to return the list of all the patches that are available to 
    train the network, knowing which volumes will be used

    Args:
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in training
        model (str): name of the model that will be trained

    Return:
        patches_list (List[str]): list of the name of the patches that 
            will be used to train the model
    """
    # The path to the patches is dependent on the model selected
    if model == "2.5D":
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
    else:
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
        
    # Iterates through the available patches
    # and registers the name of those that are 
    # from the volumes that will be used in 
    # training, returning that list
    patches_list = []
    slices_path = images_folder + "slices\\"
    for patch_name in listdir(slices_path):
        volume = patch_name.split("_")[1][-3:]
        volume = int(volume)
        if volume in volumes_list:
            patches_list.append(patch_name)
    return patches_list

def images_from_volumes(volumes_list: list):
    """
    Used to return the list of all the images that are available to 
    test the network, knowing which volumes will be used

    Args:
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in testing

    Return:
        images_list (List[str]): list of the name of the image that 
            will be used to test the model
    """
    # Declares the path to the images
    images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\slices\\uint8\\"
        
    # Iterates through the available images
    # and registers the name of those that are 
    # from the volumes that will be used in 
    # testing, returning that list
    images_list = []
    for image_name in listdir(images_folder):
        volume = image_name.split("_")[1][-3:]
        volume = int(volume)
        if volume in volumes_list:
            images_list.append(image_name)
    return images_list

class TrainDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called TrainDataset 
    with the available images, thus simplifying the training
    process
    """
    def __init__(self, train_volumes: list, model: str, 
                 patch: bool, fluid: int=None):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in training

        Args: 
            train_volumes(List[float]): list of the training 
                volumes that will be used to train the model
            model (str): name of the model that will be trained
            patch (bool): flag that indicates whether the 
                training will be done using patches or not
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, and the fluid to segment in 
        # case it is used
        super().__init__()        
        self.patch = patch
        self.model = model
        if patch:
            self.images_names = patches_from_volumes(train_volumes, model)
        else:
            self.images_names = images_from_volumes(train_volumes)

        # Random Rotation has a probability of 0.5 of rotating 
        # the image between 0 and 10 degrees
        # Random Horizontal Flip has a probability of 0.5 
        # flipping the image horizontally
        # Interpolation is set to nearest to minimize errors 
        # in categorical data
        self.transforms = Compose([
            RandomApply([RandomRotation(degrees=[0,10], interpolation=InterpolationMode.NEAREST)], p=0.5),
            RandomHorizontalFlip(p=0.5)])
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            None
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in training
        when an index is given, utilized to access the list of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            None
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # The path to read the images is different depending on the model
        if self.patch:
            if self.model == "2.5D":
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
            else:
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\" + self.images_names[index]
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\"
            mask_name = images_folder + "masks\\" + self.images_names[index]            
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\uint8\\" + self.images_names[index]
            mask_name = images_folder + "masks\\int8\\" + self.images_names[index]

        # Reads the image and the
        # fluid mask
        scan = imread(slice_name)
        mask = imread(mask_name)

        # In case the selected model is the 2.5D, also loads the previous
        # and following slice
        if self.model == "2.5D":
            scan_before = imread(str(slice_name[:-5] + "_before.tiff"))
            scan_after = imread(str(slice_name[:-5] + "_after.tiff"))
            # Stacks them to apply the transformations
            scan = stack(arrays=[scan_before, scan, scan_after], axis=0)

        # In case the model selected is the UNet3, all the labels 
        # that are not the one desired to segment are set to 0
        if self.model == "UNet3":
            mask = ((mask == self.fluid).astype(int) * self.fluid)

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Expands the scan dimentions to 
        # include an extra channel of value 1
        # as the first channel
        # The mask dimensions are also expanded 
        # to match
        if self.model != "2.5D":
            scan = expand_dims(scan, axis=0)
        mask = expand_dims(mask, axis=0)

        # Converts the scan and mask 
        # to a PyTorch Tensor
        scan = torch.from_numpy(scan)
        mask = torch.from_numpy(mask)

        # Forms a stack with the scan and the mask
        # Initial Scan Shape: 1 x H x W / 3 x H x W
        # Initial Mask Shape: 1 x H x W
        # Resulting Shape: 2 x H x W / 4 x H x W
        resulting_stack = torch.cat([scan, mask], dim=0)

        # Applies the transfomration to the stack
        transformed = self.transforms(resulting_stack)

        # Separate the scan and the mask from the stack
        # Keeps the extra dimension on the slice but not on the mask
        if self.model != "2.5D":
            scan, mask = transformed[0].unsqueeze(0), transformed[1]
        # Handles it differently for the 2.5D model, ensuring the correct order of slices 
        else:
            scan = torch.cat([transformed[0], transformed[1], transformed[2]], dim=0)
            mask = transformed[3]

        # Converts the scans back to NumPy
        scan = scan.numpy()
        mask = mask.numpy()

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask}
        return sample

class ValidationDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called ValidationDataset 
    with the available images, thus simplifying the training
    process
    """
    def __init__(self, val_volumes: list, model: str,
                 patch: bool, fluid: int=None):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in validation

        Args: 
            val_volumes(List[float]): list of the validation 
                volumes that will be used to validate the model
            model (str): name of the model that will be trained
            patch (bool): flag that indicates whether the 
                training will be done using patches or not
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, and the fluid to segment in 
        # case it is used
        super().__init__()        
        self.patch = patch
        self.model = model
        if patch:
            self.images_names = patches_from_volumes(val_volumes, model)
        else:
            self.images_names = images_from_volumes(val_volumes)
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in validation

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            None
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in 
        validation when an index is given, utilized to access the 
        list of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            None
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # The path to read the images is different depending on the model
        if self.patch:
            if self.model == "2.5D":
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
            else:
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\" + self.images_names[index]
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\"
            mask_name = images_folder + "masks\\" + self.images_names[index]            
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\uint8\\" + self.images_names[index]
            mask_name = images_folder + "masks\\int8\\" + self.images_names[index]

        # Reads the image and the
        # fluid mask
        scan = imread(slice_name)
        mask = imread(mask_name)

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Expands the scan dimentions to 
        # include an extra channel of value 1
        # as the first channel
        # The mask dimensions are also expanded 
        # to match
        if self.model != "2.5D":
            scan = expand_dims(scan, axis=0)

        # In case the selected model is the 2.5D, also loads the previous
        # and following slice
        if self.model == "2.5D":
            scan_before = imread(str(slice_name[:-5] + "_before.tiff"))
            scan_after = imread(str(slice_name[:-5] + "_after.tiff"))
            # Stacks them to apply the transformations
            scan = stack(arrays=[scan_before, scan, scan_after], axis=0)

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask}
        return sample
    
class TestDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called TestDataset 
    with the available images, thus simplifying the training
    process
    """
    def __init__(self, test_volumes: list, model: str, fluid: int=None):
        """
        Initiates the Dataset object and gets the possible 
        names of the images that will be used in testing

        Args: 
            test_volumes(List[float]): list of the test 
                volumes that will be used to test the model
            model (str): name of the model that will be trained
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, and the fluid to segment in 
        # case it is used
        super().__init__()
        self.model = model
        self.images_names = images_from_volumes(test_volumes)
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in testing

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            None
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in 
        validation when an index is given, utilized to access the 
        list of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            None
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # Declares the path to the images
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\slices\\uint8\\"        
        # Declares the path to the masks
        masks_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\masks\\int8\\"
        # Declares the path to the ROI masks
        rois_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\roi\\int8\\"

        # Indicates the path to the image depending on the index given,
        # which is associated with the image name
        slice_name = images_folder + self.images_names[index]
        mask_name = masks_folder + self.images_names[index]
        roi_name = rois_folder + self.images_names[index]

        # Reads the image and the
        # fluid mask
        scan = imread(slice_name)
        mask = imread(mask_name)
        roi = imread(roi_name)

        # In case the selected model is the 2.5D, also loads the previous
        # and following slice
        if self.model == "2.5D":
            # Gets the number of the desired slice
            slice_num = int(slice_name.split("_")[2].split(".")[0])
            # Calculates the index of the slice before and after
            slice_num_before = slice_num - 1
            slice_num_after = slice_num + 1
            # Creates the full path of the before and after slice according to the number calculated
            slice_before = images_folder + slice_name[-8:] + "_" + str(slice_num_before).zfill(3) + ".tiff"
            slice_after = images_folder + slice_name[-8:] + "_" + str(slice_num_after).zfill(3) + ".tiff"
            # In case the slice path does not exist, 
            # the current slice is loaded instead
            if exists(slice_before):
                scan_before = imread(slice_before)
            else:
                scan_before = imread(slice_name)
            if exists(slice_after):
                scan_after = imread(slice_after)
            else:
                scan_after = imread(slice_name)

            # Creates a stack with all the slices
            scan = stack(arrays=[scan_before, scan, scan_after], axis=-1)
        else:
            scan = expand_dims(scan, axis=-1)

        # In case the model selected is the UNet3, all the labels 
        # that are not the one desired to segment are set to 0
        if self.model == "UNet3":
            mask = ((mask == self.fluid).astype(int) * self.fluid)

        # The shape of the test images is handled
        scan, mask = handle_test_images(scan, mask, roi, patch_shape=(256, 512))

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask, "image_name": self.images_names[index]}
        return sample
