import lmdb
import torch
from collections import defaultdict
from io import BytesIO
from numpy import array, any, expand_dims, int8, max, min, ndarray, nonzero, stack, sum, uint8, zeros_like
from numpy.random import random_sample
from os import listdir, remove
from os.path import exists
from paths import IMAGES_PATH
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from torchvision.transforms.functional import hflip, rotate
from torchvision.transforms.v2 import InterpolationMode
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

def patches_from_volumes(volumes_list: list, model: str, 
                         patch_type: str, num_patches: int):
    """
    Used to return the list of all the patches that are available to 
    train the network, knowing which volumes will be used

    Args:
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in training
        model (str): name of the model that will be trained
        patch_type (str): string that indicates what type of patches 
            will be used. Can be "small", where patches of size 
            256x128 are extracted using the extract_patches function,
            "big", where patches of shape 496x512 are extracted from 
            each image, and patches of shape 496x128 are extracted from
            the slices
        num_patches (int): number of patches extracted from
            the images during vertical patch extraction to 
            train the model

    Return:
        patches_list (List[str]): list of the name of the patches that 
            will be used to train the model
    """
    # The path to the patches is dependent on the model selected
    if patch_type == "small":
        if model == "2.5D":
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\slices\\"
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\slices\\"
    else:
        images_folder = IMAGES_PATH + f"\\OCT_images\\segmentation\\{patch_type}_patches\\"
        if ((patch_type == "vertical") and (num_patches > 4)):
            images_folder = IMAGES_PATH + f"\\OCT_images\\segmentation\\{patch_type}_patches_overlap_{num_patches}\\"

    # Iterates through the available patches
    # and registers the name of those that are 
    # from the volumes that will be used in 
    # training, returning that list
    patches_list = []
    for patch_name in listdir(images_folder):
        volume = patch_name.split("_")[1][-3:]
        volume = int(volume)
        if volume in volumes_list:
            patches_list.append(patch_name)
    return patches_list

def generation_images_from_volumes(volumes_list: list, model_name: str, 
                                   oct_device: str='all'):
    """
    Used to return the list of all the patches that are available to 
    train the GAN, knowing which volumes will be used

    Args:
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in training
        model_name (str): name of the model that is being trained. Can 
            only be "UNet" or "GAN"
        oct_device (str): name of the OCT device from which the scans 
            while be used to train and validate the model. Can be 'all', 
            'Cirrus', 'Spectralis', 'T-1000', or 'T-2000'. The default 
            option is 'all'

    Return:
        patches_list (List[str]): list of the name of the patches that 
            will be used to train the model
    """
    # Sets the path to the images based on the model_name
    if model_name == "UNet":
        images_folder = IMAGES_PATH + "\\OCT_images\\generation\\slices_resized\\"
    elif model_name == "GAN":
        images_folder = "C:\slices_resized_64_patches\\"

    # Iterates through the available images
    # and registers the name of those that are 
    # from the volumes that will be used in 
    # training, returning that list
    images_list = []
    for patch_name in listdir(images_folder):
        parts = patch_name.split("_")
        volume_name = parts[1]
        volume_set = volume_name[:-3].lower()
        volume_number = int(volume_name[-3:])
        volume = f"{volume_number}_{volume_set}"

        if (volume in volumes_list) and (oct_device == parts[0] or oct_device == 'all'):
            images_list.append((images_folder, patch_name))

    # Creates a dictionary that will match the id of a volume to a path
    volume_dict = defaultdict(lambda: defaultdict(list))
    # Iterates through the 
    # full list of images
    for folder, path in images_list:
        # Extracts the volume identifier which is 
        # all the information prior to the slice
        # number
        parts = path.split("_")
        # Gets the information differently depending 
        # on the folder that is handling, identifiable 
        # by the number of components in the name 
        # separated by '_'
        if len(parts) == 4:
            volume_id = "_".join(parts[:-2])
            slice_number = int(parts[-2])
        else:
            volume_id = "_".join(parts[:-1])
            slice_number = int(parts[-1].split(".")[0])
        volume_dict[volume_id][slice_number].append((folder, path))

    # Remove the first and last slice for each volume
    # Initiates a list that 
    # will have the filtered paths
    filtered_paths = []
    # Iterates through all the volumes in the dataset
    for volume_id, slices_dict in volume_dict.items():
        # Sorts paths for the volume to ensure correct order
        slice_numbers = sorted(slices_dict.keys())
        # Excludes the first and last slices,
        # while ensuring that there are enough 
        # slices to analyze 
        if len(slice_numbers) > 2:
            for sn in slice_numbers[1:-1]:
                filtered_paths.extend(slices_dict[sn])
    
    # Return only the filenames with their respective folder
    return [f"{path}" for folder, path in filtered_paths]

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

class CustomTransform:
    """
    Creates a PyTorch custom transformation since 
    the mask and the scan need to be handled
    differently 
    """
    def __init__(self, p=0.5):
        """
        CustomTransform class constructor that 
        receives as argument the probability of 
        a transformation occuring

        Args:
            self (CustomTransform object): the 
                CustomTransform object itself
                to create
            p (float): probability of a 
                transformation being applied

        Returns:
            None
        """
        self.p = p

    def __call__(self, sample, model):
        """
        Function that will apply the transformations to the received sample
        which consists of the scan and the fluid mask

        Args:
            self (CustomTransform object): the CustomTransform object itself
                that contains the probability of a transformation
            sample (PyTorch Tensor): tensor of shape (2xHxW) if the model is 
                not 2.5D and (4xHxW) if the model is 2.5D
            model (str): name of the model that is being used

        Return:
            (PyTorch Tensor) tensor of shape (2xHxW) if the model is 
                not 2.5D and (4xHxW) if the model is 2.5D after the 
                transformations have been applied
        """
        # Extracts the scan and the mask from the sample
        if model != "2.5D":
            scan, mask = sample[0,:,:], sample[1,:,:]
        else:
            scan, mask = torch.stack(sample[0:3]), sample[3]

        # Leaves the scan with shape 1xHxW
        scan = scan.unsqueeze(0) 
        mask = mask.unsqueeze(0)
        # In case the probability 
        # is below the threshold, 
        # executes the rotation 
        # transformation
        if torch.rand(1) < self.p:
            # The rotation is done independently because two different 
            # interpolation modes are being used
            # Calculates the random rotation angle that will be the same for both 
            # transformations
            angle = torch.randint(0, 5, (1,)).item()  
            # Applies the rotation with bilinear interpolation to the scan
            if model != "2.5D":
                scan = rotate(scan, angle, interpolation=InterpolationMode.BILINEAR) 
            else:
                scan = torch.stack([rotate(scan[i], angle, interpolation=InterpolationMode.BILINEAR) for i in range(3)]) 
            # Applies the rotation with nearest interpolation to the mask
            mask = rotate(mask, angle, interpolation=InterpolationMode.NEAREST)  

        # In case the probability 
        # is below the threshold, 
        # executes the horizontal 
        # flip transformation
        if torch.rand(1) < self.p:
            scan = hflip(scan)
            mask = hflip(mask)

        # Returns the result, stacked
        return torch.stack([scan, mask])

class TrainDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called TrainDataset 
    with the available images, thus simplifying the training
    process
    """
    def __init__(self, train_volumes: list, model: str, 
                 patch_type: str, num_patches: int, 
                 fluid: int=None, number_of_channels: int=1):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in training

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            train_volumes(List[float]): list of the training 
                volumes that will be used to train the model
            model (str): name of the model that will be trained
            patch_type (str): string that indicates what type 
                of patches will be used. Can be "small", where 
                patches of size 256x128 are extracted using the
                extract_patches function, "big", where patches 
                of shape 496x512 are extracted from each image,
                and patches of shape 496x128 are extracted from
                the slices
            num_patches (int): number of patches extracted from
                the images during vertical patch extraction to 
                train the model
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
            number_of_channels (int): number of input channels
                in the network. The default value is 1 but can 
                also be 2 in case there is relative distance 
                maps, for example
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, the number of patches extracted
        # and the fluid to segment in case it is used
        super().__init__()        
        self.number_of_channels = number_of_channels
        self.patch_type = patch_type
        self.model = model
        self.num_patches = num_patches
        self.images_names = patches_from_volumes(train_volumes, 
                                                 model, patch_type,
                                                 num_patches)

        # Random Rotation has a probability of 0.5 of rotating 
        # the image between 0 and 10 degrees
        # Random Horizontal Flip has a probability of 0.5 
        # flipping the image horizontally
        # Interpolation is set to nearest to minimize errors 
        # in categorical data
        self.transforms = CustomTransform(p=0.5)
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
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
            (dict{str: PyTorch Tensor, str: PyTorch Tensor}): returns the
                training image and the training mask, associated with 
                the names "scan" and "mask", respectively
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # The path to read the images is different depending on the model
        if self.patch_type == "small":
            if self.model == "2.5D":
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
            else:
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\" + self.images_names[index]
            mask_name = images_folder + "masks\\" + self.images_names[index]
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + f"{self.patch_type}_patches\\" + self.images_names[index]
            mask_name = images_folder + f"{self.patch_type}_masks\\" + self.images_names[index]
            rdms_name = images_folder + f"{self.patch_type}_dms\\" + self.images_names[index]
            if ((self.patch_type == "vertical") and (self.num_patches > 4)):
                slice_name = images_folder + f"{self.patch_type}_patches_overlap_{self.num_patches}\\" + self.images_names[index]
                mask_name = images_folder + f"{self.patch_type}_masks_overlap_{self.num_patches}\\" + self.images_names[index]
                rdms_name = images_folder + f"{self.patch_type}_dms_overlap_{self.num_patches}\\" + self.images_names[index]

        # Reads the image, the
        # fluid mask, and the 
        # relative distance maps
        # in case it is applicable
        scan = imread(slice_name)
        mask = imread(mask_name)
        if (self.number_of_channels > 1) and \
            (self.patch_type == "vertical") and \
            (self.model == "UNet"):
            rdm = imread(rdms_name)

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
            mask = ((mask == int(self.fluid)).astype(uint8))

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
        resulting_stack = torch.cat([scan.float(), mask.long()], dim=0)

        # Applies the transfomration to the stack
        transformed = self.transforms(resulting_stack, self.model)

        # Separate the scan and the mask from the stack
        # Keeps the extra dimension on the slice but not on the mask
        if self.model != "2.5D":
            scan, mask = transformed[0], transformed[1]
        # Handles it differently for the 2.5D model, ensuring the correct order of slices 
        else:
            scan = torch.cat([transformed[0].float(), transformed[1].float(), transformed[2].float()], dim=0)
            mask = transformed[3]

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1 of the image
        # The mean and the standard deviation of 
        # the scan are considered 128 
        scan = (scan - 128.) / 128.

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask}
        return sample

class ValidationDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called ValidationDataset 
    with the available images, thus simplifying the validation
    process
    """
    def __init__(self, val_volumes: list, model: str,
                 patch_type: str, num_patches: int, 
                 fluid: int=None, number_of_channels: int=1):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in validation

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            val_volumes(List[float]): list of the validation 
                volumes that will be used to validate the model
            model (str): name of the model that will be trained
            patch_type (str): string that indicates what type 
                of patches will be used. Can be "small", where 
                patches of size 256x128 are extracted using the
                extract_patches function, "big", where patches 
                of shape 496x512 are extracted from each image,
                and patches of shape 496x128 are extracted from
                the slices
            num_patches (int): number of patches extracted from
                the images during vertical patch extraction to 
                train the model
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
            number_of_channels (int): number of input channels
                in the network. The default value is 1 but can 
                also be 2 in case there is relative distance 
                maps, for example
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the number of vertical patches, 
        # and the fluid to segment in case it is used
        super().__init__()   
        self.number_of_channels = number_of_channels     
        self.patch_type = patch_type
        self.model = model
        self.num_patches = num_patches
        self.images_names = patches_from_volumes(val_volumes, model, 
                                                 patch_type, num_patches)
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in validation

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
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
            (dict{str: PyTorch Tensor, str: PyTorch Tensor}): returns the
                training image and the training mask, associated with 
                the names "scan" and "mask", respectively
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # The path to read the images is different depending on the model
        if self.patch_type == "small":
            if self.model == "2.5D":
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
            else:
                images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + "slices\\" + self.images_names[index]
            mask_name = images_folder + "masks\\" + self.images_names[index]
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\"
            # Indicates the path to the image depending on the index given,
            # which is associated with the image name
            slice_name = images_folder + f"{self.patch_type}_patches\\" + self.images_names[index]
            mask_name = images_folder + f"{self.patch_type}_masks\\" + self.images_names[index]
            rdms_name = images_folder + f"{self.patch_type}_dms\\" + self.images_names[index]
            if ((self.patch_type == "vertical") and (self.num_patches > 4)):
                slice_name = images_folder + f"{self.patch_type}_patches_overlap_{self.num_patches}\\" + self.images_names[index]
                mask_name = images_folder + f"{self.patch_type}_masks_overlap_{self.num_patches}\\" + self.images_names[index]
                rdms_name = images_folder + f"{self.patch_type}_dms_overlap_{self.num_patches}\\" + self.images_names[index]

        # Reads the image, the
        # fluid mask, and the 
        # relative distance maps
        # in case it is applicable
        scan = imread(slice_name)
        mask = imread(mask_name)
        if (self.number_of_channels > 1) and (self.patch_type == "vertical"):
            rdm = imread(rdms_name)

        # In case the model selected is the UNet3, all the labels 
        # that are not the one desired to segment are set to 0
        if self.model == "UNet3":
            mask = ((mask == self.fluid).astype(uint8))

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Expands the scan dimentions to 
        # include an extra channel of value 1
        # as the first channel
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
    with the available images, thus simplifying the testing
    process
    """
    def __init__(self, test_volumes: list, model: str, 
                 patch_type: str, resize_images: bool,
                 resize_shape: tuple, fluid: int=None,
                 number_of_channels: int=1):
        """
        Initiates the Dataset object and gets the possible 
        names of the images that will be used in testing

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            test_volumes(List[float]): list of the test 
                volumes that will be used to test the model
            model (str): name of the model that will be trained
            patch_type (str): string that indicates what type of 
                patches will be used. Can be "small", where 
                patches of size 256x128 are extracted using the 
                extract_patches function, "big", where patches 
                of shape 496x512 are extracted from each image,
                and patches of shape 496x128 are extracted from
                the slices
            resize_images (bool): flag that indicates whether 
                the images will be resized or not in testing 
            resize_shape (tuple): tuple that contains the 
                shape resulting from the resizing of the images
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network
            number_of_channels (int): number of input channels
                in the network. The default value is 1 but can 
                also be 2 in case there is relative distance 
                maps, for example
                
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, and the fluid to segment in 
        # case it is used
        super().__init__()
        self.number_of_channels = number_of_channels
        self.patch_type = patch_type
        self.model = model
        self.images_names = images_from_volumes(test_volumes)
        self.fluid = fluid
        self.resize = resize_images 
        self.resize_shape = resize_shape

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in testing

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
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
            (dict{str: PyTorch Tensor, str: PyTorch Tensor}): returns the
                training image, the training mask, and the name of the 
                image associated with the names "scan", "mask", and 
                "image_name", respectively
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
        # Declares the path to the relative distance maps
        rdms_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\rdms"

        # Indicates the path to the image depending on the index given,
        # which is associated with the image name
        slice_name = images_folder + self.images_names[index]
        mask_name = masks_folder + self.images_names[index]
        roi_name = rois_folder + self.images_names[index]
        rdm_name = rdms_folder + self.images_names[index]

        # Reads the image and the
        # fluid mask
        scan = imread(slice_name)
        mask = imread(mask_name)
        if self.model == "2.5D":
            roi = imread(roi_name)
        if self.number_of_channels == 2 and self.model != "UNet":
            rdm = imread(rdm_name)

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
        elif (self.number_of_channels > 1):
            scan = stack(arrays=[scan, rdm], axis=-1)
        else:
            scan = expand_dims(scan, axis=-1)

        # In case the model selected is the UNet3, all the labels 
        # that are not the one desired to segment are set to 0
        if self.model == "UNet3":
            mask = ((mask == self.fluid).astype(uint8))

        # The shape of the test images is handled in different ways depending 
        # whether it was done using patches or not
        if self.patch_type == "small":
            scan, mask = handle_test_images(scan, mask, roi, patch_shape=(256, 512))

        # If the images are desirerd to be resized
        if self.resize:
            # Resizes the images to the shape of the Spectralis scan
            scan = resize(scan, self.resize_shape, preserve_range=True, 
                                        anti_aliasing=True)
            mask = resize(mask, self.resize_shape, order=0, preserve_range=True, 
                                        anti_aliasing=False)

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask, "image_name": self.images_names[index]}
        return sample

class TrainDatasetLMDB(Dataset):
    """
    Initiates the PyTorch object Dataset called TrainDatasetLMDB 
    with the available images, thus simplifying the training
    process. In this case, to allow faster transitions 
    between the HDD and the RAM, we are handling the data as 
    a LMDB file.
    """
    def __init__(self, model: str, num_patches: int, 
                 img_lmdb_path: str, mask_lmdb_path: str):
        """
        Initiates the Dataset object and gets the path to the 
        LMDB dataset

        Args:
            self (PyTorch Dataset): the PyTorch dataset object
                itself 
            model (str): name of the model that will be trained
            num_patches (int): number of patches per image 
                extracted 
            img_lmdb_path (str): path to the LMDB dataset with 
                the images            
            mask_lmdb_path (str): path to the LMDB dataset with 
                the masks
                
        Return:
            None
        """
        # Initiates the Dataset object
        super().__init__()
        # Saves in the object the name of 
        # the model, the number of patches
        # the path to the LMDB dataset with 
        # the patches and with the masks 
        self.model = model
        self.num_patches = num_patches
        self.img_lmdb_path = img_lmdb_path
        self.mask_lmdb_path = mask_lmdb_path

        # Opens the LMDB environment and check if num_samples exists
        with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env:
            with img_env.begin() as txn:
                num_samples = txn.get(b'num_samples')
                if num_samples is None:
                    raise ValueError(f"LMDB error: 'num_samples' key not found in {self.img_lmdb_path}")
                # Saves the number of samples available 
                # in the object
                self.length = int(num_samples.decode())

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
        """
        return self.length

    def __getitem__(self, index):
        """
        Gets an image and the respective mask from the list of 
        images that can be used in training when an index is given, 
        utilized to access the LMDB dataset
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            (dict{str: PyTorch Tensor, str: PyTorch Tensor}): returns 
                the training image and the training mask, associated 
                with the names "scan" and "mask", respectively
        """
        # Opens the LMDB environment for the images and the masks datasets
        with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env, \
             lmdb.open(self.mask_lmdb_path, readonly=True, lock=False) as mask_env:
            # With the same index, accesses the encoded images and masks 
            # and reads them as a PIL object
            with img_env.begin() as img_txn, mask_env.begin() as mask_txn:
                # Load the patch
                img_key = f"img_{index}".encode()
                img_bytes = img_txn.get(img_key)
                img = Image.open(BytesIO(img_bytes)).convert("L") if img_bytes else None

                # Load the segmentation mask
                mask_key = f"mask_{index}".encode()
                mask_bytes = mask_txn.get(mask_key)
                mask = Image.open(BytesIO(mask_bytes)).convert("L") if mask_bytes else None

        # In case the index does not find any image for the 
        # index, raises an error
        if img is None or mask is None:
            raise ValueError(f"Missing data for index {index}")

        # Converts the PIL 
        # object to a NumPy
        #  array
        img = array(img)
        mask = array(mask)

        # Handles the cases for 2.5D 
        # segmentation 
        if self.model == "2.5D":
            # Loads the previous and following image as an array of zeros 
            # with the same shape
            img_before, img_after = zeros_like(img), zeros_like(img)
            # Opens the environment once again to load the previous and following 
            # images and masks
            with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env:
                with img_env.begin() as img_txn:
                    # Gets the encoded previous and after image
                    before_key = f"img_{max(0, index - 1)}".encode()
                    after_key = f"img_{min(self.length - 1, index + 1)}".encode()

                    # Gets the encoded previous and after mask
                    before_bytes = img_txn.get(before_key)
                    after_bytes = img_txn.get(after_key)

                    # Loads the previous and following image as a NumPy array
                    img_before = array(Image.open(BytesIO(before_bytes)).convert("L")) if before_bytes else zeros_like(img)
                    img_after = array(Image.open(BytesIO(after_bytes)).convert("L")) if after_bytes else zeros_like(img)
            # Stacks the images together, ending with shape (3, H, W)
            img = stack([img_before, img, img_after], axis=0)

        # Expands the dimensions of the image 
        # and the mask to include the channel 
        # dimension. In the image, it only 
        # happens if it is not 2.5D
        if self.model != "2.5D":
            # Shape: (1, H, W)
            img = expand_dims(img, axis=0)
        # Shape: (1, H, W)
        mask = expand_dims(mask, axis=0)

        # Converts the images to PyTorch Tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        # Performs Z-Score normalization 
        # in the images
        img = (img - 128.) / 128.

        return {"scan": img, "mask": mask}

class ValidationDatasetLMDB(Dataset):
    """
    Initiates the PyTorch object Dataset called TrainDatasetLMDB 
    with the available images, thus simplifying the training
    process. In this case, to allow faster transitions 
    between the HDD and the RAM, we are handling the data as 
    a LMDB file.
    """
    def __init__(self, model: str, num_patches: int, 
                 img_lmdb_path: str, mask_lmdb_path: str, 
                 fluid: int=None):
        """
        Initiates the Dataset object and gets the path to the 
        LMDB dataset

        Args:
            self (PyTorch Dataset): the PyTorch dataset object
                itself 
            model (str): name of the model that will be trained
            num_patches (int): number of patches per image 
                extracted 
            img_lmdb_path (str): path to the LMDB dataset with 
                the images            
            mask_lmdb_path (str): path to the LMDB dataset with 
                the masks
            fluid (int): label of fluid that is expected to 
                segment. Optional because it is only used in
                one network    
                
        Return:
            None
        """
        # Initiates the Dataset object
        super().__init__()
        # Saves in the object the name of 
        # the model, the number of patches,
        # the number of the fluid the path 
        # to the LMDB dataset with the 
        # patches and with the masks 
        self.model = model
        self.num_patches = num_patches
        self.fluid = fluid
        self.img_lmdb_path = img_lmdb_path
        self.mask_lmdb_path = mask_lmdb_path

        # Opens the LMDB environment and checks if num_samples exists
        with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env:
            with img_env.begin() as txn:
                # Gets the number of samples from the 
                # LMDB environment
                num_samples = txn.get(b'num_samples')
                if num_samples is None:
                    raise ValueError(f"LMDB error: 'num_samples' key not found in {self.img_lmdb_path}")
                # Saves the number of samples available 
                # in the Dataset object
                self.length = int(num_samples.decode())

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in validation

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
        """
        return self.length

    def __getitem__(self, index):
        """
        Gets an image and the respective mask from the list of 
        images that can be used in validation when an index is 
        given, utilized to access the LMDB dataset
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            (dict{str: PyTorch Tensor, str: PyTorch Tensor}): returns 
                the validation image and the validation mask, associated 
                with the names "scan" and "mask", respectively
        """
        # Opens the LMDB environment
        with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env, \
            lmdb.open(self.mask_lmdb_path, readonly=True, lock=False) as mask_env:
            # With the same index, accesses the encoded images and masks 
            # and reads them as a PIL object
            with img_env.begin() as img_txn, mask_env.begin() as mask_txn:
                # Loads the image bytes from the 
                # LMDB train file for the 
                # respetive index
                img_key = f"img_{index}".encode()
                img_bytes = img_txn.get(img_key)
                # Saves the bytes as a PIL image
                img = Image.open(BytesIO(img_bytes)).convert("L") if img_bytes else None

                # Loads the mask bytes from the 
                # LMDB train file for the 
                # respetive index
                mask_key = f"mask_{index}".encode()
                mask_bytes = mask_txn.get(mask_key)
                mask = Image.open(BytesIO(mask_bytes)).convert("L") if mask_bytes else None

        # If no image or mask is found 
        # for the current index, a error 
        # is raised
        if img is None or mask is None:
            raise ValueError(f"Missing data for index {index}")

        # Converts the PIL Image 
        # object to a NumPy array
        img = array(img)
        mask = array(mask)

        # Handles the cases for 
        # 2.5D Model
        if self.model == "2.5D":
            # Initiates the image that is located previous and after the 
            # current image as a matrix of zeros
            img_before, img_after = zeros_like(img), zeros_like(img)
            # Opens the LMDB environment that contains the validation patches
            with lmdb.open(self.img_lmdb_path, readonly=True, lock=False) as img_env:
                with img_env.begin() as img_txn:
                    # Gets the previous and the following index 
                    # encoded in bytes 
                    before_key = f"img_{max(0, index - 1)}".encode()
                    after_key = f"img_{min(len(self.images_names) - 1, index + 1)}".encode()
                    # Gets the bytes of the previous and 
                    # following slices 
                    before_bytes = img_txn.get(before_key)
                    after_bytes = img_txn.get(after_key)
                    # Saves the image before and after as a NumPy array after loading it as a PIL Image object
                    img_before = array(Image.open(BytesIO(before_bytes)).convert("L")) if before_bytes else zeros_like(img)
                    img_after = array(Image.open(BytesIO(after_bytes)).convert("L")) if after_bytes else zeros_like(img)
            # Stacks the middle with the previous and following 
            # images, attaining a shape of (3, H, W) 
            img = stack([img_before, img, img_after], axis=0)

        # Expands the image 
        # dimensions if not 2.5D
        if self.model != "2.5D":
            # Adds the channel dimension to 
            # the image, attaining a shape of (1, H, W)
            img = expand_dims(img, axis=0) 

        # In case the image is not already a 
        # PyTorch Tensor
        if not isinstance(mask, torch.Tensor):
            # Converts the mask to a PyTorch 
            # Tensor of type long
            mask = torch.from_numpy(mask).long()

        # Converts the image to a PyTorch 
        # tensor of type float
        img = torch.from_numpy(img).float()
        # Ensures the mask 
        # is of type long
        mask = mask.long()

        # Performs Z-Score normalization 
        # in the image
        img = (img - 128.) / 128.

        return {"scan": img, "mask": mask}

class TrainDatasetGAN(Dataset):
    """
    Initiates the PyTorch object Dataset called 
    TrainDatasetGAN with the available image's paths, 
    thus simplifying the training process
    """
    def __init__(self, train_volumes: list, model_name: str,
                 oct_device: str='all'):
        """
        Initiates the Dataset object and gets the possible 
        names of the images that will be used in training

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            train_volumes (List[float]): list of the training 
                volumes that will be used to validate the model
            model_name (str): name of the model that will be 
                trained to generate the images
            oct_device (str): name of the OCT device from which 
                the scans while be used to train and validate 
                the model. Can be 'all', 'Cirrus', 'Spectralis', 
                'T-1000', or 'T-2000'. The default option is 
                'all'
                
        Return:
            None
        """
        # Initiates the Dataset object
        super().__init__()
        # Gets a list of paths to all the images that will be used to 
        # train the model and the name of the model
        self.images_names = generation_images_from_volumes(train_volumes, 
                                                           model_name, 
                                                           oct_device)
        self.model_name = model_name

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in 
        validation when an index is given, utilized to access the list 
        of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            sample (dict{str: PyTorch tensor, str: str}): returns the
                stack of images composed by the previous image, the 
                middle image and the following image, associated with 
                the key 'stack'
        """
        # Declares the path to the images
        if self.model_name == "UNet":
            images_folder = IMAGES_PATH + "\\OCT_images\\generation\\slices_resized\\"
        elif self.model_name == "GAN":
            images_folder = "C:\slices_resized_64_patches\\"

        # Gets the path of the image by the given index
        image_path = images_folder + self.images_names[index]

        # Reads the middle image as a NumPy array and normalizes 
        # it to 0-1 range
        img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()
        if self.model_name == "UNet":
            # Gets the number of the slice
            img_number = int(image_path.split(".")[0][-3:])
            # Sets the name of the previous and following images
            prev_img_path = image_path.split(".")[0][:-3] + str(img_number - 1).zfill(3) + ".tiff"
            next_img_path = image_path.split(".")[0][:-3] + str(img_number + 1).zfill(3) + ".tiff"
        elif self.model_name == "GAN":
            # Gets the information of the slice that is being read
            vendor, volume_id, img_number, patch_number = self.images_names[index].split("_")
            # Sets the name of the previous and following images
            prev_img_num = f"{int(img_number) - 1:03}"
            next_img_num = f"{int(img_number) + 1:03}"
            # Sets a base name that will only change the number of the slice
            base_name = f"{vendor}_{volume_id}_{{}}_{patch_number}"
            # Changes the number of the slice in the path
            prev_img_path = f"{images_folder}{base_name.format(prev_img_num)}"
            next_img_path = f"{images_folder}{base_name.format(next_img_num)}"

        # Loads the previous and the following images while normalizing 
        # to 0-1 range
        prev_img = torch.from_numpy(((imread(prev_img_path) / 255.) - 0.5) * 2.).float()
        next_img = torch.from_numpy(((imread(next_img_path) / 255.) - 0.5) * 2.).float()

        # Organizes the data in a dictionary that contains the images 
        # stacked along the first axis under the key "stack"
        sample = {"stack": torch.stack([prev_img, img, next_img], axis=0)}

        return sample

class ValidationDatasetGAN(Dataset):
    """
    Initiates the PyTorch object Dataset called 
    ValidationDatasetGAN with the available image's paths, 
    thus simplifying the validation process
    """
    def __init__(self, val_volumes: list, model_name: str,
                 oct_device: str='all'):
        """
        Initiates the Dataset object and gets the possible 
        names of the images that will be used in validation

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            val_volumes(List[float]): list of the validation 
                volumes that will be used to validate the model
            model_name (str): name of the model that will be 
                validated to generate the images
            oct_device (str): name of the OCT device from which 
                the scans while be used to train and validate 
                the model. Can be 'all', 'Cirrus', 'Spectralis', 
                'T-1000', or 'T-2000'. The default option is 
                'all'

        Return:
            None
        """
        # Initiates the Dataset object
        super().__init__()
        # Gets a list of paths to all the images that will be used to 
        # validate the model and the name of the model
        self.images_names = generation_images_from_volumes(val_volumes, 
                                                           model_name,
                                                           oct_device)
        self.model_name = model_name

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in validation

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in 
        validation when an index is given, utilized to access the list 
        of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            sample (dict{str: PyTorch tensor, str: str}): returns the
                stack of images composed by the previous image, the 
                middle image and the following image, associated with 
                the key 'stack'
        """
        # Declares the path to the images
        if self.model_name == "UNet":
            images_folder = IMAGES_PATH + "\\OCT_images\\generation\\slices_resized\\"
        elif self.model_name == "GAN":
            images_folder = "C:\slices_resized_64_patches\\"

        # Gets the path of the image by the given index
        image_path = images_folder + self.images_names[index]

        # Reads the middle image as a NumPy array and normalizes 
        # it to 0-1 range
        img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()
        if self.model_name == "UNet":
            # Gets the number of the slice
            img_number = int(image_path.split(".")[0][-3:])
            # Sets the name of the previous and following images
            prev_img_path = image_path.split(".")[0][:-3] + str(img_number - 1).zfill(3) + ".tiff"
            next_img_path = image_path.split(".")[0][:-3] + str(img_number + 1).zfill(3) + ".tiff"
        elif self.model_name == "GAN":
            # Gets the information of the slice that is being read
            vendor, volume_id, img_number, patch_number = self.images_names[index].split("_")
            # Sets the name of the previous and following images
            prev_img_num = f"{int(img_number) - 1:03}"
            next_img_num = f"{int(img_number) + 1:03}"
            # Sets a base name that will only change the number of the slice
            base_name = f"{vendor}_{volume_id}_{{}}_{patch_number}"
            # Changes the number of the slice in the path
            prev_img_path = f"{images_folder}{base_name.format(prev_img_num)}"
            next_img_path = f"{images_folder}{base_name.format(next_img_num)}"

        # Loads the previous and the following images while normalizing 
        # to 0-1 range
        prev_img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()
        next_img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()

        # Organizes the data in a dictionary that contains the images 
        # stacked along the first axis under the key "stack"
        sample = {"stack": torch.stack([prev_img, img, next_img], axis=0)}

        return sample

class TestDatasetGAN(Dataset):
    """
    Initiates the PyTorch object Dataset called 
    TestDatasetGAN with the available image's paths, thus 
    simplifying the testing process
    """
    def __init__(self, test_volumes: list, oct_device: str='all'):
        """
        Initiates the Dataset object and gets the possible 
        names of the images that will be used in testing

        Args: 
            self (PyTorch Dataset): the PyTorch Dataset object 
                itself
            test_volumes(List[float]): list of the test 
                volumes that will be used to test the model
            oct_device (str): name of the OCT device from which 
                the scans while be used to train and validate 
                the model. Can be 'all', 'Cirrus', 'Spectralis', 
                'T-1000', or 'T-2000'. The default option is 
                'all'
                
        Return:
            None
        """
        # Initiates the Dataset object
        super().__init__()
        # Gets a list of paths to all the images that will be used to 
        # test the model
        self.images_names = generation_images_from_volumes(test_volumes,
                                                           model_name="UNet", 
                                                           oct_device=oct_device)

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in testing

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            (int): size of the dataset
        """
        return len(self.images_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in testing
        when an index is given, utilized to access the list of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            sample (dict{str: PyTorch tensor, str: str}): returns the
                stack of images composed by the previous image, the 
                middle image and the following image, associated with 
                the key 'stack' and the path to the image, associated 
                with the key 'image_name', for the given index 
        """
        # Declares the path to the images
        images_folder = IMAGES_PATH + "\\OCT_images\\generation\\slices_resized\\"

        # Gets the path of the image by the given index
        image_path = images_folder + self.images_names[index]

        # Reads the middle image as a NumPy array
        img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()
        # Gets the number of the slice
        img_number = int(image_path.split(".")[0][-3:])
        # Sets the name of the previous and following images
        prev_img_path = image_path.split(".")[0][:-3] + str(img_number - 1).zfill(3) + ".tiff"
        next_img_path = image_path.split(".")[0][:-3] + str(img_number + 1).zfill(3) + ".tiff"

        # Loads the previous and the 
        # following images 
        prev_img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()
        next_img = torch.from_numpy(((imread(image_path) / 255.) - 0.5) * 2.).float()

        # Organizes the data in a dictionary that contains the images stacked along the first axis under the key 
        # "stack" and the path of the middle image associated with the key "image_name"
        sample = {"stack": torch.stack([prev_img, img, next_img], axis=0), "image_name": self.images_names[index]}

        return sample
