import numpy as np
from os import walk, makedirs
from os.path import isfile, exists
from shutil import rmtree
from PIL import Image
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk, binary_closing
from skimage.filters.rank import entropy
from skimage.transform import resize
from time import time
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from network_functions.dataset import TrainDataset, ValidationDataset, drop_patches
from paths import IMAGES_PATH
from .read_oct import int32_to_uint8

# Declares the multiplication factor to obtain the correct patch height 
# for each device, identified by the height of the image
SHAPE_MULT = {1024: 2., 496: 1., 650: 0.004 / 0.0035, 885: 0.004 / 0.0026}

def extract_patches_wrapper(model_name: str, patch_shape: tuple, n_pos: int,
                             n_neg: int, pos: int, neg: int, train_volumes: list, 
                             val_volumes: list, batch_size: int, 
                             patch_dropping: bool, drop_prob: float):
    """
    Args:
        model_name (str): name of the model desired to train
        patch_shape (tuple): indicates what is the shape of 
            the patches that will be extracted from the B-scans
        n_pos (int): number of patches that will be extracted 
            from the scan ROI
        n_neg (int): number of patches that will be extracted 
            from the outside of the scan ROI 
        pos (int): indicates what is the value that represents 
            the ROI in the ROI mask
        neg (int): indicates what is the value that does not 
            represent the ROI in the ROI mask
        train_volumes (List[float]): list of OCT volumes that will 
            be used to train the model, identified by their 
            index, that ranges from 1 to 70            
        val_volumes (List[int]): list of OCT volumes that will 
            be used to validate the model, identified by their 
            index, that ranges from 1 to 70
        batch_size (int): size of the batch used in 
            training
        patch_dropping (bool): flag that indicates whether patch
            dropping will be used or not
        drop_prob (float): probability of a non-pathogenic patch
            being dropped

    Return:
        train_loader (PyTorch DataLoader object): DataLoader 
            object that contains information about how to load 
            the images that will be used in training                
        val_loader (PyTorch DataLoader object): DataLoader 
            object that contains information about how to load 
            the images that will be used in validation
        n_train (int): number of images that will be used to train 
            the model
    """
    print("Extracting patches")
    # Starts timing the patch extraction
    begin = time()

    # Eliminates the previous patches and saves 
    # new patches to train and validate the model, 
    # but only for the volumes that will be used 
    # in training
    if model_name != "2.5D":
        save_patches_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\slices\\"
        save_patches_masks_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\masks\\"
        save_patches_rois_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\roi\\"

        # In case the folder to save the images does not exist, it is created
        if not (exists(save_patches_path_uint8) 
                and exists(save_patches_masks_path_uint8) 
                and exists(save_patches_rois_path_uint8)):
            makedirs(save_patches_path_uint8)
            makedirs(save_patches_masks_path_uint8)
            makedirs(save_patches_rois_path_uint8)
        else:
            rmtree(save_patches_path_uint8)
            makedirs(save_patches_path_uint8)
            rmtree(save_patches_masks_path_uint8)
            makedirs(save_patches_masks_path_uint8)
            rmtree(save_patches_rois_path_uint8)
            makedirs(save_patches_rois_path_uint8)

        print("Extracting Training Patches")
        extract_patches(IMAGES_PATH, 
                    patch_shape=patch_shape, 
                    n_pos=n_pos, n_neg=n_neg, 
                    pos=pos, neg=neg, 
                    volumes=train_volumes) 
                
        print("Extracting Validation Patches")
        extract_patches(IMAGES_PATH, 
                    patch_shape=patch_shape, 
                    n_pos=n_pos, n_neg=n_neg, 
                    pos=pos, neg=neg, 
                    volumes=val_volumes)
    else:
        save_patches_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\slices\\"
        save_patches_masks_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\masks\\"
        save_patches_rois_path_uint8 = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\roi\\"

        # In case the folder to save the images does not exist, it is created
        if not (exists(save_patches_path_uint8) 
                and exists(save_patches_masks_path_uint8) 
                and exists(save_patches_rois_path_uint8)):
            makedirs(save_patches_path_uint8)
            makedirs(save_patches_masks_path_uint8)
            makedirs(save_patches_rois_path_uint8)
        else:
            rmtree(save_patches_path_uint8)
            makedirs(save_patches_path_uint8)
            rmtree(save_patches_masks_path_uint8)
            makedirs(save_patches_masks_path_uint8)
            rmtree(save_patches_rois_path_uint8)
            makedirs(save_patches_rois_path_uint8)

        print("Extracting Training Patches")
        extract_patches_25D(IMAGES_PATH, 
                    patch_shape=patch_shape, 
                    n_pos=n_pos, n_neg=n_neg, 
                    pos=pos, neg=neg, 
                    volumes=train_volumes)            
        
        print("Extracting Validation Patches")
        extract_patches_25D(IMAGES_PATH, 
                    patch_shape=patch_shape, 
                    n_pos=n_pos, n_neg=n_neg, 
                    pos=pos, neg=neg, 
                    volumes=val_volumes)
    
    # Stops timing the patch extraction and prints it
    end = time()
    print(f"Patch extraction took {end - begin} seconds.")

    if patch_dropping:
        print("Dropping patches")
        # Starts timing the patch dropping
        begin = time()
        # Randomly drops patches of slices that do not have retinal fluid
        drop_patches(prob=drop_prob, volumes_list=train_volumes, model=model_name)
        drop_patches(prob=drop_prob, volumes_list=val_volumes, model=model_name)
        # Stops timing the patch extraction and prints it
        end = time()
        print(f"Patch dropping took {end - begin} seconds.")
    
    # Creates the train and validation Dataset objects
    # The validation dataset does not apply transformations
    train_set = TrainDataset(train_volumes, model_name)
    val_set = ValidationDataset(val_volumes, model_name)

    n_train = len(train_set)
    n_val = len(val_set)
    print(f"Train Images: {n_train} | Validation Images: {n_val}")

    # Using the Dataset object, creates a DataLoader object 
    # which will be used to train the model in batches
    begin = time()
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    print("Loading training data.")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    print("Loading validation data.")
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
    end = time()
    print(f"Data loading took {end - begin} seconds.")

    return train_loader, val_loader, n_train

def create_roi_mask(slice: np.ndarray, mask: np.ndarray, threshold: float, 
                    save_location: str, save_location_to_view: str):
    """
    Creates and saves the ROI mask

    Args:
        slice (NumPy array): contains the B-scan from where the ROI is going to 
            be extracted
        mask (NumPy array): contains the fluid mask of the corresponding B-scan
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

def extract_roi_masks(oct_path: str, folder_path: str, threshold: float):
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

    # Initiates a progress bar that tracks the volumes from which the ROI 
    # mask has been extracted
    with tqdm(total=70, desc=f"ROI_Extraction", unit="vol", leave=True, position=0) as progress_bar:
        # Initiates the variable that will 
        # count the progress of volumes 
        # whose slices are being extracted
        for (root, _, files) in walk(oct_path):
            train_or_test = root.split("-")
            if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
                vendor_volume = train_or_test[2].split("""\\""")
                if len(vendor_volume) == 2:
                    vendor = vendor_volume[0]
                    volume_name = vendor_volume[1]
                    vendor_volume = vendor + "_" + volume_name
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

                        create_roi_mask(OCT_slice, OCT_slice_mask, threshold, save_location=save_name, save_location_to_view=save_name_to_view)

                        slice_num += 1
                        slice_path = volume_path + "_" + str(slice_num).zfill(3) + ".tiff"
                        mask_path = volume_masks_path + "_" + str(slice_num).zfill(3) + ".tiff"
            
                    # Updates the progress bar
                    progress_bar.update(1)

    print("All ROI masks have been extracted.")
    print("EOF.")

def extract_patch_centers_(roi_mask: np.ndarray, patch_shape: tuple, 
                           npos: int, pos: int, neg: int):
    """
    Extract positive patch centers from the ROI mask, inspired by multiple
    functions in nutsml.imageutil

    Args:
        roi_mask (int8 NumPy array): image that contains the ROI
        patch_shape: shape of the resulting patch
        npos: number of positive patches that are going to be
            extracted
        pos: intensity value corresponding to the values that 
            are inside the ROI mask
        neg: intensity value corresponding to the values that 
            are outside the ROI mask

    Returns:
        (List[List[int]]): list of possible centers, which are also lists
            that contain the row, the column, and a label that 
            indicates whether they are a positive or negative patch
    """
    # Begins by considering all the positive points as possible 
    # centers. Transpose is used for easier manipulation
    possible_centers = np.transpose(np.where(roi_mask == pos))
    # Extracts the image dimensions
    h, w = roi_mask.shape[:2]
    # Calculates what are the limits for which the patches 
    # can be extracted
    h2, w2 = patch_shape[0] // 2, patch_shape[1] // 2
    minr, maxr, minc, maxc = h2 - 1, h - h2, w2 - 1, w - w2
    # Conditions the number of centers to those within the
    # possible values
    rs, cs = possible_centers[:, 0], possible_centers[:, 1]
    possible_centers = possible_centers [np.all([rs > minr, rs < maxr, cs > minc, cs < maxc], axis=0)]
    # In case there are not enough points, it 
    # extracts as many centers as possible
    npos = min(npos, possible_centers.shape[0])
    possible_centers = possible_centers[np.random.choice(possible_centers.shape[0], npos, replace=False), :]
    # In case the negative patches are obtained, the label is changed to 0
    if neg > pos:
        label = 0
    # In case the positive patches are obtained, the label is changed to 1
    elif pos > neg:
        label = 1
    # Labels are created
    labels = np.full((possible_centers.shape[0], 1), label, dtype=np.uint8)
    # Labels are added to the output
    possible_centers = np.hstack((possible_centers, labels))
    return possible_centers

def extract_patch_centers(roi_mask: np.ndarray, patch_shape: tuple, 
                          npos: int, pos: int, neg: int):
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
        (List[List[int]]): list of possible positive centers, which 
            are also lists that contain the row, the column, 
            and a label that indicates whether they are a 
            positive or negative patch
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
            # The last 1 means that the patches are positive
            c_hold.append([y, x, 1])

            # After all the number of patches extracted from the image corresponds to 
            # the determined number, no more patches are extracted
        if len(c_hold) >= npos:
            break

    return c_hold

def extract_patches(folder_path: str, patch_shape: tuple, n_pos: int, 
                    n_neg: int, pos: int, neg: int, volumes: list=None):
    """
    Extract the patches from the OCT scans

    Args:
        folders_path (str): Path indicating where the images are stored
        patch_shape (tuple): Shape of the patches to extract (height, width)
        n_pos (int): Number of patches from the ROI to extract
        n_neg (int): Number of patches outside the ROI to extract
        pos (int): Intensity indicating a positive region on the ROI mask
        neg (int): Intensity indicating a negative region on the ROI mask
        volumes (List[float]) optional: List of volumes to extract patches 
            from. The default value is None because it is optional

    Return:
        None
    """

    images_path = folder_path + "\\OCT_images\\segmentation\\slices\\int32\\"
    masks_path = folder_path + "\\OCT_images\\segmentation\\masks\\int8\\"
    ROI_path = folder_path + "\\OCT_images\\segmentation\\roi\\int8\\"
    save_patches_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2D\\slices\\"
    save_patches_masks_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2D\\masks\\"
    save_patches_rois_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2D\\roi\\"

    # Iterates through the saved ROI masks
    for (root, _, files) in walk(images_path):
        for slice in files:
            # Checks if the volume is in the list of 
            # patches to extract, in case there is a list
            volume = int(slice.split("_")[1][-3:])
            if (volumes is None) or (volume in volumes):
                # Reads the masks, the slices, 
                # and the fluid masks 
                slice_path = root + slice
                ROI_mask_path = ROI_path + slice
                mask_path = masks_path + slice 
                slice = imread(slice_path)
                roi = imread(ROI_mask_path)
                mask = imread(mask_path)
                # Adjusts the height of the mask 
                # to the one device used to obtain the OCT
                img_height = slice.shape[0]
                npshape = (int(patch_shape[0] * SHAPE_MULT[img_height]), patch_shape[1])
                # Extracts positive patch centers through two different functions
                patch_centers = extract_patch_centers(roi_mask=roi, patch_shape=npshape, npos=n_pos-int(float(n_pos)*.2), pos=pos, neg=neg)
                patch_centers_ = extract_patch_centers_(roi_mask=roi, patch_shape=npshape, npos=int(float(n_pos)*.2), pos=pos, neg=neg)
                # Appends the second patch centers to the first
                for r, c, l in patch_centers_:
                    patch_centers.append([r, c, l])

                # Extracts negative patch centers
                if n_neg > 0:
                    negative_patch_centers = extract_patch_centers_(roi_mask=roi, patch_shape=npshape, npos=n_neg, pos=neg, neg=pos)
                    for r, c, l in negative_patch_centers:
                        # Appends the negative patch centers to the others
                        patch_centers.append([r, c, l])

                # Iterates through the calculated centers
                # and extracts the patches
                pos_patch_counter = 0
                neg_patch_counter = 0
                for r, c, l in patch_centers:
                    # Calculates the patchesfor the B-scan, ROI, and fluid masks
                    h, w = patch_shape[0], patch_shape[1]
                    r, c = int(r - h // 2), int(c - w // 2)
                    tmp_slice = slice[r:r + h, c:c + w]
                    tmp_roi = roi[r:r + h, c:c + w]
                    tmp_mask = mask[r:r + h, c:c + w]

                    # Attributes a number to each patch
                    # considering their label
                    if l == 1:
                        label = "pos"
                        patch_counter = pos_patch_counter
                        pos_patch_counter += 1
                    elif l == 0:
                        label = "neg"
                        patch_counter = neg_patch_counter
                        neg_patch_counter += 1

                    # Indicates the name of the patches
                    vol_name = slice_path.split("\\")[-1][:-5]
                    patch_name = vol_name + "_" + label + "_patch_" + str(patch_counter).zfill(2) + ".tiff"

                    # Indicates the name of the slice patch
                    slice_patch_name_uint8 = save_patches_path_uint8 + patch_name

                    # Indicates the name of the mask patch
                    mask_patch_name_uint8 = save_patches_masks_path_uint8 + patch_name

                    # Indicates the name of the ROI patch
                    roi_patch_name_uint8 = save_patches_rois_path_uint8 + patch_name
                    
                    # Saves each slice patch as uint8 after resizing it to match the patch shape
                    tmp_slice = resize(tmp_slice.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    slice_uint8 = Image.fromarray(tmp_slice)
                    slice_uint8.save(slice_patch_name_uint8)

                    # Saves each mask patch as uint8 after resizing it to match the patch shape
                    tmp_mask = resize(tmp_mask.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    mask_uint8 = Image.fromarray(tmp_mask)
                    mask_uint8.save(mask_patch_name_uint8)

                    # Saves each ROI patch as uint8 after resizing it to match the patch shape
                    tmp_roi = resize(tmp_roi.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    roi_uint8 = Image.fromarray(tmp_roi)
                    roi_uint8.save(roi_patch_name_uint8)

def extract_patches_25D(folder_path: str, patch_shape: tuple, n_pos: int, 
                        n_neg: int, pos: int, neg: int, volumes: list=None):
    """
    Extract the subvolumes of patches from the OCT scans

    Args:
        folders_path (str): Path indicating where the images are stored
        patch_shape (tuple): Shape of the patches to extract (height, width)
        n_pos (int): Number of patches from the ROI to extract
        n_neg (int): Number of patches outside the ROI to extract
        pos (int): Intensity indicating a positive region on the ROI mask
        neg (int): Intensity indicating a negative region on the ROI mask
        volumes (List[float]) optional: List of volumes to extract patches 
            from. The default value is None because it is optional

    Return:
        None
    """

    images_path = folder_path + "\\OCT_images\\segmentation\\slices\\int32\\"
    masks_path = folder_path + "\\OCT_images\\segmentation\\masks\\int8\\"
    ROI_path = folder_path + "\\OCT_images\\segmentation\\roi\\int8\\"
    save_patches_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2.5D\\slices\\"
    save_patches_masks_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2.5D\\masks\\"
    save_patches_rois_path_uint8 = folder_path + "\\OCT_images\\segmentation\\patches\\2.5D\\roi\\"

    # Iterates through the saved ROI masks
    for (root, _, files) in walk(images_path):
        for slice in files:
            # Checks if the volume is in the list of 
            # patches to extract, in case there is a list
            volume = int(slice.split("_")[1][-3:])
            if (volumes is None) or (volume in volumes):
                # Reads the masks, the slices, 
                # and the fluid masks 
                slice_path = root + slice
                ROI_mask_path = ROI_path + slice
                mask_path = masks_path + slice 
                slice_number = int(slice_path[-8:-5])
                # In case it is the first slice of the volume, the first
                # and middle slice of the subvolume are the same
                if slice_number == 0:
                    slice_before_path = slice_path
                    roi_before_path = ROI_mask_path
                    mask_before_path = mask_path
                else:
                    slice_before_path = slice_path[:-8] + str(slice_number - 1).zfill(3) + ".tiff"
                    roi_before_path = ROI_mask_path[:-8] + str(slice_number - 1).zfill(3) + ".tiff"
                    mask_before_path = mask_path[:-8] + str(slice_number - 1).zfill(3) + ".tiff"

                slice_after_path = slice_path[:-8] + str(slice_number + 1).zfill(3) + ".tiff"
                roi_after_path = ROI_mask_path[:-8] + str(slice_number + 1).zfill(3) + ".tiff"
                mask_after_path = mask_path[:-8] + str(slice_number + 1).zfill(3) + ".tiff"

                # In case it is the last slice of the volume, the middle
                # and last slice of the subvolume are the same
                if not isfile(slice_after_path):
                    slice_after_path = slice_path
                    roi_after_path = ROI_mask_path
                    mask_after_path = mask_path

                # Subvolumes containing the previous, current, and 
                # following slices are created
                slice_before = imread(slice_before_path)
                slice = imread(slice_path)
                slice_after = imread(slice_after_path)
                slices_subvolumes = np.stack([slice_before, slice, slice_after], axis=-1)

                roi_before = imread(roi_before_path)
                roi = imread(ROI_mask_path)
                roi_after = imread(roi_after_path)
                rois_subvolumes = np.stack([roi_before, roi, roi_after], axis=-1)

                mask_before = imread(mask_before_path)
                mask = imread(mask_path)
                mask_after = imread(mask_after_path)
                masks_subvolumes = np.stack([mask_before, mask, mask_after], axis=-1)

                # Adjusts the height of the mask 
                # to the one device used to obtain the OCT
                img_height = slice.shape[0]
                npshape = (int(patch_shape[0] * SHAPE_MULT[img_height]), patch_shape[1])
                # Extracts positive patch centers through two different functions
                patch_centers = extract_patch_centers(roi_mask=roi, patch_shape=npshape, npos=n_pos-int(float(n_pos)*.2), pos=pos, neg=neg)
                patch_centers_ = extract_patch_centers_(roi_mask=roi, patch_shape=npshape, npos=int(float(n_pos)*.2), pos=pos, neg=neg)
                # Appends the second patch centers to the first
                for r, c, l in patch_centers_:
                    patch_centers.append([r, c, l])

                # Extracts negative patch centers
                if n_neg > 0:
                    negative_patch_centers = extract_patch_centers_(roi_mask=roi, patch_shape=npshape, npos=n_neg, pos=neg, neg=pos)
                    for r, c, l in negative_patch_centers:
                        # Appends the negative patch centers to the others
                        patch_centers.append([r, c, l])

                # Iterates through the calculated centers
                # and extracts the patches
                pos_patch_counter = 0
                neg_patch_counter = 0
                for r, c, l in patch_centers:
                    # Calculates the patchesfor the B-scan, ROI, and fluid masks
                    h, w = patch_shape[0], patch_shape[1]
                    r, c = int(r - h // 2), int(c - w // 2)
                    tmp_slice = slices_subvolumes[r:r + h, c:c + w]
                    tmp_roi = rois_subvolumes[r:r + h, c:c + w]
                    tmp_mask = masks_subvolumes[r:r + h, c:c + w]

                    # Attributes a number to each patch
                    # considering their label
                    if l == 1:
                        label = "pos"
                        patch_counter = pos_patch_counter
                        pos_patch_counter += 1
                    elif l == 0:
                        label = "neg"
                        patch_counter = neg_patch_counter
                        neg_patch_counter += 1

                    # Indicates the name of the patches
                    vol_name = slice_path.split("\\")[-1][:-5]
                    before_patch_name = vol_name + "_" + label + "_patch_" + str(patch_counter).zfill(2) + "_before.tiff"
                    patch_name = vol_name + "_" + label + "_patch_" + str(patch_counter).zfill(2) + ".tiff"
                    after_patch_name = vol_name + "_" + label + "_patch_" + str(patch_counter).zfill(2) + "_after.tiff"

                    # Indicates the name of the slice patch after resizing it to match the patch shape
                    slice_before_patch_name_uint8 = save_patches_path_uint8 + before_patch_name
                    slice_patch_name_uint8 = save_patches_path_uint8 + patch_name
                    slice_after_patch_name_uint8 = save_patches_path_uint8 + after_patch_name

                    # Indicates the name of the mask patch after resizing it to match the patch shape
                    mask_before_patch_name_uint8 = save_patches_masks_path_uint8 + before_patch_name
                    mask_patch_name_uint8 = save_patches_masks_path_uint8 + patch_name
                    mask_after_patch_name_uint8 = save_patches_masks_path_uint8 + after_patch_name

                    # Indicates the name of the ROI patch after resizing it to match the patch shape
                    roi_before_patch_name_uint8 = save_patches_rois_path_uint8 + before_patch_name
                    roi_patch_name_uint8 = save_patches_rois_path_uint8 + patch_name
                    roi_after_patch_name_uint8 = save_patches_rois_path_uint8 + after_patch_name
                    
                    # Saves each slice patch as uint8
                    tmp_slice = resize(tmp_slice.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    slice_before_uint8 = Image.fromarray(int32_to_uint8(tmp_slice[:,:,0]))
                    slice_before_uint8.save(slice_before_patch_name_uint8)
                    slice_uint8 = Image.fromarray(int32_to_uint8(tmp_slice[:,:,1]))
                    slice_uint8.save(slice_patch_name_uint8)
                    slice_after_uint8 = Image.fromarray(int32_to_uint8(tmp_slice[:,:,2]))
                    slice_after_uint8.save(slice_after_patch_name_uint8)

                    # Saves each mask patch as uint8
                    tmp_mask = resize(tmp_mask.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    mask_before_uint8 = Image.fromarray(tmp_mask[:,:,0].astype(np.uint8))
                    mask_before_uint8.save(mask_before_patch_name_uint8)
                    mask_uint8 = Image.fromarray(tmp_mask[:,:,1].astype(np.uint8))
                    mask_uint8.save(mask_patch_name_uint8)
                    mask_after_uint8 = Image.fromarray(tmp_mask[:,:,2].astype(np.uint8))
                    mask_after_uint8.save(mask_after_patch_name_uint8)

                    # Saves each ROI patch as uint8
                    tmp_roi = resize(tmp_roi.astype(np.uint8), patch_shape, order=0, preserve_range=True).astype('uint8')
                    roi_before_uint8 = Image.fromarray(tmp_roi[:,:,0].astype(np.uint8))
                    roi_before_uint8.save(roi_before_patch_name_uint8)
                    roi_uint8 = Image.fromarray(tmp_roi[:,:,1].astype(np.uint8))
                    roi_uint8.save(roi_patch_name_uint8)
                    roi_after_uint8 = Image.fromarray(tmp_roi[:,:,2].astype(np.uint8))
                    roi_after_uint8.save(roi_after_patch_name_uint8)
