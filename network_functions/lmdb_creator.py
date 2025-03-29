import lmdb
import os
import pandas as pd
from io import BytesIO
from paths import IMAGES_PATH
from PIL import Image
from tqdm.auto import tqdm

def save_to_lmdb(image_list, img_folder, mask_folder, img_lmdb_path, mask_lmdb_path):
        """
        Function used to save the images in the LMDB 
        file

        Args:
            image_list (List[str]): list of all the 
                images, written, for example, as 
                Cirrus_TRAIN001_000.tiff, that are 
                available for the set that is being
                iterated (which can be train or 
                validation)
            img_folder (str): path to the folder 
                where the patches are stored
            mask_folder (str): path to the folder
                where the masks are stored
            img_lmdb_path (str): path and name of 
                the LMDB file that will be created
                with the patches             
            mask_lmdb_path (str): path and name of 
                the LMDB file that will be created
                with the masks 
        
        Returns: 
            None
        """
        # Creates two lmdb environments, one for the images 
        # and one for the masks
        env_img = lmdb.open(img_lmdb_path, map_size=int(1e9))
        env_mask = lmdb.open(mask_lmdb_path, map_size=int(1e9))

        # In each respective environment, files will be written
        with env_img.begin(write=True) as txn_img, env_mask.begin(write=True) as txn_mask:
            # Iterates through name of the files in the set 
            # that is being handled: training or validation
            for i, filename in enumerate(tqdm(image_list, desc=f"Processing {img_lmdb_path}")):
                # Indicates the path to the image and the path to the mask for the file name 
                # that is being iterated
                img_path = os.path.join(img_folder, filename)
                mask_path = os.path.join(mask_folder, filename)

                # Load images with PIL (TIFF format)
                img = Image.open(img_path)
                mask = Image.open(mask_path)

                # Convert to the image to bytes
                img_bytes = BytesIO()
                mask_bytes = BytesIO()
                # Saves each image as TIFF
                img.save(img_bytes, format="TIFF")
                mask.save(mask_bytes, format="TIFF")

                # Stores the image in the respective LMDB file
                txn_img.put(f"img_{i}".encode(), img_bytes.getvalue())
                txn_mask.put(f"mask_{i}".encode(), mask_bytes.getvalue())

        print(f"LMDB dataset saved: {img_lmdb_path} and {mask_lmdb_path}")

def create_lmdb(fold_test: int=1, fold_val: int=2, num_patches: int=13):
    """
    Creates the LMDB file for the training in a significant 
    number of patches. This rises from the high demand in RAM 
    storage required to store the dataset. However, since the
    we are using more than 50,000 images in training when using 
    13 patches, the whole dataset does not fit in RAM. Therefore,
    the training function is accessing the HDD where the data is 
    stored, which is much slower than accessing the RAM. Since 
    the speed of the GPU is faster than accessing the images in 
    the HDD, this results in the bottlenecking of training, that 
    rises the total time of training to days instead of hours. 
    Therefore, the LMDB (Lightning Memory-Mapped Database), 
    solves this issue by providing a linear access (instead of 
    random) to the training data, significantly improving HDD 
    reading speeds. Lastly, this requires a previous creation 
    of files before training. 
    This function creates a LMDB file for the training and 
    validation dataset, given the training an validation folds.
    The number of patches is also necessary to access the folder 
    in which the patches and respective masks are saved.

    Args:
        fold_test (int): number of the fold that is being used 
            in testing
        fold_val (int): number of the fold that is being used
            in validation
        num_patches (int): number of patches that are being 
            used in this train

    Returns:
        None
    """
    # Indicates the path to the folder that stores the images and the masks
    images_folder = os.path.join(IMAGES_PATH, f"OCT_images\\segmentation\\vertical_patches_overlap_{num_patches}")
    masks_folder = os.path.join(IMAGES_PATH, f"OCT_images\\segmentation\\vertical_masks_overlap_{num_patches}")

    # Reads the CSV file that contains the folds partition
    df = pd.read_csv(".\\splits\\competitive_fold_selection.csv")
    
    # Creates a list with the volumes used in training and validation
    train_volumes = df.drop([str(fold_val), str(fold_test)], axis=1).values.flatten().tolist()
    val_volumes = df[str(fold_val)].tolist()

    # Gets the name of all the patches in the folder
    images_filenames = sorted(os.listdir(images_folder))

    # Selects only the names of the images in the training set and validation set saving each result 
    # in a list 
    selected_train_images = [f for f in images_filenames if int(f.split("_")[1][-3:]) in train_volumes]
    selected_val_images = [f for f in images_filenames if int(f.split("_")[1][-3:]) in val_volumes]
    
    # Define LMDB paths and creates the 
    # directory in case it does not exist
    lmdb_dir = ".\\lmdb"
    os.makedirs(lmdb_dir, exist_ok=True)

    # Indicates the path on which the LMDB file will be saved
    train_patches_save_path = os.path.join(lmdb_dir, f"train_patches_{num_patches}_{fold_test}_{fold_val}.lmdb")
    train_masks_save_path = os.path.join(lmdb_dir, f"train_masks_{num_patches}_{fold_test}_{fold_val}.lmdb")
    val_patches_save_path = os.path.join(lmdb_dir, f"val_patches_{num_patches}_{fold_test}_{fold_val}.lmdb")
    val_masks_save_path = os.path.join(lmdb_dir, f"val_masks_{num_patches}_{fold_test}_{fold_val}.lmdb")

    # Calls the function responsible to save the images in the LMDB file
    save_to_lmdb(selected_train_images, images_folder, masks_folder, train_patches_save_path, train_masks_save_path)
    save_to_lmdb(selected_val_images, images_folder, masks_folder, val_patches_save_path, val_masks_save_path)   
