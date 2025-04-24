import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from IPython import get_ipython
from os import makedirs
from os.path import exists
from pandas import DataFrame, read_csv, Series
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from shutil import rmtree
from init.read_oct import load_oct_image
from networks.loss import dice_coefficient
from networks.unet25D import TennakoonUNet
from networks.unet import UNet
from network_functions.dataset import TestDataset
from paths import IMAGES_PATH, RETOUCH_PATH

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

# Dictionary of fluid to labels in masks
fluids_to_label = {
    "IRF": 1,
    "SRF": 2,
    "PED": 3
}  

# Dictionary of labels in masks to fluid names
label_to_fluids = {
    0: "Background",
    1: "IRF",
    2: "SRF",
    3: "PED"
}

def pad_to_nearest_16(image):
    """
    If patches were not used, the images will be 
    handled with their original shape. This might
    result in shape's mismatch during inference 
    since the max-pooling might result in 
    features with shapes different than the ones 
    resulting in upsampling due to rounding errors. 
    Therefore, the image is padded to ensure that
    it is a multiple of sixteen (four max-pooling 
    operations are performed in the encoding path
    thus if the dimensions are divisible by 2^4, 
    no shape mismatch will happen)

    Args:
        image (PyTorch Tensor): PyTorch Tensor 
            that contains is going to be handled

    Returns:
        (PyTorch Tensor): processed image  
    """
    # Gets the height and width of the image
    h, w = image.shape[-2:]
    # Calculates the necessary padding
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    # Returns the padded image
    return pad(image, (0, pad_w, 0, pad_h))

def collate_fn(batch):
    """
    This function is used when getting the images in the DataLoader, 
    since it requires a function that handles inputs of different 
    shapes (the default collate function does not) 

    Args:
        batch (List[Dict[str, NumPy array | str]]): List that 
            contains the name of the object in the dictionary, 
            and the respective image, scan, or name
    Returns:
        None 
    """
    # Gets the information of the batch
    sample = batch[0]
    
    # Gets the scan and the mask from the batch
    scan = sample['scan']
    mask = sample['mask']
    
    # Ensures that the scan and mask are tensors
    if not isinstance(scan, torch.Tensor):
        scan = torch.tensor(scan)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    # Add batch dimension back
    scan = scan.unsqueeze(0)  # Shape: (1, H, W, C)
    mask = mask.unsqueeze(0)  # Shape: (1, H, W)
    
    scan = scan.permute(0, 3, 1, 2)

    scan = pad_to_nearest_16(scan)
    mask = pad_to_nearest_16(mask)

    scan = scan.permute(0, 2, 3, 1)

    return {'scan': scan, 'mask': mask, 'image_name': sample['image_name']}


def folds_results(first_run_name: str, iteration: int, k: int=5, 
                  resized_images: bool=False, 
                  binary_segmentation: bool=False):
    """
    Function used to compare the results obtained in the k folds,
    calculating the mean and standard deviation of the results 
    obtained for each vendor and class, from the files output 
    from the test function

    Args:
        first_run_name (str): name of the first run of the folds 
            considered. It is expected that the name of the first 
            run is something like "Run001" and the runs of the 
            same iteration but different folds increment one to 
            run number thus being named "Run002", "Run003", and
            "Run004", for k=5, for example
        iteration (int): number of the iteration that comprises 
            the k - 1 runs
        k (int): number of folds used in this iteration
        resized_images (bool): flag that indicates whether the 
            images were resized or not
        binary_segmentation (bool): flag that indicates that the 
            models were trained for a binary segmentation task.
            This is relevant because the name of the files are 
            not the same as in the other runs. Since most runs
            are in fact multi-class, the default value of this
            argument is False

    Return: 
        None
    """
    # Initiates a dictionary that will store 
    # the DataFrames from the different runs
    df_dict = {}
    # Gets the number of the first run
    first_run_index = int(first_run_name[3:])
    # Iterates through the runs corresponding to the folds
    # Starts in the index of the first run and stops k - 1 
    # integers after
    for fold in range(first_run_index, k + first_run_index - 1):
        # Gets the name of the run from the fold number
        # e.g. fold=3 -> run_name="Run003"
        run_name = "Run" + str(fold).zfill(3)
        if not resized_images:
            # Indicates the name of the file that stores the Dice 
            # per class
            class_file_name = f".\\results\\{run_name}_class_dice.csv"
            # Indicates the name of the file that stores the Dice 
            # per vendor
            vendor_file_name = f".\\results\\{run_name}_vendor_dice.csv"            
            # Indicates the name of the file that stores the Dice 
            # in binarized fluids
            fluid_file_name = f".\\results\\{run_name}_fluid_dice.csv"            
        else:
            # Indicates the name of the file that stores the Dice 
            # per class
            class_file_name = f".\\results\\{run_name}_class_dice_resized.csv"
            # Indicates the name of the file that stores the Dice 
            # per vendor
            vendor_file_name = f".\\results\\{run_name}_vendor_dice_resized.csv"
            # Indicates the name of the file that stores the Dice 
            # in binarized fluids
            fluid_file_name = f".\\results\\{run_name}_fluid_dice_resized.csv"
        # Reads the DataFrame that handles the data per class
        class_df = read_csv(class_file_name)
        # Reads the DataFrame that handles the data per vendor
        vendor_df = read_csv(vendor_file_name, index_col="Vendor")
        if not binary_segmentation:
            # Reads the DataFrame that handles the data per class
            class_df_wf = read_csv(class_file_name[:-4] + "_wfluid.csv")
            # Reads the DataFrame that handles the data per class
            class_df_wof = read_csv(class_file_name[:-4] + "_wofluid.csv")
            # Reads the DataFrame that handles 
            # the overall fluid
            fluid_df = read_csv(fluid_file_name)
            fluid_file_middle_name = "fluids"
        else:
            fluid_df = class_df
            fluid_file_middle_name = "classes"
        # Reads the DataFrame that handles the data per vendor
        vendor_df_wf = read_csv(vendor_file_name[:-4] + "_wfluid.csv", index_col="Vendor")        
        # Reads the DataFrame that handles the data per vendor
        vendor_df_wof = read_csv(vendor_file_name[:-4] + "_wofluid.csv", index_col="Vendor")
        # Removes the name of the column that has the table's index
        vendor_df.index.name = None
        vendor_df_wf.index.name = None
        vendor_df_wof.index.name = None

        # Saves, to the corresponding fold in the 
        # dictionary, the two DataFrames as a 
        # single tuple
        if not binary_segmentation:
            df_dict[fold] = (vendor_df, vendor_df_wf, vendor_df_wof, class_df, class_df_wf, class_df_wof, fluid_df)
        else:
            df_dict[fold] = (vendor_df, vendor_df_wf, vendor_df_wof, fluid_df)
    # Gets the list of vendors and fluids
    vendors = vendor_df.index.to_list()
    fluids = vendor_df.columns.to_list()

    # Initiates the DataFrame with the name 
    # of the fluids as the columns names for 
    # the vendor data
    # e.g. of a column name: Dice_IRF
    vendor_df = DataFrame(columns=fluids)
    vendor_df_wf = DataFrame(columns=fluids)
    vendor_df_wof = DataFrame(columns=fluids)
    # Iterates through the 
    # vendors in the DataFrame
    # (rows)
    for vendor in vendors:
        # In each vendor stores an array of values 
        # that will later be inserted as a row
        values = []
        values_wf = []
        values_wof = []
        # Iterates through the fluids
        for fluid in fluids:
            # Initiates a list that will store the 
            # results across all folds
            results = []
            results_wf = []
            results_wof = []
            # Iterates across all folds in the dictionary
            for fold, tuple_df in df_dict.items():
                # Appends to the results list, the value at 
                # the current vendor and fluid
                # The DataFrame that is being handled is the 
                # one that comprises information about classes
                # and vendors
                results.append(float(tuple_df[0].at[vendor, fluid].split(" ")[0]))
                results_wf.append(float(tuple_df[1].at[vendor, fluid].split(" ")[0]))
                results_wof.append(float(tuple_df[2].at[vendor, fluid].split(" ")[0]))
            # Calculates the mean across all folds
            mean = np.array(results).mean()
            mean_wf = np.array(results_wf).mean()
            mean_wof = np.array(results_wof).mean()
            # Calculates the standard deviation across all folds
            std = np.array(results).std()
            std_wf = np.array(results_wf).std()
            std_wof = np.array(results_wof).std()
            # Saves the value as "mean (std)"
            value = f"{mean:.2f} ({std:.2f})"
            value_wf = f"{mean_wf:.2f} ({std_wf:.2f})"
            value_wof = f"{mean_wof:.2f} ({std_wof:.2f})"
            # Appends the value to the list that will form a row
            values.append(value)
            values_wf.append(value_wf)
            values_wof.append(value_wof)
        # Appends the results in a row to the DataFrame
        vendor_df.loc[len(vendor_df)] = values
        vendor_df_wf.loc[len(vendor_df_wf)] = values_wf
        vendor_df_wof.loc[len(vendor_df_wf)] = values_wof
    # Sets the name of the axis in the 
    # DataFrame to the name of the vendors
    vendor_df = vendor_df.set_axis(vendors)
    vendor_df = vendor_df.rename_axis("vendors")
    vendor_df_wf = vendor_df_wf.set_axis(vendors)
    vendor_df_wf = vendor_df_wf.rename_axis("vendors") 
    vendor_df_wof = vendor_df_wof.set_axis(vendors)
    vendor_df_wof = vendor_df_wof.rename_axis("vendors")
    # Saves the DataFrame with a name refering to the iteration
    if not resized_images:
        vendor_df.to_csv(f".\\results\\Iteration{iteration}_vendors_results.csv")
        vendor_df_wf.to_csv(f".\\results\\Iteration{iteration}_vendors_results_wfluid.csv")
        vendor_df_wof.to_csv(f".\\results\\Iteration{iteration}_vendors_results_wofluid.csv")
    else:
        vendor_df.to_csv(f".\\results\\Iteration{iteration}_vendors_results_resized.csv")
        vendor_df_wf.to_csv(f".\\results\\Iteration{iteration}_vendors_results_resized_wfluid.csv")
        vendor_df_wof.to_csv(f".\\results\\Iteration{iteration}_vendors_results_resized_wofluid.csv")

    if not binary_segmentation:
        # Initiates the DataFrame with the name 
        # of the fluids as the columns names for 
        # the class data
        # e.g. of a column name: Dice_IRF
        class_df = DataFrame(columns=fluids)
        class_df_wf = DataFrame(columns=fluids)
        class_df_wof = DataFrame(columns=fluids)
        # Initiates a list that will store the 
        # values of the classes 
        values = []
        values_wf = []
        values_wof = []
        # Iterates through the fluids
        for fluid in fluids:
            # Initiates a list that will store the 
            # results across all folds
            results = []
            results_wf = []
            results_wof = []
            # Iterates across all folds in the dictionary
            for fold, tuple_df in df_dict.items():
                # Appends to the results list, the value at 
                # the current vendor and fluid
                # The DataFrame that is now being handled 
                # comprises information of the classes
                results.append(float(tuple_df[3].at[0, fluid].split(" ")[0]))
                results_wf.append(float(tuple_df[4].at[0, fluid].split(" ")[0]))
                results_wof.append(float(tuple_df[5].at[0, fluid].split(" ")[0]))
            # Calculates the mean across all folds
            mean = np.array(results).mean()
            mean_wf = np.array(results_wf).mean()
            mean_wof = np.array(results_wof).mean()
            # Calculates the standard deviation across all folds
            std = np.array(results).std()
            std_wf = np.array(results_wf).std()
            std_wof = np.array(results_wof).std()
            # Saves the value as "mean (std)"
            value = f"{mean:.2f} ({std:.2f})"
            value_wf = f"{mean_wf:.2f} ({std_wf:.2f})"
            value_wof = f"{mean_wof:.2f} ({std_wof:.2f})"
            # Appends the value to the list that will form a row
            values.append(value)
            values_wf.append(value_wf)
            values_wof.append(value_wof)
        # Appends the results in a row to the DataFrame
        class_df.loc[len(class_df)] = values
        class_df_wf.loc[len(class_df_wf)] = values_wf
        class_df_wof.loc[len(class_df_wof)] = values_wof
        # Saves the DataFrame with a name refering to the iteration, not including the index
        if not resized_images:
            class_df.to_csv(f".\\results\\Iteration{iteration}_classes_results.csv", index=False)
            class_df_wf.to_csv(f".\\results\\Iteration{iteration}_classes_results_wfluid.csv", index=False)
            class_df_wof.to_csv(f".\\results\\Iteration{iteration}_classes_results_wofluid.csv", index=False)
        else:
            class_df.to_csv(f".\\results\\Iteration{iteration}_classes_results_resized.csv", index=False)
            class_df_wf.to_csv(f".\\results\\Iteration{iteration}_classes_results_resized_wfluid.csv", index=False)
            class_df_wof.to_csv(f".\\results\\Iteration{iteration}_classes_results_resized_wofluid.csv", index=False)

    # Initiates the DataFrame with the name of the columns
    fluid_df = DataFrame(columns=["AllSlices", "SlicesWithFluid", "SlicesWithoutFluid"])

    # Initiates the lists that will hold 
    # the values for the following files
    values = []
    results = []
    results_wf = []
    results_wof = []

    # Iterates through the different folds
    for fold, tuple_df in df_dict.items():
        # Appends to the results list, the value at 
        # the current vendor and fluid
        # The DataFrame that is being handled is the 
        # one that comprises information about classes
        # and vendors
        results.append(float(tuple_df[-1].iat[0, 0].split(" ")[0]))
        results_wf.append(float(tuple_df[-1].iat[0, 1].split(" ")[0]))
        results_wof.append(float(tuple_df[-1].iat[0, 2].split(" ")[0]))

    # Calculates the mean across all folds
    mean = np.array(results).mean()
    mean_wf = np.array(results_wf).mean()
    mean_wof = np.array(results_wof).mean()
    # Calculates the standard deviation across all folds
    std = np.array(results).std()
    std_wf = np.array(results_wf).std()
    std_wof = np.array(results_wof).std()
    # Saves the value as "mean (std)"
    value = f"{mean:.2f} ({std:.2f})"
    value_wf = f"{mean_wf:.2f} ({std_wf:.2f})"
    value_wof = f"{mean_wof:.2f} ({std_wof:.2f})"
    # Appends the value to the list that will form a row
    values.append(value)
    values.append(value_wf)
    values.append(value_wof)
    # Appends the results in a row to the DataFrame
    fluid_df.loc[len(fluid_df)] = values

    # Saves the new DataFrame as a CSV file
    if not resized_images:
        fluid_df.to_csv(f".\\results\\Iteration{iteration}_{fluid_file_middle_name}_results.csv", index=False)
    else:
        fluid_df.to_csv(f".\\results\\Iteration{iteration}_{fluid_file_middle_name}_results_resized.csv", index=False)

def test_model (
        fold_test: int,
        model_name: str,
        weights_name: str,
        batch_size: int=1,
        device_name: str="GPU",
        final_test: bool=False,
        fluid: str=None,
        number_of_channels: int=1,
        number_of_classes: int=4,
        patch_type: str="vertical",
        resize_images: bool=False,
        save_images: bool=True,
        split: str="competitive_fold_selection.csv",
    ):
    """
    Function used to test the trained models

    Args:
        fold_test (int): number of the fold that will be used 
            in the network testing 
        model_name (str): name of the model that will be 
            evaluated         
        weights_name (str): path to the model's weight file
        batch_size (int): size of the batch used in testing
        device_name (str): indicates whether the network will 
            be trained using the CPU or the GPU
        final_test (bool): indicates that the final test will be 
            performed, changing the name of the saved files. Since final
            testing is rare, the default value is False 
        fluid (str): name of the fluid that is desired to segment
        number_of_channels (int): number of channels the 
            input will present
        number_of_classes (int): number of classes the 
            output will present
        patch_type (str): string that indicates what type of patches 
            will be used. Can be "small", where patches of size 
            256x128 are extracted using the extract_patches function,
            "big", where patches of shape 496x512 are extracted from 
            each image, and patches of shape 496x128 are extracted from
            the slices
        resize_images (bool): flag that indicates whether the images 
            will be resized or not in testing 
        save_images (bool): flag that indicates whether the 
            predicted images will be saved or not
        split (str): name of the k-fold split file that will be 
            used in this run
            
    Return:
        None
    """
    # Dictionary of models, associates a string to a PyTorch module
    models = {
        "UNet": UNet(in_channels=number_of_channels, num_classes=number_of_classes),
        "UNet3": UNet(in_channels=number_of_channels, num_classes=number_of_classes),
        "2.5D": TennakoonUNet(in_channels=number_of_channels, num_classes=number_of_classes)
    }
    
    # Checks whether the selected model exists or not
    if model_name not in models.keys():
        print("Model not recognized. Possible models:")
        for key in models.keys():
            print(key)
        return 0
    
    # Checks if the weight file corresponds to the model selected
    if model_name not in weights_name.split("_"):
        print("Model name and weights name do not match.")
        return 0
    
    # Declares the path to the weights
    weights_path = "models\\" + weights_name 

    # Gets the list of volumes used to test the model
    df = read_csv(f"splits/{split}")
    test_volumes = df[str(fold_test)].dropna().to_list()

    # Checks if the declared device exists
    if device_name not in ["CPU", "GPU"]:
        print("Unrecognized device. Possible devices:")
        print("CPU")
        print("GPU")
    elif (device_name == "GPU"):
        # Checks whether the GPU is available 
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            print("GPU is not available. CPU was selected.")
    elif (device_name=="CPU"):
        device_name = "cpu"
    # Saves the variable device as torch.device 
    device = torch.device(device_name)
    
    # Gets the selected model and assigns the device to it
    model = models.get(model_name)
    model = model.to(device=device, memory_format=torch.channels_last)

    # Loads the trained model and informs the model about evaluation mode
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()

    # Initiates resize shape for the cases where 
    # resize_images is False to not raise the error 
    # of calling an unassigned variable 
    resize_shape = (0,0)
    # Saves the desired output shape from the resizing
    if resize_images:
        # Loads a Spectralis file to check what is the patch size desired
        spectralis_path = RETOUCH_PATH + "\RETOUCH-TrainingSet-Spectralis\TRAIN025\oct.mhd"
        img, _, _ = load_oct_image(spectralis_path)
        # Saves the desired shape as a tuple
        resize_shape = (img.shape[1], img.shape[2])

    # Creates the TestDataset and DataLoader object with the test volumes
    # Number of workers was set to the most optimal
    test_dataset = TestDataset(test_volumes, model_name, patch_type, resize_images, resize_shape, fluids_to_label.get(fluid), number_of_channels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, collate_fn=collate_fn)
    
    # Initiates the list that will 
    # store the results of the slices 
    slice_results = []

    # Extracts the name of the run from 
    # the name of the weights file
    run_name = weights_name.split("_")[0]

    # Declares the name of the folder in which the images will be saved
    if save_images:
        if not resize_images:
            if final_test: 
                folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}_final\\"
            # In case the folder to save exists, it is deleted and created again
            else:
                folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}\\"
            if exists(folder_to_save):
                rmtree(folder_to_save)
                makedirs(folder_to_save)
            else:
                makedirs(folder_to_save)
        else:
            if final_test:
                folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}_resized_final\\"
            else:
                folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}_resized\\"
            # In case the folder to save exists, it is deleted and created again
            if exists(folder_to_save):
                rmtree(folder_to_save)
                makedirs(folder_to_save)
            else:
                makedirs(folder_to_save)
    
    # Informs that no backward propagation will be calculated 
    # because it is an inference, thus reducing memory consumption
    with torch.no_grad():
        # Creates a progress bar to track the progress on testing images
        with tqdm(test_dataloader, total=len(test_dataloader), desc='Testing Model', unit='img', leave=True, position=0) as progress_bar:
            # Iterates through every batch and path 
            # (that compose the batch) in the dataloader
            # In this case, the batches are of size one, 
            # so every batch is handled like a single image
            for batch in test_dataloader:
                # Gets the images and the masks from the DataLoader
                images, true_masks, image_name = batch['scan'], batch['mask'], batch['image_name']

                # Handles the images and masks according to the device, specified data type and memory format
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Converts the image from channels 
                # last to channels first
                images = images.permute(0, 3, 1, 2)

                # Predicts the output of the batch
                outputs = model(images)

                # The prediction is assumed as the value 
                # that has a higher logit
                preds = torch.argmax(outputs, dim=1)
                
                # Calculates the Dice coefficient of the predicted mask, and gets the result of the union and intersection between the GT and 
                # the predicted mask 
                dice_scores, voxel_counts, union_counts, intersection_counts, binary_dice = dice_coefficient(model_name, preds, true_masks, number_of_classes)

                # Appends the results per slice, per volume, and per vendor
                slice_results.append([image_name, *dice_scores, *voxel_counts, *union_counts, *intersection_counts, binary_dice])

                # Saves the predicted masks and the GT, in case it is desired
                if save_images:
                    # Declares the name under which the masks will be saved and writes the path to the original B-scan
                    predicted_mask_name = folder_to_save + image_name[:-5] + "_predicted" + ".tiff"
                    gt_mask_name = folder_to_save + image_name[:-5] + "_gt" + ".tiff"

                    # Gets the original OCT B-scan
                    oct_image = images[0].cpu().numpy()[0]

                    # Converts each voxel classified as background to 
                    # NaN so that it will not appear in the overlaying
                    # mask
                    preds = np.array(preds.cpu().numpy(), dtype=np.float32)[0]
                    preds[preds == 0] = np.nan

                    true_masks = np.array(true_masks.cpu().numpy(), dtype=np.float32)[0]
                    true_masks[true_masks == 0] = np.nan

                    # Converts the mask from 
                    # binary to the correct class 
                    if fluid is not None:
                        preds = preds * int(fluids_to_label.get(fluid))
                        true_masks = true_masks * int(fluids_to_label.get(fluid))

                    # The voxels classified in "IRF", "SRF", and "PED" 
                    # will be converted to color as Red for IRF, green 
                    # for SRF, and blue for PED
                    fluid_colors = ["red", "green", "blue"]
                    fluid_cmap = mcolors.ListedColormap(fluid_colors)
                    # Declares in which part of the color bar each
                    # label is going to be placed
                    fluid_bounds = [1, 2, 3, 4]
                    # Normalizes the color map according to the 
                    # bounds declared.
                    fluid_norm = mcolors.BoundaryNorm(fluid_bounds, fluid_cmap.N)

                    # Saves the OCT scan with an overlay of the predicted masks
                    plt.figure(figsize=(oct_image.shape[1] / 100, oct_image.shape[0] / 100))
                    plt.imshow(oct_image, cmap=plt.cm.gray)
                    plt.imshow(preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                    plt.axis("off")
                    plt.savefig(predicted_mask_name, bbox_inches='tight', pad_inches=0)

                    # Saves the OCT scan with an overlay of the ground-truth masks
                    plt.figure(figsize=(oct_image.shape[1] / 100, oct_image.shape[0] / 100))
                    plt.imshow(oct_image, cmap=plt.cm.gray)
                    plt.imshow(true_masks, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                    plt.axis("off")
                    plt.savefig(gt_mask_name, bbox_inches="tight", pad_inches=0)

                    # Closes the figure
                    plt.clf()
                    plt.close("all")

                # Update the progress bar
                progress_bar.update(1)

    # Creates the folder results in case 
    # it does not exist yet
    makedirs("results", exist_ok=True)

    # Saves the Dice score per slice
    if model_name != "UNet3":
        slice_df = DataFrame(slice_results, 
                            columns=["slice", 
                                    *[f"dice_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                    *[f"voxels_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                    *[f"union_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                    *[f"intersection_{label_to_fluids.get(i)}" for i in range(number_of_classes)],
                                    "binary_dice"])
    else:
        slice_df = DataFrame(slice_results, 
                    columns=["slice", 
                            *[f"dice_{label_to_fluids.get(i)}" for i in [0, fluids_to_label.get(fluid)]], 
                            *[f"voxels_{label_to_fluids.get(i)}" for i in [0, fluids_to_label.get(fluid)]], 
                            *[f"union_{label_to_fluids.get(i)}" for i in [0, fluids_to_label.get(fluid)]], 
                            *[f"intersection_{label_to_fluids.get(i)}" for i in [0, fluids_to_label.get(fluid)]],
                            "binary_dice"])
    
    # Declares the name under which the DataFrame will be saved 
    if not final_test:
        if not resize_images:
            slice_df.to_csv(f"results/{run_name}_slice_dice.csv", index=False)
        else:
            slice_df.to_csv(f"results/{run_name}_slice_dice_resized.csv", index=False)        
    else:
        if not resize_images:
            slice_df.to_csv(f"results/{run_name}_slice_dice_final.csv", index=False)
        else:
            slice_df.to_csv(f"results/{run_name}_slice_dice_resized_final.csv", index=False)

    if model_name != "UNet3":
        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df[['vendor', 'volume', 'slice_number']] = slice_df['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df[["dice_IRF", "dice_SRF", "dice_PED"]].mean().to_frame().T
        class_df_std = slice_df[["dice_IRF", "dice_SRF", "dice_PED"]].std().to_frame().T
        resulting_class_df = class_df_mean.astype(str) + " (" + class_df_std.astype(str) + ")"

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice.csv", index=True)
        elif not final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized.csv", index=True)            
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_final.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized_final.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_final.csv", index=True)

        # Handles the information only on the slices that have the fluid
        slice_df_wf = slice_df.copy()
        # Iterates through all the classes available
        for i in range(number_of_classes):
            # Sets the Dice values to NaN whenever there is no fluid of that type
            slice_df_wf.loc[slice_df_wf[f"voxels_{label_to_fluids.get(i)}"] == 0, f"dice_{label_to_fluids.get(i)}"] = np.nan

        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df_wf[['vendor', 'volume', 'slice_number']] = slice_df_wf['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df_wf[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df_wf[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df_wf[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df_wf[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df_wf[["dice_IRF", "dice_SRF", "dice_PED"]].mean().to_frame().T
        class_df_std = slice_df_wf[["dice_IRF", "dice_SRF", "dice_PED"]].std().to_frame().T
        resulting_class_df = class_df_mean.astype(str) + " (" + class_df_std.astype(str) + ")"

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_wfluid.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_wfluid.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_wfluid.csv", index=True)
        elif not final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wfluid.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized_wfluid.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wfluid.csv", index=True)
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wfluid_final.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized_wfluid_final.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wfluid_final.csv", index=True)
        
        # Handles the information only on the slices that do not have fluid
        slice_df_wof = slice_df.copy()
        # Iterates through all the classes available
        for i in range(number_of_classes):
            # Gets the DataFrame that contains a non-negative number of voxels for each column
            slice_df_wof.loc[slice_df_wof[f"voxels_{label_to_fluids.get(i)}"] > 0, f"dice_{label_to_fluids.get(i)}"] = np.nan

        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df_wof[['vendor', 'volume', 'slice_number']] = slice_df_wof['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df_wof[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df_wof[["volume", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df_wof[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df_wof[["vendor", "dice_IRF", "dice_SRF", "dice_PED"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df_wof[["dice_IRF", "dice_SRF", "dice_PED"]].mean().to_frame().T
        class_df_std = slice_df_wof[["dice_IRF", "dice_SRF", "dice_PED"]].std().to_frame().T
        resulting_class_df = class_df_mean.astype(str) + " (" + class_df_std.astype(str) + ")"

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_wofluid.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_wofluid.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_wofluid.csv", index=True)
            binary_dices_name = "fluid_dice"
        elif not final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wofluid.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized_wofluid.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wofluid.csv", index=True)
            binary_dices_name = "fluid_dice_resized"
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wofluid_final.csv", index=True)
            resulting_class_df.to_csv(f"results/{run_name}_class_dice_resized_wofluid_final.csv", index=False)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wofluid_final.csv", index=True)
            binary_dices_name = "fluid_dice_resized_final"

        # Initializes a list that will hold the results for the binary case
        binary_dices = []
        # Appends the binary Dice coefficient to a list that holds these values
        binary_dices.append(f"{slice_df['binary_dice'].mean()} ({slice_df['binary_dice'].std()})")

        # Get the fluid voxel column names (i = 1, 2, 3)
        fluid_voxel_cols = [f"voxels_{label_to_fluids[i]}" for i in [1, 2, 3]]

        # Copies the original DataFrame 
        # on which changes will be applied
        slice_df_wf = slice_df.copy()
        slice_df_wof = slice_df.copy()

        # For the 'with fluid' DataFrame, the slices with no fluid will be set to NaN in the 
        # column that contains the binary Dice        
        slice_df_wf.loc[slice_df_wf[fluid_voxel_cols].sum(axis=1) == 0, 'binary_dice'] = np.nan

        # For the 'without fluid' DataFrame, the slices with any fluid will be set to NaN in the 
        # column that contains the binary Dice
        slice_df_wof.loc[slice_df_wof[fluid_voxel_cols].sum(axis=1) > 0, 'binary_dice'] = np.nan

        # The mean and std of each column is added to a list that contains the results in binary conditions
        binary_dices.append(f"{slice_df_wf['binary_dice'].mean()} ({slice_df_wf['binary_dice'].std()})")
        binary_dices.append(f"{slice_df_wof['binary_dice'].mean()} ({slice_df_wof['binary_dice'].std()})")

        # Saves the results as a DataFrame
        df = Series(binary_dices).to_frame().T
        df.columns = ["AllSlices", "SlicesWithFluid", "SlicesWithoutFluid"]
        df.to_csv(f"results/{run_name}_{binary_dices_name}.csv", index=False)
    else:
        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df[['vendor', 'volume', 'slice_number']] = slice_df['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df[["volume", f"dice_{fluid}"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df[["volume", f"dice_{fluid}"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df[["vendor", f"dice_{fluid}"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df[["vendor", f"dice_{fluid}"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df[f"dice_{fluid}"].mean()
        class_df_std = slice_df[f"dice_{fluid}"].std()
        resulting_class_df = class_df_mean.astype(str) + " (" + class_df_std.astype(str) + ")"

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice.csv", index=True)
        elif final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_final.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_final.csv", index=True)
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized.csv", index=True)

        # Handles the information only on the slices that have the fluid
        slice_df_wf = slice_df.copy()
        # Sets the Dice values to NaN whenever there is no fluid of that type
        slice_df_wf.loc[slice_df_wf[f"voxels_{fluid}"] == 0, f"dice_{fluid}"] = np.nan

        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df_wf[['vendor', 'volume', 'slice_number']] = slice_df_wf['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df_wf[["volume", f"dice_{fluid}"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df_wf[["volume", f"dice_{fluid}"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df_wf[["vendor", f"dice_{fluid}"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df_wf[["vendor", f"dice_{fluid}"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df_wf[f"dice_{fluid}"].mean()
        class_df_std = slice_df_wf[f"dice_{fluid}"].std()
        resulting_class_df_wf = str(class_df_mean) + " (" + str(class_df_std) + ")"

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_wfluid.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_wfluid.csv", index=True)
        elif final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wfluid_final.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wfluid_final.csv", index=True)
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wfluid.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wfluid.csv", index=True)
        
        # Handles the information only on the slices that do not have fluid
        slice_df_wof = slice_df.copy()
        # Gets the DataFrame that contains a non-negative number of voxels for each column
        slice_df_wof.loc[slice_df_wof[f"voxels_{fluid}"] > 0, f"dice_{fluid}"] = np.nan

        # Adds the vendor, volume, and number of the slice information to the DataFrame
        slice_df_wof[['vendor', 'volume', 'slice_number']] = slice_df_wof['slice'].str.replace('.tiff', '', regex=True).str.split('_', n=2, expand=True)
        # Saves the DataFrame with the mean and standard deviation for each OCT volume (e.g. mean (standard deviation))
        volume_df_mean = slice_df_wof[["volume", f"dice_{fluid}"]].groupby("volume").mean()
        volume_df_mean.index.name = "Volume"
        volume_df_std = slice_df_wof[["volume", f"dice_{fluid}"]].groupby("volume").std()
        volume_df_std.index.name = "Volume"
        resulting_volume_df = volume_df_mean.astype(str) + " (" + volume_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each vendor (e.g. mean (standard deviation))
        vendor_df_mean = slice_df_wof[["vendor", f"dice_{fluid}"]].groupby("vendor").mean()
        vendor_df_mean.index.name = "Vendor"
        vendor_df_std = slice_df_wof[["vendor", f"dice_{fluid}"]].groupby("vendor").std()
        vendor_df_std.index.name = "Vendor"
        resulting_vendor_df = vendor_df_mean.astype(str) + " (" + vendor_df_std.astype(str) + ")"

        # Saves the DataFrame with the mean and standard deviation for each class (e.g. mean (standard deviation))
        class_df_mean = slice_df_wof[f"dice_{fluid}"].mean()
        class_df_std = slice_df_wof[f"dice_{fluid}"].std()
        resulting_class_df_wof = str(class_df_mean) + " (" + str(class_df_std) + ")"

        # Appends all the results to a list that will be saved in a Series
        classes_results = [resulting_class_df, resulting_class_df_wf, resulting_class_df_wof] 

        classes_df = Series(classes_results).to_frame().T
        classes_df.columns = ["AllSlices", "SlicesWithFluid", "SlicesWithoutFluid"]

        # Saves the DataFrame that contains the values for each volume, class, and vendor
        if not resize_images:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_wofluid.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_wofluid.csv", index=True)
            classes_df.to_csv(f"results/{run_name}_class_dice.csv", index=False)
        elif final_test:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wofluid_final.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wofluid_final.csv", index=True)
            classes_df.to_csv(f"results/{run_name}_class_dice_resized_final.csv", index=False)
        else:
            resulting_volume_df.to_csv(f"results/{run_name}_volume_dice_resized_wofluid.csv", index=True)
            resulting_vendor_df.to_csv(f"results/{run_name}_vendor_dice_resized_wofluid.csv", index=True)
            classes_df.to_csv(f"results/{run_name}_class_dice_resized.csv", index=False)
        
# In case it is preferred to run 
# directly in this file, here lays 
# an example
if __name__ == "__main__":
    test_model(
        fold_test=2,
        model_name="UNet",
        weights_name="Run1000_UNet_best_model.pth",
        number_of_channels=1,
        number_of_classes=4,
        device_name="GPU",
        batch_size=1,
        patch_type="small",
        save_images=False,
        resize_images=False
    )
