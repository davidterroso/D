import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython import get_ipython
from os import makedirs
from os.path import exists
from pandas import DataFrame, read_csv
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from shutil import rmtree
from networks.loss import dice_coefficient
from networks.unet25D import TennakoonUNet
from networks.unet import UNet
from network_functions.dataset import TestDataset
from paths import IMAGES_PATH

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

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


def folds_results(first_run_name: str, iteration: int, k: int=5):
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
        # Indicates the name of the file that will store the Dice 
        # per class
        class_file_name = f".\\results\\{run_name}_class_dice.csv"
        # Indicates the name of the file that will store the Dice 
        # per vendor
        vendor_file_name = f".\\results\\{run_name}_vendor_dice.csv"
        # Reads the DataFrame that handles the data per class
        class_df = read_csv(class_file_name)
        # Reads the DataFrame that handles the data per vendor
        vendor_df = read_csv(vendor_file_name, index_col="vendor")
        # Removes the name of the column that has the table's index
        vendor_df.index.name = None
        # Saves, to the corresponding fold in the 
        # dictionary, the two DataFrames as a 
        # single tuple
        df_dict[fold] = (class_df, vendor_df)
    # Gets the list of vendors and fluids
    vendors = vendor_df.index.to_list()
    fluids = class_df.columns.to_list()

    # Initiates the DataFrame with the name 
    # of the fluids as the columns names for 
    # the vendor data
    # e.g. of a column name: Dice_IRF
    vendor_df = DataFrame(columns=fluids)
    # Iterates through the 
    # vendors in the DataFrame
    # (rows)
    for vendor in vendors:
        # In each vendor stores an array of values 
        # that will later be inserted as a row
        values = []
        # Iterates through the fluids
        for fluid in fluids:
            # Initiates a list that will store the 
            # results across all folds
            results = []
            # Iterates across all folds in the dictionary
            for fold, tuple_df in df_dict.items():
                # Appends to the results list, the value at 
                # the current vendor and fluid
                # The DataFrame that is being handled is the 
                # one that comprises information about classes
                # and vendors
                results.append(tuple_df[1].at[vendor, fluid])
            # Calculates the mean across all folds
            mean = np.array(results).mean()
            # Calculates the standard deviation across all folds
            std = np.array(results).std()
            # Saves the value as "mean (std)"
            value = f"{mean:.2f} ({std:.2f})"
            # Appends the value to the list that will form a row
            values.append(value)
        # Appends the results in a row to the DataFrame
        vendor_df.loc[len(vendor_df)] = values
    # Sets the name of the axis in the 
    # DataFrame to the name of the vendors
    vendor_df = vendor_df.set_axis(vendors)
    # Saves the DataFrame with a name refering to the iteration
    vendor_df.to_csv(f".\\results\\Iteration{iteration}_vendors_results.csv")

    # Initiates the DataFrame with the name 
    # of the fluids as the columns names for 
    # the class data
    # e.g. of a column name: Dice_IRF
    class_df = DataFrame(columns=fluids)
    # Initiates a list that will store the 
    # values of the classes 
    values = []
    # Iterates through the fluids
    for fluid in fluids:
        # Initiates a list that will store the 
        # results across all folds
        results = []
        # Iterates across all folds in the dictionary
        for fold, tuple_df in df_dict.items():
            # Appends to the results list, the value at 
            # the current vendor and fluid
            # The DataFrame that is now being handled 
            # comprises information of the classes
            results.append(tuple_df[0].at[0, fluid])
        # Calculates the mean across all folds
        mean = np.array(results).mean()
        # Calculates the standard deviation across all folds
        std = np.array(results).std()
        # Saves the value as "mean (std)"
        value = f"{mean:.2f} ({std:.2f})"
        # Appends the value to the list that will form a row
        values.append(value)
    # Appends the results in a row to the DataFrame
    class_df.loc[len(class_df)] = values
    # Saves the DataFrame with a name refering to the iteration, not including the index
    class_df.to_csv(f".\\results\\Iteration{iteration}_classes_results.csv", index=False)

def test_model (
        fold_test: int,
        model_name: str,
        weights_name: str,
        device_name: str,
        number_of_channels: int,
        number_of_classes: int,
        batch_size: int,
        patch_type: str,
        save_images: bool
    ):
    """
    Function used to test the trained models

    Args:
        fold_test (int): number of the fold that will be used 
            in the network testing 
        model_name (str): name of the model that will be 
            evaluated         
        weights_name (str): path to the model's weight file
        device_name (str): indicates whether the network will 
            be trained using the CPU or the GPU
        number_of_channels (int): number of channels the 
            input will present
        number_of_classes (int): number of classes the 
            output will present
        batch_size (int): size of the batch used in testing
        patch_type (str): string that indicates what type of patches 
            will be used. Can be "small", where patches of size 
            256x128 are extracted using the extract_patches function,
            "big", where patches of shape 496x512 are extracted from 
            each image, and patches of shape 496x128 are extracted from
            the slices
        save_images (bool): flag that indicates whether the 
            predicted images will be saved or not

    Return:
        None
    """
    # Gets the list of volumes used to test the model
    df = read_csv("splits/competitive_fold_selection.csv")
    test_volumes = df[str(fold_test)].dropna().to_list()

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

    # Creates the TestDataset and DataLoader object with the test volumes
    # Number of workers was set to the most optimal
    test_dataset = TestDataset(test_volumes, model_name, patch_type)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, collate_fn=collate_fn)
    
    # Initiates the list that will 
    # store the results of the slices 
    slice_results = []
    # Initiates the dictionary that will 
    # store the results per volume, per 
    # vendor, and per class
    volume_results = defaultdict(list)
    vendor_results = defaultdict(list)

    # Extracts the name of the run from 
    # the name of the weights file
    run_name = weights_name.split("_")[0]

    # Declares the name of the folder in which the images will be saved
    if save_images:
        # In case the folder to save exists, it is deleted and created again
        folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}\\"
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
                # Gets the images and the masks from the dataloader
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
                dice_scores, voxel_counts, union_counts, intersection_counts = dice_coefficient(model_name, preds, true_masks, number_of_classes)
                # Gets the information from the slice's name
                vendor, volume, slice_number = image_name[:-5].split("_")
                volume_name = f"{vendor}_{volume}"
                
                # Appends the results per slice, per volume, and per vendor
                slice_results.append([image_name, *dice_scores, *voxel_counts, *union_counts, *intersection_counts])
                volume_results[volume_name].append((union_counts, intersection_counts))
                vendor_results[vendor].append((union_counts, intersection_counts))

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
    slice_df = DataFrame(slice_results, 
                        columns=["slice", 
                                *[f"dice_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"voxels_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"union_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"intersection_{label_to_fluids.get(i)}" for i in range(number_of_classes)]])
    
    slice_df.to_csv(f"results/{run_name}_slice_dice.csv", index=False)

    # Creates a list with all the Dices associated with a volume
    volume_dice_results = []
    # Iterates through a dictionary that contains the name 
    # of the volume and the intersection and union of values 
    # obtained in the said volume per class
    for volume_name, slice_values in volume_results.items():
        # For each volume, initiates a list that will contain 
        # the total number of intersections and unions in the 
        # slices considered
        total_union = [0] * number_of_classes
        total_intersection = [0] * number_of_classes
        # In class of the slice, gets the union and intersections obtained
        for (union_values, intersection_values) in slice_values:
            for i in range(number_of_classes):
                total_union[i] += union_values[i]
                total_intersection[i] += intersection_values[i]
        # Iterates through the classes and in case there are 
        # voxels labeled in the said class, calculates the Dice value
        # Otherwise is set to 0
        dice_per_class = [0] * number_of_classes
        for i in range(number_of_classes):
            if total_union[i] > 0:
                dice_per_class[i] = (2. * total_intersection[i] / total_union[i])
            else:
                dice_per_class[i] = 0
        # Appends the results to a list that contains all the Dice 
        # in all the volumes
        volume_dice_results.append([volume_name, *dice_per_class])
                
    # Names the columns that will compose the CSV file
    columns = ["volume"] + [f"Dice_{label_to_fluids.get(i)}" for i in range(number_of_classes)]
    # Adds the Dice values to the Pandas DataFrame
    volume_df = DataFrame(volume_dice_results, columns=columns)
    # Saves it as CSV
    volume_df.to_csv(f"results/{run_name}_volume_dice.csv", index=False)    
    
    # Initiates a list that will have the count of the union, the 
    # count of the intersection, and the Dice values per class
    total_union = [0] * number_of_classes
    total_intersection = [0] * number_of_classes
    overall_dice_per_class = [0] * number_of_classes
    # Iterates through all the volumes and their slices
    for volume_name, slice_values in volume_results.items():
        for (union_values, intersection_values) in slice_values:
            for i in range(number_of_classes):
                # In each class sums to the previous 
                # total the number of union and 
                # intersection values
                total_union[i] += union_values[i]
                total_intersection[i] += intersection_values[i]

    # With the intersection and union values, calculates the Dice coefficient
    # In case there are no positive voixels in the said class, sets the Dice to zero
    for i in range(number_of_classes):
        if total_union[i] > 0:
            overall_dice_per_class[i] = (2. * total_intersection[i] / total_union[i])
        else:
            overall_dice_per_class[i] = 0
    
    # Names the columns according to their fluid type
    columns = [f"Dice_{label_to_fluids.get(i)}" for i in range(number_of_classes)]
    # Creates a Pandas DataFrame with the column names and its respective values 
    class_df = DataFrame([overall_dice_per_class], columns=columns)
    # Converts this DataFrame to a CSV file 
    class_df.to_csv(f"results/{run_name}_class_dice.csv", index=False)

    # Dictionary to store accumulated union and intersection counts per vendor
    vendor_union = defaultdict(lambda: [0] * number_of_classes)
    vendor_intersection = defaultdict(lambda: [0] * number_of_classes)

    # Iterate over each vendor and accumulate union 
    # and intersection counts depending on the class
    for vendor, results in vendor_results.items():
        for (union_counts, intersection_counts) in results:
            for i in range(number_of_classes):
                vendor_union[vendor][i] += union_counts[i]
                vendor_intersection[vendor][i] += intersection_counts[i]

    # Compute Dice coefficients per class for each vendor
    # Initiates an empty list
    vendor_dice_results = []
    # Iterates through the vendors 
    for vendor in vendor_union.keys():
        # For each class calculates the Dice coefficient 
        for i in range(number_of_classes):
            if vendor_union[vendor][i] > 0:
                dice_per_class[i] = (2. * vendor_intersection[vendor][i] / vendor_union[vendor][i])
            else:
                dice_per_class[i] = 0
        # Appends the Dice values to the list
        vendor_dice_results.append([vendor, *dice_per_class])

    # Names the DataFrame columns
    columns = ["vendor"] + [f"Dice_{label_to_fluids.get(i, f'Class_{i}')}" for i in range(number_of_classes)]
    # Creates the DataFrame
    vendor_df = DataFrame(vendor_dice_results, columns=columns)
    vendor_df.to_csv(f"results/{run_name}_vendor_dice.csv", index=False)
    # Save the DataFrame as CSV

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
        save_images=False
    )
