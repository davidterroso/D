import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from collections import defaultdict
from os import makedirs
from os.path import exists
from pandas import DataFrame, read_csv
from skimage.io import imread
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from shutil import rmtree
from networks.loss import dice_coefficient
from networks.unet25D import TennakoonUNet
from networks.unet import UNet
from network_functions.dataset import TestDataset
from paths import IMAGES_PATH

def compute_class_avg(overall_results, num_classes):
    """
    Function used for the calculation of the Dice coefficient per class
    depending on the voxels per class

    Args:
        overall_results (dict(int: float, int: int)): dictionary that 
            contains the class value as a key to access the Dice coefficient 
            and the total number of voxels in the class
        num_classes (int): number of classes used

    Return:
        class_avg (List[float]): list that contains the Dice coefficient 
            values for each class
    """
    # Initiates the list that 
    # will contain the Dice 
    # value for each class
    class_avg = []
    # Iterates through the 
    # different classes
    for i in range(num_classes):
        # Gets the sum of the Dices in the said class 
        # and the respective total number of voxels
        dice_sum, voxel_count_sum = overall_results[i]
        # Appends the weighted average to the list
        class_avg.append(dice_sum / voxel_count_sum if voxel_count_sum > 0 else 0)
    # Returns the list with the averages per class
    return class_avg

def compute_weighted_avg(group_results, number_of_classes):
    """
    Used to compute the weighted average of the Dice coefficient, 
    according to the number of voxels identified per class

    Args:
        group_results (List[List[str, List[float], List[int], float]]):
            stores the name of the slices, the Dice coefficient per slice 
            per class, the number of voxels per slice per class and the 
            Dice coefficient of the slice
        number_of_classes (int): number of classes segmented 
    """
    # Initiates the list that will 
    # have the average results
    avg_results = []
    # Iterates through the values of the list
    for name, values in group_results.items():
        # Counts the total number of voxels
        total_voxels = torch.tensor([sum(v[i] for _, v in values) for i in range(number_of_classes)])
        # Calculates the total dice for the slice
        total_dice = torch.tensor([sum(d[i] * v[i] for d, v in values) for i in range(number_of_classes)])
        # Calculates the average value
        avg_dice = (total_dice / total_voxels).tolist()
        # Appends the results
        avg_results.append([name, *avg_dice, total_dice.sum().item() / total_voxels.sum().item()])
    return avg_results

def test_model (
        fold_test,
        model_name,
        weights_name,
        device_name,
        number_of_channels,
        number_of_classes,
        batch_size,
        save_images
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
        batch_size (int): size of the batch used in testing
        save_images (bool): flag that indicates whether the 
            predicted images will be saved or not

    Return:
        None
    """
    # Gets the list of volumes used to test the model
    df = read_csv("splits/segmentation_test_splits.csv")
    test_fold_column_name = f"Fold{fold_test}_Volumes"
    test_volumes = df[test_fold_column_name].dropna().to_list()

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
    test_dataset = TestDataset(test_volumes, model_name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
    # Initiates the list that will 
    # store the results of the slices 
    slice_results = []
    # Initiates the dictionary that will 
    # store the results per volume, per 
    # vendor, and per class
    volume_results = defaultdict(list)
    vendor_results = defaultdict(list)
    class_results = defaultdict(lambda: [0, 0])

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
                
                # Calculates the Dice coefficient of the predicted mask
                dice_scores, voxel_counts, total_dice = dice_coefficient(model_name, preds, true_masks, number_of_classes)
                # Gets the information from the slice's name
                vendor, volume, slice_number = image_name[0][:-5].split("_")
                volume_name = f"{vendor}_{volume}"
                
                # Appends the results per slice, per volume, and per vendor
                slice_results.append([image_name[0], *dice_scores, *voxel_counts, total_dice])
                volume_results[volume_name].append((dice_scores, voxel_counts))
                vendor_results[vendor].append((dice_scores, voxel_counts))

                # Aggregate Dice scores per class across all images
                for i in range(number_of_classes):
                    # Calculates the weighted sum
                    class_results[i][0] += dice_scores[i] * voxel_counts[i]
                    # Calculates the total voxel count
                    class_results[i][1] += voxel_counts[i]

                # Saves the predicted masks and the GT, in case it is desired
                if save_images:
                    # Declares the name under which the masks will be saved and writes the path to the original B-scan
                    predicted_mask_name = folder_to_save + image_name[0][:-5] + "_predicted" + ".tiff"
                    gt_mask_name = folder_to_save + image_name[0][:-5] + "_gt" + ".tiff"

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
                    plt.figure()
                    plt.imshow(oct_image, cmap=plt.cm.gray)
                    plt.imshow(preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                    plt.axis("off")
                    plt.savefig(predicted_mask_name, bbox_inches='tight', pad_inches=0)

                    # Saves the OCT scan with an overlay of the ground-truth masks
                    plt.figure()
                    plt.imshow(oct_image, cmap=plt.cm.gray)
                    plt.imshow(true_masks, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                    plt.axis("off")
                    plt.savefig(gt_mask_name, bbox_inches='tight', pad_inches=0)

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
                                *[f"dice_class_{i}" for i in range(number_of_classes + 1)], 
                                *[f"voxels_class_{i}" for i in range(number_of_classes + 1)], 
                                "total_dice"])
    slice_df.to_csv(f"results/{run_name}_slice_dice.csv", index=False)

    # Saves the Dice score per volume
    volume_df = DataFrame(compute_weighted_avg(volume_results, number_of_classes), 
                        columns=["volume", 
                                *[f"dice_class_{i}" for i in range(0, number_of_classes)], 
                                "total_dice"])
    volume_df.to_csv(f"results/{run_name}_volume_dice.csv", index=False)

    # Saves the Dice score per vendor
    vendor_df = DataFrame(compute_weighted_avg(vendor_results, number_of_classes), 
                        columns=["vendor", 
                                *[f"dice_class_{i}" for i in range(0, number_of_classes)], 
                                "total_dice"])
    vendor_df.to_csv(f"results/{run_name}_vendor_dice.csv", index=False)

    # Saves the Dice score per class
    class_avg_dice = compute_class_avg(class_results, number_of_classes)
    class_df = DataFrame([class_avg_dice], columns=[f"dice_class_{i}" for i in range(number_of_classes)])
    class_df.to_csv(f"results/{run_name}_class_dice.csv", index=False)

# In case it is preferred to run 
# directly in this file, here lays 
# an example
if __name__ == "__main__":
    test_model(
        fold_test=2,
        model_name="UNet",
        weights_name="Run4_UNet_best_model.pth",
        number_of_channels=1,
        number_of_classes=4,
        device_name="GPU",
        batch_size=1,
        save_images=True
    )
