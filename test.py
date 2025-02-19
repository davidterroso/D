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

# Dictionary of labels in masks to fluid names
label_to_fluids = {
        0: "Background",
        1: "IRF",
        2: "SRF",
        3: "PED"
    }

def calculate_dice(union: int, intersection: int, epsilon: float=1e-6) -> float:
    """
    Given the number of voxels that result from the union and intersection
    of the GT mask with the predicted mask, calculates the Dice coefficient

    Args:
        union (int): number of voxels that result from the union of the GT 
            and predicted masks
        intersection (int): number of voxels that result from the intersection 
            of the GT and predicted masks

    Return:
        dice (float): Dice coefficient of the respective voxels
    """
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

def test_model (
        fold_test: int,
        model_name: str,
        weights_name: str,
        device_name: str,
        number_of_channels: int,
        number_of_classes: int,
        batch_size: int,
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
                vendor, volume, slice_number = image_name[0][:-5].split("_")
                volume_name = f"{vendor}_{volume}"
                
                # Appends the results per slice, per volume, and per vendor
                slice_results.append([image_name[0], *dice_scores, *voxel_counts, *union_counts, *intersection_counts])
                volume_results[volume_name].append((union_counts, intersection_counts))
                vendor_results[vendor].append((union_counts, intersection_counts))

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
                break

    # Creates the folder results in case 
    # it does not exist yet
    makedirs("results", exist_ok=True)

    # Saves the Dice score per slice
    slice_df = DataFrame(slice_results, 
                        columns=["slice", 
                                *[f"dice_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"voxels_class_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"union_class_{label_to_fluids.get(i)}" for i in range(number_of_classes)], 
                                *[f"intersection_class_{label_to_fluids.get(i)}" for i in range(number_of_classes)]])
    
    slice_df.to_csv(f"results/{run_name}_slice_dice_test.csv", index=False)


# In case it is preferred to run 
# directly in this file, here lays 
# an example
if __name__ == "__main__":
    test_model(
        fold_test=2,
        model_name="UNet",
        weights_name="Run5_UNet_best_model.pth",
        number_of_channels=1,
        number_of_classes=4,
        device_name="GPU",
        batch_size=1,
        save_images=True
    )
