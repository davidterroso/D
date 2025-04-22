import torch
from IPython import get_ipython
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import array, float32, nan, uint8, where, zeros_like
from os import makedirs
from pandas import read_csv
from torch.utils.data import DataLoader
from typing import List, Union
from networks.loss import dice_coefficient
from networks.unet import UNet
from network_functions.dataset import TestDataset
from paths import IMAGES_PATH
from test_model import collate_fn

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def visualize_binary_segmentations(volume_index: int, binary_split: bool, 
                                   merging_strat: str, order: Union[List[int], None]):
    """
    This function will be used to visualize the overlay of multiple binary 
    segmentation masks. Two binary segmentation splits were used: one with 
    the same partition as the multi-class segmentation, and one specific 
    for the class that is segmented. This network receives as input the 
    index of the volume desired to segment and the models of which split 
    will be used

    Args:
        volume_index (int): index of the volume desired to segment
        binary_split (bool): boolean that indicates whether the split 
            used will be the one used specifically for binary segmentation
            when assigned True or the one for multi-class segmentation when
            assigned False
        merging_strat (str): strategy used when merging the multiple binary
            masks. Can be: 'priority' (needs the argument order) or 'softmax'
        order (List[int]): order of priority of the different masks. The 
            first value is the most important
            
    Returns:
        None
    """

    # Defines the dictionaries that assign 
    # the number of the run to the validation
    # split used in the same run
    irf_run_to_val_split = {
        63: 2,
        64: 3,
        65: 4,
        66: 0,
        75: 2,
        76: 3,
        77: 4,
        78: 0
    }

    srf_run_to_val_split = {
        67: 2,
        68: 3,
        69: 4,
        70: 0,
        79: 2,
        80: 3,
        81: 4,
        82: 0
    }

    ped_run_to_val_split = {
        71: 2,
        72: 3,
        73: 4,
        74: 0,
        83: 2,
        84: 3,
        85: 4,
        86: 0,
    }

    # Dictionary that associates 
    # a label with a variable
    mask_dict = {
        1: irf_preds,
        2: srf_preds,
        3: ped_preds
    }

    # In case the split used specifically for the binary fluid segmentation 
    # is used three splits have to be checked
    if binary_split:
        # Declares the folder in which the images will be saved
        folder_to_save = IMAGES_PATH + f"\\OCT_images\\prediction\\binary_segmentation_binary_split\\"
        
        # Reads the splits for the splits used in the binary segmentation
        irf_split = read_csv("splits\\competitive_fold_selection_IRF.csv")
        srf_split = read_csv("splits\\competitive_fold_selection_SRF.csv")
        ped_split = read_csv("splits\\competitive_fold_selection_PED.csv")

        # Gets the name of the fold in which the volume selected to segment 
        # is located
        irf_fold = irf_split.columns[irf_split.isin([volume_index]).any()][0]
        srf_fold = srf_split.columns[srf_split.isin([volume_index]).any()][0]
        ped_fold = ped_split.columns[ped_split.isin([volume_index]).any()][0]

        # If the volume is located in the test split, the volume's masks 
        # are not inferred
        if ((irf_fold == "1") or (srf_fold == "1") or (ped_fold == "1")):
            print(f"The volume {volume_index} is reserved to fold 1.")
            return

        # Gets the name of the run in which the validation fold contains the volume to infer 
        # from the dictionary 
        irf_run = [key for key, val in irf_run_to_val_split.items() if val == int(irf_fold)][1]
        srf_run = [key for key, val in srf_run_to_val_split.items() if val == int(srf_fold)][1]
        ped_run = [key for key, val in ped_run_to_val_split.items() if val == int(ped_fold)][1]

    # In case the split used is the same as the multi-class
    else:
        # Declares the name of the folder in which the images will be saved
        folder_to_save = IMAGES_PATH + "\\OCT_images\\prediction\\binary_segmentation_multiclass_split\\"
        
        # Reads the split from used in multi-class segmentation
        split = read_csv("splits\\competitive_fold_selection.csv")

        # Gets the fold in which the volume desired to segment is located
        fold = split.columns[split.isin([volume_index]).any()][0]
        
        # If the volume is located in the test split, the volume's masks 
        # are not inferred
        if (fold == "1"):
            print(f"The volume {volume_index} is reserved to fold 1.")
            return

        # Gets the name of the run in which the validation fold contains the volume to infer 
        # from the dictionary 
        irf_run = [key for key, val in irf_run_to_val_split.items() if val == int(fold)][0]
        srf_run = [key for key, val in srf_run_to_val_split.items() if val == int(fold)][0]
        ped_run = [key for key, val in ped_run_to_val_split.items() if val == int(fold)][0]

    # The name of the folder is given 
    # according to the merging strat used
    if merging_strat == "priority":
        folder_to_save = folder_to_save + f"Volume{volume_index}_{irf_run}_{srf_run}_{ped_run}_{merging_strat}_{''.join(str(k) for k in order)}\\"
    elif merging_strat == "softmax":
        folder_to_save = folder_to_save + f"Volume{volume_index}_{irf_run}_{srf_run}_{ped_run}_{merging_strat}\\"
    # Creates the folder in case it does not exist yet
    makedirs(folder_to_save, exist_ok=True)

    # Declares the path to the models that will be used to predict each fluid's mask
    irf_weights_path = f"models\\Run{str(irf_run).zfill(3)}_IRF_UNet3_best_model.pth"
    srf_weights_path = f"models\\Run{str(srf_run).zfill(3)}_SRF_UNet3_best_model.pth"
    ped_weights_path = f"models\\Run{str(ped_run).zfill(3)}_PED_UNet3_best_model.pth"

    # Loads the models to the GPU
    irf_model = UNet(in_channels=1, num_classes=2).to(device=torch.device("cuda"))
    srf_model = UNet(in_channels=1, num_classes=2).to(device=torch.device("cuda"))
    ped_model = UNet(in_channels=1, num_classes=2).to(device=torch.device("cuda"))

    # Loads the weights to the model and assignes the model to the GPU
    irf_model.load_state_dict(torch.load(irf_weights_path, weights_only=True, map_location=torch.device("cuda")))
    irf_model.eval()
    srf_model.load_state_dict(torch.load(srf_weights_path, weights_only=True, map_location=torch.device("cuda")))
    srf_model.eval()
    ped_model.load_state_dict(torch.load(ped_weights_path, weights_only=True, map_location=torch.device("cuda")))
    ped_model.eval()

    # Creates the Dataset object as a multi-class since multiple models will have the classes compared
    test_dataset = TestDataset([volume_index], "UNet", "vertical", True, (496,512), None, 1)
    # Creates the DataLoader object
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=12, collate_fn=collate_fn)

    # Initiates an array that 
    # will contain the 
    # Dice-Sorensen coefficient 
    # for each class in the 
    # binary segmentations
    slices_dsc_irf = []
    slices_dsc_srf = []
    slices_dsc_ped = []

    # Initiates an array that 
    # will contain the 
    # Dice-Sorensen coefficient 
    # for each class in the 
    # merged segmentations
    merged_dsc_irf = []
    merged_dsc_srf = []
    merged_dsc_ped = []

    # Indicates that no gradient 
    # must be calculated in the models
    with torch.no_grad():
        # Creates a progress bar to track the progress on testing images
        with tqdm(test_dataloader, total=len(test_dataloader), desc='Testing Models', unit='img', leave=True, position=0) as progress_bar:
            # Iterates through every batch and path 
            # (that compose the batch) in the dataloader
            # In this case, the batches are of size one, 
            # so every batch is handled like a single image
            for batch in test_dataloader:
                # Gets the images and the masks from the DataLoader
                images, true_masks, image_name = batch['scan'], batch['mask'], batch['image_name']

                # Handles the images and masks according to the device, specified data type and memory format
                images = images.to(device=torch.device("cuda"), dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=torch.device("cuda"), dtype=torch.long)

                # Converts the image from channels 
                # last to channels first
                images = images.permute(0, 3, 1, 2)

                # Predicts the output of the batch
                irf_outputs = irf_model(images)
                srf_outputs = srf_model(images)
                ped_outputs = ped_model(images)

                # Stacks all the logits
                logit_stack = torch.stack([irf_outputs, srf_outputs, ped_outputs], axis=-1)

                # Calculates the prediction of each independent model
                irf_preds = torch.argmax(irf_outputs, dim=1) * 1
                srf_preds = torch.argmax(srf_outputs, dim=1) * 2
                ped_preds = torch.argmax(ped_outputs, dim=1) * 3

                # Stacks all the preds
                preds_stack = torch.stack([irf_preds, srf_preds, ped_preds], axis=-1)

                # Creates a binary mask signalling where there 
                # is overlaps and where there is not
                overlap_mask = ((irf_preds > 0).astype(int) +
                (srf_preds > 0).astype(int) +
                (ped_preds > 0).astype(int)) > 1

                # Initializes the mask of merged predictions as tensor of zeros
                merged_preds = torch.zeros_like(irf_preds, dtype=uint8).to(device=torch.device("cuda"))

                # In case the merging strat 
                # selected is "priority"
                if merging_strat == "priority":
                    # Builds a prediction mask 
                    # using the input priority
                    # It iterates in the reverse
                    # order to apply lowest priority first
                    for class_id in order[::-1]:
                        mask = (mask_dict[class_id] > 0).int()
                        merged_preds = where(mask > 0, class_id, merged_preds)
                
                # In case the merging strategy 
                # selected is "softmax"                
                elif merging_strat == "softmax":
                    # Creates a mask of the pixels 
                    # where there is no overlapping                
                    non_overlap = ~overlap_mask
                    # Assigns the labels as of the only class where there is no zero
                    merged_preds[non_overlap] = torch.argmax(preds_stack[non_overlap], dim=-1) + 1
                    # Gets the indices of where there is overlapping
                    overlap_indices = torch.where(overlap_mask)
                    # Gets the logits of where there is overlapping
                    overlap_logits = logit_stack[overlap_indices]
                    # Applies softmax across the different predictions
                    overlap_softmax = torch.softmax(overlap_logits, dim=-1)
                    # Whichever class has the highest probability (or logit)
                    # is considered the predicted class
                    fused_classes = torch.argmax(overlap_softmax, dim=-1) + 1
                    # Sets the mask in the overlap pixels to the 
                    # maximum probability between the predicted classes
                    merged_preds[overlap_indices] = fused_classes

                # Declares the name of the predicted mask
                predicted_mask_name = folder_to_save + image_name[:-5] + ".tiff"
    
                # Gets the Dice coefficient for each mask
                dice_scores_irf, _, _, _, _ = dice_coefficient(target=true_masks, prediction=irf_preds, model_name="UNet3", num_classes=2)
                dice_scores_srf, _, _, _, _ = dice_coefficient(target=true_masks, prediction=srf_preds, model_name="UNet3", num_classes=2)
                dice_scores_ped, _, _, _, _ = dice_coefficient(target=true_masks, prediction=ped_preds, model_name="UNet3", num_classes=2)
                # Gets the Dice coefficient for the merged mask
                dice_scores, _, _, _, _ = dice_coefficient(target=true_masks, prediction=merged_preds, model_name="UNet", num_classes=4)

                # Gets the original OCT B-scan
                oct_image = images[0].cpu().numpy()[0]

                # Converts each voxel classified as background to 
                # NaN so that it will not appear in the overlaying
                # mask
                irf_preds = array(irf_preds.cpu().numpy(), dtype=float32)[0]
                irf_preds[irf_preds == 0] = nan
                srf_preds = array(srf_preds.cpu().numpy(), dtype=float32)[0]
                srf_preds[srf_preds == 0] = nan
                ped_preds = array(ped_preds.cpu().numpy(), dtype=float32)[0]
                ped_preds[ped_preds == 0] = nan
                merged_preds = array(merged_preds.cpu().numpy(), dtype=float32)[0]
                merged_preds[merged_preds == 0] = nan
                true_masks = array(true_masks.cpu().numpy(), dtype=float32)[0]
                true_masks[true_masks == 0] = nan

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

                # Initiates a subplot that will have three images along one row
                fig, axes = plt.subplots(1, 3, figsize=(2 * oct_image.shape[1] / 100, oct_image.shape[0] / 100))

                # Plots the ground-truth masks
                axes[0].imshow(oct_image, cmap=plt.cm.gray)
                axes[0].imshow(true_masks, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                axes[0].axis("off")
                axes[0].set_title("Ground Truth Masks")

                # Plots the predicted masks and the overlaying in purple
                axes[1].imshow(oct_image, cmap=plt.cm.gray)
                axes[1].imshow(irf_preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                axes[1].imshow(srf_preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                axes[1].imshow(ped_preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                axes[1].imshow(overlap_mask, alpha=0.7, cmap="Purples")
                axes[1].axis("off")
                axes[1].set_title(f"Predicted Masks. DSC: {dice_scores_irf[1]}, {dice_scores_srf[1]}, {dice_scores_ped[1]}")

                # Plots the predicted masks merged
                axes[2].imshow(oct_image, cmap=plt.cm.gray)
                axes[2].imshow(merged_preds, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                axes[2].axis("off")
                axes[2].set_title(f"Merged Masks. DSC: {dice_scores[1:3]}")

                # Saves the combined figures
                plt.savefig(predicted_mask_name, bbox_inches='tight', pad_inches=0)
                plt.clf()
                plt.close("all")

                # Appends the Dice coefficient for the 
                # slice in the binary models
                slices_dsc_irf.append(dice_scores_irf[1])
                slices_dsc_srf.append(dice_scores_srf[1])
                slices_dsc_ped.append(dice_scores_ped[1])

                # Appends the Dice coefficient for the 
                # slice with the merged binary models
                merged_dsc_irf.append(dice_scores[1])
                merged_dsc_srf.append(dice_scores[2])
                merged_dsc_ped.append(dice_scores[3])

                # Update the progress bar
                progress_bar.update(1)

    # Prints the statistics of the model
    print(f"Volume {volume_index}")

    print(f"IRF: {array(slices_dsc_irf).mean()}")
    print(f"SRF: {array(slices_dsc_srf).mean()}")
    print(f"PED: {array(slices_dsc_ped).mean()}")

    print(f"Merged PED: {array(merged_dsc_irf).mean()}")
    print(f"Merged PED: {array(merged_dsc_srf).mean()}")
    print(f"Merged PED: {array(merged_dsc_ped).mean()}")

visualize_binary_segmentations(volume_index=1, binary_split=True, merging_strat="binary", order=[2,1,3])
