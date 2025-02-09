import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from pandas import DataFrame, read_csv
from networks.loss import dice_coefficient
from networks.unet25D import TennakoonUNet
from networks.unet import UNet
from network_functions.dataset import TestDataset

def compute_weighted_avg(group_results, number_of_classes):
    """
    Used to compute the weighted average of the Dice coefficient, 
    according to the number of voxels identified per class

    Args:
        group_results (List[List[str, List[float], List[int], float]]):
    stores the name of the slices, the Dice coefficient per slice per
    class, the number of voxels per slice per class and the Dice 
    coefficient of the slice
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
        number_of_channels,
        number_of_classes,
        batch_size
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
    Return:
        None
    """
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
    
    weights_path = "models\\" + weights_name 

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
    test_dataset = TestDataset(test_volumes, model_name)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initiates the list that will 
    # store the results of the slices 
    slice_results = []
    # Initiates the dictionary that will 
    # store the results per volume and 
    # per vendor
    volume_results = defaultdict(list)
    vendor_results = defaultdict(list)
    
    # Informs that no backward propagation will be calculated 
    # because it is an inference, thus reducing memory consumption
    with torch.no_grad():
        # Iterates through every batch and path 
        # (that compose the batch) in the dataloader
        for batch, paths in test_dataloader:
            # Declares to which device 
            # will be allocated 
            batch = batch.to(device)
            # Predicts the output of the batch
            outputs = model(batch)
            # The prediction is assumed as the value 
            # that has a higher logit
            preds = torch.argmax(outputs, dim=1)
            
            # Itereates through the paths that 
            # compose the DataLoader object
            for i, path in enumerate(paths):
                # Calculates the Dice coefficient of the predicted mask
                dice_scores, voxel_counts, total_dice = dice_coefficient(preds[i], batch[i], number_of_classes)
                # Gets the information from the slice's path/name
                folder, vendor, volume, slice_name = path.split("/")[-4:]
                volume_name = f"{vendor}_{volume}"
                
                # Appends the results per slice, per volume, and per vendor
                slice_results.append([path, *dice_scores, *voxel_counts, total_dice])
                volume_results[volume_name].append((dice_scores, voxel_counts))
                vendor_results[vendor].append((dice_scores, voxel_counts))
    
    # Saves the Dice score per slice
    slice_df = DataFrame(slice_results, columns=["slice", *[f"dice_class_{i}" for i in range(1, number_of_classes + 1)], *[f"voxels_class_{i}" for i in range(1, number_of_classes + 1)], "total_dice"])
    slice_df.to_csv("results/slice_dice.csv", index=False)
    
    # Saves the Dice score per volume
    volume_df = DataFrame(compute_weighted_avg(volume_results, number_of_classes), columns=["volume", *[f"dice_class_{i}" for i in range(1, number_of_classes + 1)], "total_dice"])
    volume_df.to_csv("results/volume_dice.csv", index=False)
    
    # Saves the Dice score per vendor
    vendor_df = DataFrame(compute_weighted_avg(vendor_results, number_of_classes), columns=["vendor", *[f"dice_class_{i}" for i in range(1, number_of_classes + 1)], "total_dice"])
    vendor_df.to_csv("results/vendor_dice.csv", index=False)

# In case it is preferred to run 
# directly in this file, here lays 
# an example
if __name__ == "__main__":
    test_model(
        fold_test=2,
        model_name="UNet",
        weights_name="Run1_UNet_best_model.pth",
        number_of_channels=1,
        number_of_classes=4,
        device_name="GPU",
        batch_size=32
    )
