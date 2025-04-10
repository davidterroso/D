import torch
from IPython import get_ipython
from os import makedirs
from os.path import exists
from pandas import read_csv
from PIL import Image
from torch.utils.data import DataLoader
from shutil import rmtree
from init.read_oct import load_oct_image
from networks.unet25D import TennakoonUNet
from networks.unet import UNet
from network_functions.dataset import TestDataset
from paths import IMAGES_PATH, RETOUCH_PATH
from test_model import collate_fn

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

def infer(
        fold_test: int,
        model_name: str,
        weights_name: str,
        device_name: str,
        number_of_channels: int,
        number_of_classes: int,
        batch_size: int,
        patch_type: str,
        resize_images: bool=False,
        fluid: int=None
    ):
    """
    Function used to infer the predictions using the trained models

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
        resize_images (bool): flag that indicates whether the images 
            will be resized or not in testing 
        fluid (int): integer that is associated with the label desired 
            to segment
            
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
    test_dataset = TestDataset(test_volumes, model_name, patch_type, resize_images, resize_shape, fluids_to_label.get(fluid))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, collate_fn=collate_fn)

    # Extracts the name of the run from 
    # the name of the weights file
    run_name = weights_name.split("_")[0]

    # Declares the name of the folder in which the images will be saved
    if not resize_images:
        # In case the folder to save exists, it is deleted and created again
        folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}_predicted_masks\\"
        if exists(folder_to_save):
            rmtree(folder_to_save)
            makedirs(folder_to_save)
        else:
            makedirs(folder_to_save)
    else:
        # In case the folder to save exists, it is deleted and created again
        folder_to_save = IMAGES_PATH + f"\\OCT_images\\segmentation\\predictions\\{run_name}_resized_predicted_masks\\"
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

                # Saves PyTorch tensor as an image
                img = Image.fromarray(preds.cpu().numpy())
                img.save(str(folder_to_save + "\\" + image_name))

                # Update the progress bar
                progress_bar.update(1)