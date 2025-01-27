import torch
from os import walk, listdir
from paths import IMAGES_PATH
from pandas import read_csv
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset
from networks.unet import UNet
from init.readOCT import load_oct_image
from init.patchExtraction import extractPatches

def getPatchesFromVolumes(volumes_list, model):
    if model == "2.5D":
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
    else:
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
        
    patches_list = []
    slices_path = images_folder + "slices\\"
    for patch_name in listdir(slices_path):
        volume = patch_name.split("_")[1][-3:]
        volume = int(volume)
        if volume in volumes_list:
            patches_list.append(patch_name)
    return patches_list

class TrainDataset(Dataset):
    def __init__(self, train_volumes, model):
        super().__init__()
        self.patches_names = getPatchesFromVolumes(train_volumes, model)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
    
        if self.model == "2.5D":
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"

        slice_name = images_folder + "slices\\" + self.patches_names[index]
        mask_name = images_folder + "masks\\" + self.patches_names[index]

        scan = imread(slice_name)
        mask = imread(mask_name)

        sample = {"scan": scan, "mask": mask}
        return sample

def train_model (
        model,
        device,
        epochs,
        batch_size,
        learning_rate,
        optimizer,
        number_of_classes,
        number_of_channels,
        fold_test,
        tuning,
        patch_shape, 
        n_pos, 
        n_neg, 
        pos, 
        neg
):
    """
    Function that trains the deep learning models.

    Args:
        model (str): name of the model desired to train
        device (str): indicates whether the network will 
        be trained using the CPU or the GPU
        epochs (int): maximum number of epochs the model 
        will train for
        batch_size (int): size of the batch used in 
        training
        learning_rate (int): learning rate of the 
        optimization function
        optimizer (string): optimization function used
        number_of_classes (int): number of classes the 
        model is supposed to output
        number_of_channels (int): number of channels the 
        input will present
        fold_test (int): number of the fold that will be used 
        in the network testing 
        tuning (bool): indicates whether this train will
        be performed to tune the hyperparameters or not
    
    Return:
        None
    """

    # Dictionary of models, associates a string to a 
    # PyTorch module
    models = {
        "UNet": "",
        "UNet3": "",
        "2.5D": ""
    }

    # Checks whether the selected model exists or not
    if model not in models.keys():
        print("Model not recognized. Possible models:")
        for key in models.keys():
            print(key)
        return 0
    
    # Checks whether the option selected is possible
    if device not in ["CPU", "GPU"]:
        print("Unrecognized device. Possible devices:")
        print("CPU")
        print("GPU")
    elif (device == "GPU"):
        # Checks whether the GPU is available 
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            print("GPU is not available. CPU was selected.")
    elif (device=="CPU"):
        device_name = "cpu"
    # Saves the variable device as torch.device 
    device = torch.device(device_name)

    # Checks if the selected fold for testing exists
    if ((fold_test < 0) or (fold_test > 5)):
        print("There are five folds. Please select one of them.")
        return 0

    # Reads the CSV file that contains the volumes that 
    # will be used to train the network
    df = read_csv("splits/segmentation_train_splits.csv")
    fold_column_name = f"Fold{fold_test}_Volumes"
    train_volumes = df[fold_column_name].dropna().to_list()
    getPatchesFromVolumes(train_volumes, model)

    # for epoch in range(epochs):
    #     extractPatches(IMAGES_PATH, 
    #                    patch_shape=patch_shape, 
    #                    n_pos=n_pos, n_neg=n_neg, 
    #                    pos=pos, neg=neg)
    # dataset = TrainDataset(train_volumes, model)
    
if __name__ == "__main__":
    train_model(
        model="UNet",
        device="GPU",
        epochs=100,
        batch_size=32,
        learning_rate=2e-5,
        optimizer="Adam",
        number_of_classes=4,
        number_of_channels=1,
        fold_test=1,
        tuning=False,
        patch_shape=(256,128), 
        n_pos=12, 
        n_neg=2, 
        pos=1, 
        neg=0
    )
