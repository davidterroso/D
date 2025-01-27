import torch
from os import listdir
from paths import IMAGES_PATH
from pandas import read_csv
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset
from networks.unet import UNet
from init.patchExtraction import extractPatches

def getPatchesFromVolumes(volumes_list, model):
    """
    Used to return the list of all the patches that are available to 
    train the network, knowing which volumes will be used

    Args:
        volumes_list (List[float]): list of the OCT volume's identifier 
        that will be used in training
        model (str): name of the model that will be trained

    Return:
        patches_list (List[str]): list of the name of the patches that 
        will be used to train the model
    """
    # The path to the patches is dependent on the model selected
    if model == "2.5D":
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
    else:
        images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"
        
    # Iterates through the available patches
    # and registers the name of those that are 
    # from the volumes that will be used in 
    # training, returning that list
    patches_list = []
    slices_path = images_folder + "slices\\"
    for patch_name in listdir(slices_path):
        volume = patch_name.split("_")[1][-3:]
        volume = int(volume)
        if volume in volumes_list:
            patches_list.append(patch_name)
    return patches_list

class TrainDataset(Dataset):
    """
    Initiates the PyTorch object Dataset called TrainDataset 
    with the available images, thus simplifying the training
    process
    """
    def __init__(self, train_volumes, model):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in training

        Args: 
            train_volumes(List[float]): list of the training 
            volumes that will be used to train the model
            model (str): name of the model that will be trained
        
        Return:
            None
        """
        super().__init__()
        self.model = model
        self.patches_names = getPatchesFromVolumes(train_volumes, model)

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training
        """
        return len(self.patches_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in training
        when an index is given, utilized to access the list of images
        """
        # In case the index is a tensor,
        # converts it to a list
        if torch.is_tensor(index):
            index = index.tolist()

        # The path to read the images is different depending on the model
        if self.model == "2.5D":
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
        else:
            images_folder = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"

        # Indicates the path to the image depending on the index given,
        # which is associated with the image name
        slice_name = images_folder + "slices\\" + self.patches_names[index]
        mask_name = images_folder + "masks\\" + self.patches_names[index]

        # Reads the image and the
        # fluid mask
        scan = imread(slice_name)
        mask = imread(mask_name)

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
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
    
    Return:
        None
    """

    # Dictionary of models, associates a string to a PyTorch module
    models = {
        "UNet": UNet(in_channels=number_of_channels, num_classes=number_of_classes),
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
    torch_device = torch.device(device_name)

    # Checks if the selected fold for testing exists
    if ((fold_test < 0) or (fold_test > 5)):
        print("There are five folds. Please select one of them.")
        return 0

    # Reads the CSV file that contains the volumes that 
    # will be used to train the network
    df = read_csv("splits/segmentation_train_splits.csv")
    fold_column_name = f"Fold{fold_test}_Volumes"
    train_volumes = df[fold_column_name].dropna().to_list()

    # In case it is desired to tune the model, the test fold (first 
    # during hyperparameter tuning) is not used to test. Instead, the 
    # second will be used to test the changed hyperparameters. Therefore,
    # to evaluate the new hyperparameters on unseen data, the volumes from 
    # the second fold must be removed from the list of training volumes 
    if tuning:
        if fold_test == 1:
            # Reads the CSV file that contains the test splits for segmentation
            df_test = read_csv("splits/segmentation_test_splits.csv")
            # Iterates through the volumes selected to train and removes those
            # that present in the second fold of testing, which will now be used 
            # as evaluation for the new hyperparameters 
            fold_column_test = f"Fold2_Volumes"
            test_volumes = df_test[fold_column_test].dropna().to_list()
            train_volumes = [x for x in train_volumes if x not in test_volumes]        
        else:
            # Condition to check if the tuning was selected for the correct
            # set of folds, which is identified by the fold used in testing
            print("To tune the hyperparameters, please indicate the first fold as test set. The second fold will be used to test.")

    # Iterates through every epoch
    for epoch in range(1, epochs + 1):
        # Eliminates the previous patches and saves 
        # new patches to train the model
        extractPatches(IMAGES_PATH, 
                       patch_shape=patch_shape, 
                       n_pos=n_pos, n_neg=n_neg, 
                       pos=pos, neg=neg)
        # Creates the Dataset object
        dataset = TrainDataset(train_volumes, model)

        # Using the Dataset object, creates a DataLoader object 
        # which will be used to train the model in batches
        if not (model == "2.5D"):
              dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
              dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
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
        tuning=True,
        patch_shape=(256,128), 
        n_pos=12, 
        n_neg=2, 
        pos=1, 
        neg=0
    )
