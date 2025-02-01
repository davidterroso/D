import torch
import wandb
import logging
from os import listdir, cpu_count
from paths import IMAGES_PATH
from pandas import read_csv
from skimage.io import imread
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
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
        model_name,
        device,
        epochs,
        batch_size,
        learning_rate,
        optimizer,
        momentum,
        scheduler,
        number_of_classes,
        number_of_channels,
        fold_test,
        tuning,
        patch_shape, 
        n_pos, 
        n_neg, 
        pos, 
        neg,
        val_percent,
        save_checkpoint,
        load
):
    """
    Function that trains the deep learning models.

    Args:
        model_name (str): name of the model desired to train
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
        momentum (float): momentum of the optimization 
        algorithm
        scheduler (bool): flag that indicates whether a 
        learning rate scheduler will be used or not
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
        val_percent (float): decimal value that represents the 
        percentage of the training set that will be used in the 
        model validation
        save_checkpoint (bool): flag that indicates whether the
        checkpoints are going to be saved or not
        load (str): path that indicates where the model desired 
        to load was saved
    
    Return:
        None
    """

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Dictionary of models, associates a string to a PyTorch module
    models = {
        "UNet": UNet(in_channels=number_of_channels, num_classes=number_of_classes),
        "UNet3": "",
        "2.5D": ""
    }

    # Checks whether the selected model exists or not
    if model_name not in models.keys():
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
 
    # Gets the model selected and indicates in which device it is going 
    # to be trained on and that the memory format is channels_last: 
    # h x w x c
    # instead of
    # c x h x w
    model = models.get(model_name)
    model = model.to(device=torch_device, memory_format=torch.channels_last)

    # Logs the information of the input 
    # and output channels 
    logging.info(
        f"Network\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
    )

    # In case there is a model desired to load indicated by 
    # the presence of the path, it is loaded 
    if load:
        state_dict = torch.load(load, map_location=torch_device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {load}")

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

    # Creates the Dataset object
    dataset = TrainDataset(train_volumes, model_name)

    # Splits the dataset in training and 
    # validation to train the model, with a 
    # fixed seed to allow reproducibility
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    # Registers the information that will be logged
    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {torch_device.type}
    """)

    optimizers_dict = {
        # foreach=True makes the the optimization less time consuming but more memory consuming
        # maximize=False means we are looking to minimize the loss (in case the Dice coefficient 
        # was used instead of Dice loss, for example, this parameter should be True)
        "Adam": optim.Adam(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False), 
        "SGD": optim.SGD(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum),
        "RMSprop": optim.RMSprop(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum)
    }

    # Checks if the optimizer indicated is available and 
    # in case it is not, cancels the run
    if optimizer in optimizers_dict.keys():
        optimizer_torch = optimizers_dict.get(optimizer)
    else:
        print("The requested optimizer is not available. The available options are:")
        for key in optimizers_dict.keys():
            print(key)
        return 0
    
    # In case a learning rate scheduler is going to be used, 
    # it initiates it. Otherwise it is set to None
    if scheduler:
        torch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_torch, "min", patience=5)
    else:
        torch_scheduler = None

    # Initiates a wandb run that allows live visualization online through the 
    # link printed in the command line 
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    experiment.config.update(
         dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint)
    )

    # Iterates through every epoch
    for epoch in range(1, epochs + 1):
        # Eliminates the previous patches and saves 
        # new patches to train the model, but only 
        # for the volumes that will be used in training
        # extractPatches(IMAGES_PATH, 
        #                patch_shape=patch_shape, 
        #                n_pos=n_pos, n_neg=n_neg, 
        #                pos=pos, neg=neg, 
        #                volumes=train_volumes)
        
        # Creates the Dataset object
        dataset = TrainDataset(train_volumes, model)

        # Splits the dataset in training and 
        # validation to train the model, with a 
        # fixed seed to allow reproducibility
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        # Using the Dataset object, creates a DataLoader object 
        # which will be used to train the model in batches
        loader_args = dict(batch_size=batch_size, num_workers=cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        test_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        model.train()
        epoch_loss = 0
    
if __name__ == "__main__":
    train_model(
        model_name="UNet",
        device="GPU",
        epochs=100,
        batch_size=32,
        learning_rate=2e-5,
        optimizer="Adam",
        momentum=0,
        scheduler=False,
        number_of_classes=4,
        number_of_channels=1,
        fold_test=1,
        tuning=True,
        patch_shape=(256,128), 
        n_pos=12, 
        n_neg=2, 
        pos=1, 
        neg=0,
        val_percent=0.1,
        save_checkpoint=True,
        load=False
    )
