import numpy as np
import logging
import torch
import tqdm
import wandb
from os import listdir, cpu_count
from pathlib import Path
from paths import IMAGES_PATH
from pandas import read_csv
from skimage.io import imread
from torch import optim
from torch.nn import BCELoss
from torch.nn.functional import softmax, one_hot
from torch.utils.data import Dataset, DataLoader, random_split
from init.patchExtraction import extractPatches
from networks.loss import multiclass_balanced_cross_entropy_loss
from networks.unet import UNet

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

        # Expands the scan dimentions to 
        # include an extra channel of value 1
        # as the first channel
        scan = np.expand_dims(scan, axis=0)

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask}
        return sample

def train_model (
        model_name,
        device_name,
        epochs,
        batch_size,
        learning_rate,
        optimizer_name,
        momentum,
        gradient_clipping,
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
        load,
        amp
):
    """
    Function that trains the deep learning models.

    Args:
        model_name (str): name of the model desired to train
        device_name (str): indicates whether the network will 
        be trained using the CPU or the GPU
        epochs (int): maximum number of epochs the model 
        will train for
        batch_size (int): size of the batch used in 
        training
        learning_rate (int): learning rate of the 
        optimization function
        optimizer_name (string): optimization function used
        number_of_classes (int): number of classes the 
        model is supposed to output
        momentum (float): momentum of the optimization 
        algorithm
        gradient_clipping (float): threshold after which it
        scales the gradient down, to prevent gradient exploding
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
        load (str): path that indicates where the model desired 
        to load was saved
        amp (bool): bool that indicates whether automatic mixed
        precision is going to be used or not 
        
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
 
    # Gets the model selected and indicates in which device it is going 
    # to be trained on and that the memory format is channels_last: 
    # h x w x c
    # instead of
    # c x h x w
    model = models.get(model_name)
    model = model.to(device=device, memory_format=torch.channels_last)

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
        state_dict = torch.load(load, map_location=device)
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
            print("To tune the hyperparameters, please indicate the first \
                  fold as test set. The second fold will be used to test.")

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
        Device:          {device.type}
        Mixed Precision: {amp}
    """)

    optimizers_dict = {
        # foreach=True makes the the optimization less time consuming but more memory consuming
        # maximize=False means we are looking to minimize the loss (in case the Dice coefficient 
        # was used instead of Dice loss, for example, this parameter should be True)
        "Adam": optim.Adam(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False), 
        "SGD": optim.SGD(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum),
        "RMSprop": optim.RMSprop(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum)
    }

    # Checks if the optimizer_name indicated is available and 
    # in case it is not, cancels the run
    if optimizer_name in optimizers_dict.keys():
        optimizer = optimizers_dict.get(optimizer_name)
    else:
        print("The requested optimizer_name is not available. The available options are:")
        for key in optimizers_dict.keys():
            print(key)
        return 0
    
    # In case a learning rate scheduler is going to be used, 
    # it initiates it. Otherwise it is set to None
    if scheduler:
        torch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    else:
        torch_scheduler = None

    # Initiates the grad scaler in case mixed 
    # precision is used
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # Initiates a wandb run that allows live visualization online through the 
    # link printed in the command line 
    # "project" argument indicates the name of the project
    # "resume" indicates that it is possible to continue a previous run if so 
    # is indicated
    # "anonymous" indicates that the run will always be done anonymously, 
    # independently of whether the user is signed in or not
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    # Indicates what configurations are going to be saved in the run
    experiment.config.update(
         dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, amp=amp)
    )

    global_step = 0
    # Iterates through every epoch
    for epoch in range(1, epochs + 1):
        # Eliminates the previous patches and saves 
        # new patches to train the model, but only 
        # for the volumes that will be used in training
        extractPatches(IMAGES_PATH, 
                       patch_shape=patch_shape, 
                       n_pos=n_pos, n_neg=n_neg, 
                       pos=pos, neg=neg, 
                       volumes=train_volumes)
        
        # Creates the Dataset object
        dataset = TrainDataset(train_volumes, model)

        # Splits the dataset in training and 
        # validation to train the model, with a 
        # fixed seed to allow reproducibility
        n_val = int(dataset.__len__ * val_percent)
        n_train = dataset.__len__ - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        # Using the Dataset object, creates a DataLoader object 
        # which will be used to train the model in batches
        loader_args = dict(batch_size=batch_size, num_workers=cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

        # Indicates the model that it is going to be trained
        model.train()
        # Initiates the loss of the current epoch as 0
        epoch_loss = 0

        # Creates a progress bar using tqdm. The limit is when all the images in training are 
        # used in that epoch, the description indicates in which epoch the training is being 
        # done, and the unit indicates the unit that is filling the bar
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as progress_bar:
            # Iterates through the 
            # batches of images
            for batch in train_loader:
                # From the DataLoader object extracts the scan 
                # and the GT mask
                images, true_masks = batch["scan"], batch["mask"]

                # Checks if the number of channels given as input matches the number of 
                # images read with the dataloader
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check if ' \
                    'the images are loaded correctly.'

                # Declares what type the images and the true_masks variables, including the device that is
                # going to be used, the data type and whether it is channels first or channels last
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Allows for mixed precision calculations, attributes a device to be used in 
                # these calculations
                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    # Predicts the masks of the received images
                    masks_pred = model(images)
                    # Performs softmax on the predicted masks
                    # dim=1 indicates that the softmax is calculated 
                    # across the masks, since the channels is the first 
                    # dimension
                    masks_pred_prob = softmax(masks_pred, dim=1).float()
                    # Performs one hot encoding on the true masks, in channels last format
                    masks_true_one_hot = one_hot(true_masks, model.n_classes).float()

                    # Calculates the balanced loss for the background mask
                    # Permute changes the images from channels first to channels last
                    background_loss = multiclass_balanced_cross_entropy_loss(
                                        y_true=masks_true_one_hot,
                                        y_pred=masks_pred_prob.permute(0, 3, 1, 2), 
                                        batch_size=images.shape[0], 
                                        n_classes=number_of_classes, 
                                        eps=1e-7)
                    # Calculates the loss for the IRF mask
                    irf_loss = BCELoss(masks_pred_prob[:, 1, :, :], masks_true_one_hot[:, 1, :, :])
                    # Calculates the loss for the SRF mask
                    srf_loss = BCELoss(masks_pred_prob[:, 2, :, :], masks_true_one_hot[:, 2, :, :])
                    # Calculates the loss for the PED mask
                    ped_loss = BCELoss(masks_pred_prob[:, 3, :, :], masks_true_one_hot[:, 3, :, :])
                    # Calculates the total loss as the sum of all losses
                    loss = background_loss + irf_loss + srf_loss + ped_loss

                # Saves the value that are zero as 
                # None so that it saves memory
                optimizer.zero_grad(set_to_none=True)
                # Acumulates scaled gradients
                grad_scaler.scale(loss).backward()
                # Unscales the gradients so that 
                # they can be clipped
                grad_scaler.unscale_(optimizer)
                # Clips the gradients above the threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # Updates the parameters 
                # based on the current gradient
                grad_scaler.step(optimizer)
                # Updates the scale 
                # for the next iteration
                grad_scaler.update()

                # Updates the progress bar by indicating 
                # how many images have been trained
                progress_bar.update(images.shape[0])
                # Updates the global step
                # and global loss
                global_step += 1
                epoch_loss += loss.item()
                # Logs the loss, the step, and 
                # the epochs on the wandb
                experiment.log({
                    "train_loss": loss.item(),
                    "step": global_step,
                    "epoch": epoch
                })
                # Adds the loss of the batch at the end of the progress bar
                progress_bar.set_postfix(**{"Loss (batch)": loss.item()})

                # The evaluation of the model is done five times per epoch
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        # Iterates through the model parameters and creates histograms for all of 
                        # those that do not have infinite or zero values
                        for tag, value in model.named_parameters():
                            # Name matching so that it 
                            # can be read and saved properly
                            tag = tag.replace("/", ".")
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                # Calculates the histogram of the weight using the CPU
                                histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                # Calculates the histogram of the gradient using the CPU
                                histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

                        # Calculates the validation score for the model
                        val_score = evaluate(model, val_loader, device, amp)
                        
                        # In case a scheduler is used, the
                        # learning rate is adjusted accordingly
                        if scheduler:
                            torch_scheduler.step(val_score)

                        # Adds the validation score to the logging
                        logging.info("Validation Dice Score: {}".format(val_score))
                        # Attempts to log this information
                        try:
                            # Logs the information in the wandb session
                            experiment.log({
                                "Learning Rate": optimizer.param_groups[0]["lr"],
                                "Validation Dice": val_score,
                                "Images": wandb.Image(images[0].cpu()),
                                "Masks":{
                                    "True": wandb.Image(true_masks[0].float().cpu()),
                                    "Prediction": wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                "Step": global_step,
                                "Epoch": epoch,
                                **histograms
                            })
                        # In case something goes wrong, 
                        # the program does not crash but 
                        # does not save the information 
                        except:
                            pass  
    
if __name__ == "__main__":
    train_model(
        model_name="UNet",
        device_name="GPU",
        epochs=100,
        batch_size=32,
        learning_rate=2e-5,
        optimizer_name="Adam",
        momentum=0.999,
        gradient_clipping=1.0,
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
        load=False,
        amp=True
    )
