import csv
import logging
import numpy as np
import random
import torch
import wandb
from IPython import get_ipython
from os import makedirs, remove
from os.path import exists
from pandas import read_csv
from torch import optim
from torch.nn.functional import one_hot, softmax
from init.patch_extraction import extract_patches_wrapper
from network_functions.evaluate import evaluate
from networks.unet25D import TennakoonUNet
from networks.loss import balanced_bce_loss, multiclass_balanced_cross_entropy_loss
from networks.unet import UNet

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
# Dictionary of labels in masks to fluid names
label_to_fluids = {
    1: "IRF",
    2: "SRF",
    3: "PED"
}

def get_class_weights(fluid: str=None):
    """
    Function used to calculate the weights of a determined 
    class using the total number of voxels across a dataset

    Args:
        fluid (str): name of the fluid desired to segment

    Returns:
        binary_weights (PyTorch tensor): weights of 
            background and fluid for the binary segmentation
            with shape [2]
    """
    # Loads the dataframe that contains the information 
    # of all the volumes
    volumes_df = read_csv("splits\\volumes_info.csv")
    # Inititates a list with all possible classes
    fluids = ["Background", "IRF", "SRF", "PED"]
    # Removes from the list the desired fluid
    fluids.remove(fluid)
    # Sums all the voxels across all volumes in the 
    # dataset that are labeled as not the desired class
    num_background_voxels = volumes_df[fluids].sum().sum()
    # Sums all the voxels across all volumes in 
    # the dataset that are labeled as the desired class
    num_fluid_voxels = volumes_df[fluid].sum()

    # Calculates the weights of each class as: N / (2 * N_{c}), where N corresponds to the total 
    # number of voxels in the dataset and N_{c} the total number of voxels of class C in the dataset
    # background_weights = (num_background_voxels + num_fluid_voxels) / (2 * num_background_voxels)
    # fluid_weights = (num_background_voxels + num_fluid_voxels) / (2 * num_fluid_voxels)
    # Calculates the positive weight as the total number 
    # of background voxels divided by the number of fluid voxels
    pos_weight = num_background_voxels / num_fluid_voxels

    # Converts the weights to a PyTorch tensor of shape [2,1,1]
    # binary_weights = torch.tensor([[[background_weights]], [[fluid_weights]]])
    binary_weights = torch.tensor([pos_weight], dtype=torch.float)

    return binary_weights

def train_model (
        run_name: str,
        fold_val: int,
        amp: bool=True,
        assyncronous_patch_extraction: bool=True,
        batch_size: int=32,
        device: str="GPU",
        drop_prob: float=0.75,
        epochs: int=100,
        fluid: str=None,
        fold_test: int=1,
        gradient_clipping: float=1.0,
        learning_rate: float=2e-5,
        model_name: str="UNet",
        momentum: float=0.999,
        n_neg: int=0,
        n_pos: int=12, 
        neg: int=0,
        num_patches: int=4,
        number_of_channels: int=1,
        number_of_classes: int=4,
        optimizer_name: str="Adam",
        patch_dropping: bool=True,
        patch_shape: tuple=(256,128), 
        patch_type: str="vertical",
        patience: int=200,
        patience_after_n: int=0,
        pos: int=1, 
        scheduler: bool=False,
        seed: int=None,
        split: str="competitive_fold_selection.csv",
        tuning: bool=True,
        weight_decay: float=0.0001
):
    """
    Function that trains the deep learning models.

    Args:
        run_name (str): name of the run under which the best model
            will be saved
        fold_val (int): number of the fold that will be used 
            in the network validation 
        amp (bool): bool that indicates whether automatic mixed
            precision is going to be used or not
        assyncronous_patch_extraction (bool): flag that indicates 
            whether patch extraction and dropping is done in each 
            epoch (syncronous) or before the training epochs, 
            once (assyncronous)
        batch_size (int): size of the batch used in 
            training
        device (str): indicates whether the network will 
            be trained using the CPU or the GPU
        drop_prob (float): probability of a non-pathogenic patch
            being dropped
        epochs (int): maximum number of epochs the model 
            will train for
        fluid (str): name of the fluid that is desired to segment 
            in the triple U-Net framework. Default is None because 
            it is not required in other models        
        fold_test (int): number of the fold that will be used 
            in the network testing         
        gradient_clipping (float): threshold after which it
            scales the gradient down, to prevent gradient 
            exploding
        learning_rate (float): learning rate of the 
            optimization function
        model_name (str): name of the model desired to train
        momentum (float): momentum of the optimization 
            algorithm
        n_neg (int): number of patches that will be extracted 
            from the outside of the scan ROI
        n_pos (int): number of patches that will be extracted 
            from the scan ROI
        neg (int): indicates what is the value that does not 
            represent the ROI in the ROI mask
        num_patches (int): number of patches extracted from the 
            images during vertical patch extraction to train the
            model
        number_of_channels (int): number of channels the 
            input will present
        number_of_classes (int): number of classes the 
            model is supposed to output
        optimizer_name (string): optimization function used
        patch_dropping (bool): flag that indicates whether patch
            dropping will be used or not
        patch_type (str): string that indicates what type 
            of patches will be used. Can be "small", where 
            patches of size 256x128 are extracted using the
            extract_patches function, "big", where patches 
            of shape 496x512 are extracted from each image,
            and patches of shape 496x128 are extracted from
            the slices
        patch_shape (tuple): indicates what is the shape of 
            the patches that will be extracted from the B-scans 
        patience (int): number of epochs where the validation 
            errors calculated are worse than the best validation 
            error before terminating training
        patience_after_n (int): number of epochs needed to wait 
            before starting to count the patience. The default 
            value is 0
        pos (int): indicates what is the value that represents 
            the ROI in the ROI mask
        scheduler (bool): flag that indicates whether a 
            learning rate scheduler will be used or not        
        seed (int): indicates the seed that will be used in the 
            random operations of the training. When seed is not
            indicated, the default value is None and the seed is
            not fixed 
        split (str): name of the k-fold split file that will be 
            used in this run
        tuning (bool): indicates whether this train will
            be performed to tune the hyperparameters or not
        weight_decay (float): optimizer's weight decay value
        
    Return:
        None
    """
    # Initiates logging 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Fixes the seed in case it is declared
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Checks whether the option selected is possible
    device = torch.device("cuda" if torch.cuda.is_available() and device == "GPU" else "cpu")
    # Saves the variable device as torch.device 
    print(f"Training on {device}.")

    # Restrictions for the triple U-Net framework
    # Has to be done before model initiation in case the number of 
    # classes is selected incorrectly
    if model_name == "UNet3":
        # Restriction in case no fluid was selected
        if fluid is None:
            print("The fluid desired to segment must be specified.")
            return 0
        # Restriction in case the fluid selected does not exist
        if fluid not in fluids_to_label.keys():
            print("The indicated fluid was not recognized." \
                  "Possible fluids to segment:")
            for key in fluids_to_label.keys():
                print(key)
            return 0
        # Gets the weights of the background and the selected fluid
        class_weights = get_class_weights(fluid).to(device=device)
        # In case none of the other 
        # restrictions has been raised, 
        # the fluid variable will 
        # correspond to the label 
        # in the mask
        fluid = fluids_to_label.get(fluid)
        # The number of classes in this model must always be 2 because 
        # it is a binary segmentation problem
        if number_of_classes != 2:
            print("Because of the selected model, binary will " \
                  "be performed so the number of classes is set to 2")
            number_of_classes = 2
    # Warning in case the model selected does not require fluid but fluid 
    # was selected
    elif fluid is not None:
        print("Model does not require an indication of fluid to segment," \
              "every fluid will be segmented.")

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

    # Reads the CSV file that contains the volumes that 
    # will be used to train and validate the network
    df = read_csv(f"splits/{split}")
    # Initiates the list of the volumes used to train
    train_volumes = []
    # Only in case the network's hyperparameters are being 
    # tuned, validation fold is created 
    if tuning:
        # Initiates the list of the volumes used to validate
        val_volumes = []
        # Iterates through all the columns in the DataFrame
        for col in df.columns:
            # In case the column number does not match with either 
            # the validation or testing fold number, its volumes 
            # are added to the list of the train volumes
            if (col != str(fold_test)) and (col != str(fold_val)):
                train_volumes = train_volumes + df[col].dropna().to_list()
            # In case the column number matches the validation 
            # fold number, the volumes are added to the list of 
            # validation volumes
            if (col == str(fold_val)):
                val_volumes = val_volumes + df[col].dropna().to_list()
    # In case the training is not for the tuning of the network 
    # parameters, no list of validation volumes is created
    else:
        val_volumes = None
        for col in df.columns:
            if (col != str(fold_test)):
                train_volumes = train_volumes + df[col].dropna().to_list()
    
    # Registers the information that will be logged
    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
        Mixed Precision: {amp}
        Seed:            {seed}
    """)

    # Dictionary that contains the optimizers 
    # that can be used in this network
    optimizers_dict = {
        # foreach=True makes the the optimization less time consuming but more memory consuming
        # maximize=False means we are looking to minimize the loss (in case the Dice coefficient 
        # was used instead of Dice loss, for example, this parameter should be True)
        "Adam": optim.Adam(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, weight_decay=weight_decay), 
        "SGD": optim.SGD(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum, weight_decay=weight_decay),
        "RMSprop": optim.RMSprop(params=model.parameters(), lr=learning_rate, foreach=True, maximize=False, momentum=momentum, weight_decay=weight_decay)
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
    # experiment = wandb.init(project="U-Net", name=run_name, entity="davidterroso19")

    # Indicates what configurations are going to be saved in the run
    # experiment.config.update(
    #      dict(epochs=epochs, batch_size=batch_size, 
    #           learning_rate=learning_rate, amp=amp)
    # )

    # Creates two CSV log file for the run, one for the training 
    # loss per batch and the other for the training and validation 
    # loss per epoch 
    csv_epoch_filename = f"logs\{run_name}_training_log_epoch.csv"
    csv_batch_filename = f"logs\{run_name}_training_log_batch.csv"

    # Creates the folder logs in case 
    # it does not exist yet
    makedirs("logs", exist_ok=True)

    # In case the files desired to write the logs do not exist yet,
    # they are created
    if not (exists(csv_epoch_filename) and exists(csv_batch_filename)):
        with open(csv_epoch_filename, mode="w", newline="") as file:
            # Creates the log file for the loss per epoch, initiating the columns
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Epoch Training Loss", "Epoch Validation Loss"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            # Creates the log file for the loss per batch, initiating the columns
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Training Loss"])
    else:
        # Deletes the files before writting them
        remove(csv_epoch_filename)
        remove(csv_batch_filename)
        with open(csv_epoch_filename, mode="w", newline="") as file:
            # Creates the log file for the loss per epoch, initiating the columns
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Epoch Training Loss", "Epoch Validation Loss"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            # Creates the log file for the loss per batch, initiating the columns
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Training Loss"])

    # Checks if the selected patch type is one of the expected
    # and in case the selected is note expected prints the ones 
    # that are available
    if patch_type not in ["small", "big", "vertical"]:
        print("Patch type not recognized. Available patch_type values:")
        for patch_type_name in ["small", "big", "vertical"]:
            print(patch_type_name)

    # In case patch extraction is done before training
    if (assyncronous_patch_extraction):
        train_loader, val_loader, n_train, n_val = extract_patches_wrapper(
            model_name=model_name, patch_type=patch_type, patch_shape=patch_shape, n_pos=n_pos, 
            n_neg=n_neg, pos=pos, neg=neg, train_volumes=train_volumes, 
            val_volumes=val_volumes, batch_size=batch_size, 
            patch_dropping=patch_dropping, drop_prob=drop_prob, 
            num_patches=num_patches, seed=seed, fold_val=fold_val, 
            fold_test=fold_test, fluid=fluid, number_of_channels=number_of_channels, 
            number_of_classes=number_of_classes)

    # Initiates the counter of patience
    patience_counter = 0
    # Initiates the best validation loss as an infinite value
    best_val_loss = float("inf")
    # Initiates the global step counter
    global_step = 0
    # Iterates through every epoch
    for epoch in range(1, epochs + 1):
        # In case the patch extraction is done syncronously
        if ((not assyncronous_patch_extraction) and (patch_type=="small")):
            print(f"Preparing epoch {epoch} training")
            train_loader, val_loader, n_train, n_val = extract_patches_wrapper(
                model_name=model_name, patch_type=patch_type, patch_shape=patch_shape, n_pos=n_pos, 
                n_neg=n_neg, pos=pos, neg=neg, train_volumes=train_volumes, 
                val_volumes=val_volumes, batch_size=batch_size, 
                patch_dropping=patch_dropping, drop_prob=drop_prob, seed=seed, fold_val=fold_val, 
                fold_test=fold_test, fluid=fluid, number_of_channels=number_of_channels, 
                number_of_classes=number_of_classes)

        # Indicates the model that it is going to be trained
        model.train()
        # Initiates the loss of the current epoch as 0
        epoch_loss = 0

        print(f"Training Epoch {epoch}")
        # Creates a progress bar using tqdm. The limit is when all the images in training are 
        # used in that epoch, the description indicates in which epoch the training is being 
        # done, and the unit indicates the unit that is filling the bar
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img", leave=True, position=0) as progress_bar:
            # Iterates through the 
            # batches of images
            for batch_num, batch in enumerate(train_loader):
                # From the DataLoader object extracts the scan 
                # and the GT mask
                images, true_masks = batch["scan"], batch["mask"]

                ######## USED TO DISPLAY IMAGES WHEN DEBUGGING #########
                # # Iterates through the images in the batch
                # for im in range(batch_size):
                #     oct_image = images.numpy()[im][0]
                #     mask = true_masks.numpy()[im][0]
                #     # Converts each voxel classified as background to 
                #     # NaN so that it will not appear in the overlaying
                #     # mask
                #     mask = mask.astype(float)
                #     mask[mask == 0] = np.nan

                #     # The voxels classified in "IRF", "SRF", and "PED" 
                #     # will be converted to color as Red for IRF, green 
                #     # for SRF, and blue for PED
                #     fluid_colors = ["red", "green", "blue"]
                #     fluid_cmap = mcolors.ListedColormap(fluid_colors)
                #     # Declares in which part of the color bar each
                #     # label is going to be placed
                #     fluid_bounds = [1, 2, 3, 4]
                #     # Normalizes the color map according to the 
                #     # bounds declared.
                #     fluid_norm = mcolors.BoundaryNorm(fluid_bounds, fluid_cmap.N)

                #     # Saves the OCT scan with an overlay of the ground-truth masks
                #     plt.figure(figsize=(oct_image.shape[1] / 100, oct_image.shape[0] / 100))
                #     plt.imshow(oct_image, cmap=plt.cm.gray)
                #     plt.imshow(mask, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
                #     plt.axis("off")
                #     plt.show()

                #     # Closes the figure
                #     plt.clf()
                #     plt.close("all")

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
                # Calculates loss
                with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=amp):
                    # Predicts the masks of the received images
                    masks_pred = model(images)
                    # Performs one hot encoding on the true masks, in channels last format
                    masks_true_one_hot = one_hot(true_masks.long(), model.n_classes).float().squeeze(1)

                    if model_name != "UNet3": 
                        # Performs softmax on the predicted masks
                        # dim=1 indicates that the softmax is calculated 
                        # across the masks, since the channels is the first 
                        # dimension
                        masks_pred_prob_bchw = softmax(masks_pred, dim=1).float()
                        # Permute changes the images from channels first to channels last
                        masks_pred_prob = masks_pred_prob_bchw.permute(0, 2, 3, 1)
                        # Calculates the balanced loss for the background mask
                        loss = multiclass_balanced_cross_entropy_loss(
                                            model_name=model_name,
                                            y_true=masks_true_one_hot,
                                            y_pred=masks_pred_prob, 
                                            batch_size=images.shape[0], 
                                            n_classes=model.n_classes, 
                                            eps=1e-7)
                    else:
                        # loss = balanced_bce_loss(y_true=masks_true_one_hot,
                        #                          y_pred=masks_pred_prob, 
                        #                          batch_size=images.shape[0], 
                        #                          n_classes=model.n_classes, 
                        #                          eps=1e-7)
                        # Permute changes the images from channels first to channels last
                        # masks_true_one_hot_cf = masks_true_one_hot.permute(0, 3, 1, 2) 
                        true_masks = true_masks.unsqueeze(1).float()
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
                        # loss = criterion(masks_pred, masks_true_one_hot_cf)
                        loss = criterion(masks_pred, true_masks)

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
                # and epoch loss
                global_step += 1
                epoch_loss += loss.item()
                # Adds the loss of the batch at the end of the progress bar
                progress_bar.set_postfix(**{"Loss (batch)": loss.item()})

                # Writes the batch loss in the CSV file
                with open(csv_batch_filename, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, batch_num, loss.item()])

        print(f"Validating Epoch {epoch}")
        #################### THIS IS ONLY USED FOR DEBUGGING ########################
        # histograms = {}
        # # Iterates through the model parameters and creates histograms for all of 
        # # those that do not have infinite or zero values
        # for tag, value in model.named_parameters():
        #     # Name matching so that it 
        #     # can be read and saved properly
        #     tag = tag.replace("/", ".")
        #     if not (torch.isinf(value) | torch.isnan(value)).any():
        #         # Calculates the histogram of the weight using the CPU
        #         histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
        #         # Calculates the histogram of the gradient using the CPU
        #         histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

        # Calculates the validation score for 
        # the model, in case tuning is being done
        if tuning:
            val_loss = evaluate(model_name, model, val_loader, device, amp, n_val, class_weights)
        else:
            val_loss = 0
        
        # In case a scheduler is used, the
        # learning rate is adjusted accordingly
        if scheduler and tuning:
            torch_scheduler.step(val_loss)

        # Adds the validation score to the logging, 
        # in case tuning is being done
        if tuning:
            logging.info(f"Validation Mean Loss: {val_loss}")

        # Writes the loss of an epoch in training and validation
        with open(csv_epoch_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss / len(train_loader), val_loss])

        # Early stopping check
        # If the validation loss is better 
        # than the previously best obtained, 
        # saves the model as a PyTorch (.pth) file
        # and resets the patience counter
        # Only when tuning is being done, the
        # best models are saved
        if tuning:
            if val_loss < best_val_loss:
                # Creates the folder models in case 
                # it does not exist yet
                makedirs("models", exist_ok=True)
                best_val_loss = val_loss
                patience_counter = 0
                # File is saved with a name that depends on the argument input, the name 
                # of the model, and fluid desired to segment in case it exists
                if model_name != "UNet3":
                    torch.save(model.state_dict(),
                                f"models/{run_name}_{model_name}_best_model.pth")
                else:
                    torch.save(model.state_dict(), 
                            f"models/{run_name}_{label_to_fluids.get(fluid)}_{model_name}_best_model.pth")
                print("Model saved.")
            # In case the model has not 
            # obtained a better performance, 
            # the patience counter increases
            else:
                patience_counter += 1
        
        # In case the number of epochs after which no 
        # improvement has been made surpasses the 
        # patience value, the model stops training
        # Only when tuning is being done, the
        # early stopage can be triggered
        if patience_counter >= patience and tuning and epoch > patience_after_n:
            logging.info("Early stopping triggered.")
            break
        
        # Resets the patience counter every epoch 
        # below the number after which it starts counting
        if epoch <= patience_after_n:
            patience_counter = 0

        # This section of the code is commented because, after implementation, 
        # no real use was given to the output images and was only slowing down 
        # the process and occupying storage space
        # Get the predictions in each voxel
        # pred_mask = masks_pred_prob_bchw.argmax(dim=1)
        ############################ THIS IS ONLY USED FOR DEBUGGING ###############################
        # # In case the model selected is not the "UNet3"
        # if model_name != "UNet3":
        #     # This section of the code is commented because, after implementation, 
        #     # no real use was given to the output images and was only slowing down 
        #     # the process and occupying storage space
        #     # Get the predicted masks
        #     # irf_predicted_mask = (pred_mask == 1).float()  
        #     # srf_predicted_mask = (pred_mask == 2).float()
        #     # ped_predicted_mask = (pred_mask == 3).float()
        #     # # Get the true masks
        #     # irf_true_mask = (true_masks == 1).float()
        #     # srf_true_mask = (true_masks == 2).float()
        #     # ped_true_mask = (true_masks == 3).float()

        #     # Attempts to log this information
        #     try:
        #         # Logs the information in the wandb session
        #         experiment.log({
        #             "Step": global_step,
        #             "Learning Rate": optimizer.param_groups[0]["lr"],
        #             "Validation Mean Loss": val_loss,
        #             # This section of the code is commented because, after implementation, 
        #             # no real use was given to the output images and was only slowing down 
        #             # the process and occupying storage space
        #             # "Images": wandb.Image(images[0].cpu() * 128 + 128), # Calculations 
        #             #                                                     # to revert the 
        #             #                                                     # standardization
        #             # "Masks":{
        #             #     # Multiplied by 255 to visualize
        #             #     "IRF True Mask": wandb.Image(irf_true_mask[0].float().cpu() * 255),
        #             #     "SRF True Mask": wandb.Image(srf_true_mask[0].float().cpu() * 255),
        #             #     "PED True Mask": wandb.Image(ped_true_mask[0].float().cpu() * 255),
        #             #     "IRF Predicted Mask": wandb.Image(irf_predicted_mask[0].float().cpu() * 255),
        #             #     "SRF Predicted Mask": wandb.Image(srf_predicted_mask[0].float().cpu() * 255),
        #             #     "PED Predicted Mask": wandb.Image(ped_predicted_mask[0].float().cpu() * 255),
        #             # },
        #             "Step": global_step,
        #             "Epoch": epoch,
        #             **histograms
        #         })
        #     # In case something goes wrong, 
        #     # the program does not crash but 
        #     # does not save the information 
        #     except:
        #         pass 

        # else:
        #     # This section of the code is commented because, after implementation, 
        #     # no real use was given to the output images and was only slowing down 
        #     # the process and occupying storage space
        #     # Get the predicted masks
        #     # fluid_predicted_mask = (pred_mask == 1).float()
        #     # Get true masks
        #     # fluid_true_mask = (true_masks == 1).float()

        #     # Attempts to log this information
        #     try:
        #         # Logs the information in the wandb session
        #         experiment.log({
        #             "Learning Rate": optimizer.param_groups[0]["lr"],
        #             "Validation Mean Loss": val_loss,
        #             # This section of the code is commented because, after implementation, 
        #             # no real use was given to the output images and was only slowing down 
        #             # the process and occupying storage space
        #             # "Images": wandb.Image(images[0].cpu() * 128 + 128), # Calculations 
        #             #                                                     # to revert the 
        #             #                                                     # standardization
        #             # "Masks":{
        #             #     # Multiplied by 255 to visualize
        #             #     f"{label_to_fluids.get(fluid)} True Mask": 
        #             #     wandb.Image(fluid_true_mask[0].float().cpu() * 255),

        #             #     f"{label_to_fluids.get(fluid)} Predicted Mask": 
        #             #     wandb.Image(fluid_predicted_mask[0].float().cpu() * 255),
        #             # },
        #             "Step": global_step,
        #             "Epoch": epoch,
        #             **histograms
        #         })
        #     # In case something goes wrong, 
        #     # the program does not crash but 
        #     # does not save the information 
        #     except:
        #         pass  
    # wandb.finish()
