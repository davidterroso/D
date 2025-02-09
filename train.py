import logging
import torch
import wandb
from numpy.random import choice, seed
from os import cpu_count
from pandas import read_csv
from time import time
from torch import optim
from torch.nn.functional import one_hot, softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from init.patchExtraction import extractPatches, extractPatches25D
from network_functions.dataset import TrainDataset, ValidationDataset, dropPatches
from network_functions.evaluate import evaluate
from networks.unet25D import TennakoonUNet
from networks.loss import multiclass_balanced_cross_entropy_loss
from networks.unet import UNet
from paths import IMAGES_PATH

def train_model (
        run_name,
        model_name,
        device_name,
        epochs,
        batch_size,
        learning_rate,
        optimizer_name,
        momentum,
        weight_decay,
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
        amp,
        patience,
        fluid=None,
):
    """
    Function that trains the deep learning models.

    Args:
        run_name (str): name of the run under which the best model
            will be saved
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
        weight_decay (float): optimizer's weight decay value
        gradient_clipping (float): threshold after which it
            scales the gradient down, to prevent gradient 
            exploding
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
            percentage of the training set that will be used in 
            the model validation
        amp (bool): bool that indicates whether automatic mixed
            precision is going to be used or not
        patience (int): number of epochs where the validation 
            errors calculated are worse than the best validation 
            error before terminating training
        fluid (str): name of the fluid that is desired to segment 
            in the triple U-Net framework. Default is None because 
            it is not required in other models
        
    Return:
        None
    """
    # Initiates logging 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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

    # Checks if the selected fold for testing exists
    if ((fold_test < 0) or (fold_test > 5)):
        print("There are five folds. Please select one of them.")
        return 0

    # Reads the CSV file that contains the volumes that 
    # will be used to train the network
    df = read_csv("splits/segmentation_train_splits.csv")
    fold_column_name = f"Fold{fold_test}_Volumes"
    initial_train_volumes = df[fold_column_name].dropna().to_list()

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
            initial_train_volumes = [x for x in initial_train_volumes if x not in test_volumes]        
        else:
            # Condition to check if the tuning was selected for the correct
            # set of folds, which is identified by the fold used in testing
            print("To tune the hyperparameters, please indicate the first \
                  fold as test set. The second fold will be used to test.")

    # Converts the list from float to int
    initial_train_volumes = [int(x) for x in initial_train_volumes]

    # Gets the number of validation volumes and the number 
    # of training volumes that will be used
    val_size = int(len(initial_train_volumes) * val_percent)
    train_size = len(initial_train_volumes) - val_size

    # Declares the used seed to promote reproducibility
    seed(0)

    # Gets the list of the train and validation volumes that will be used
    train_volumes_list = list(choice(initial_train_volumes, train_size, replace=False))
    val_volumes_list = [x for x in initial_train_volumes if x not in train_volumes_list] 

    # Creates the Dataset object, but is just used to get the 
    # number of slices used, not taking into account the dropped 
    # patches
    train_dataset = TrainDataset(train_volumes_list, model_name, fluid)
    val_dataset = ValidationDataset(val_volumes_list, model_name, fluid)

    # Gets the number of images 
    # in the train and validation dataset
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # Registers the information that will be logged
    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
        Mixed Precision: {amp}
    """)

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
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    # Indicates what configurations are going to be saved in the run
    experiment.config.update(
         dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, amp=amp)
    )

    global_step = 0
    # Iterates through every epoch
    for epoch in range(1, epochs + 1):
        print(f"Preparing epoch {epoch} training")
        print("Extracting patches")

        # Starts timing the patch extraction
        begin = time()

        # Eliminates the previous patches and saves 
        # new patches to train and validate the model, 
        # but only for the volumes that will be used 
        # in training
        if model_name != "2.5D":
            extractPatches(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=train_volumes_list)            
            
            extractPatches(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=val_volumes_list)
        else:
            extractPatches25D(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=train_volumes_list)            
            
            extractPatches25D(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=val_volumes_list)
        
        # Stops timing the patch extraction and prints it
        end = time()
        print(f"Patch extraction took {end - begin} seconds.")

        print("Dropping patches")
        # Starts timing the patch dropping
        begin = time()
        # Randomly drops patches of slices that do not have retinal fluid
        dropPatches(prob=0.75, volumes_list=train_volumes_list, model=model_name)
        dropPatches(prob=0.75, volumes_list=val_volumes_list, model=model_name)
        # Stops timing the patch extraction and prints it
        end = time()
        print(f"Patch dropping took {end - begin} seconds.")
        
        # Creates the train and validation Dataset objects
        # The validation dataset does not apply transformations
        train_set = TrainDataset(train_volumes_list, model_name)
        val_set = ValidationDataset(val_volumes_list, model_name)

        n_train = len(train_set)
        n_val = len(val_set)
        print(f"Train Images: {n_train} | Validation Images: {n_val}")

        # Using the Dataset object, creates a DataLoader object 
        # which will be used to train the model in batches
        loader_args = dict(batch_size=batch_size, num_workers=cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

        # Indicates the model that it is going to be trained
        model.train()
        # Initiates the loss of the current epoch as 0
        epoch_loss = 0
        # Initiates the best validation loss as an infinite value
        best_val_loss = float("inf")
        # Initiates the counter of patience
        patience_counter = 0

        print(f"Training Epoch {epoch}")
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
                # Calculates loss
                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    # Predicts the masks of the received images
                    masks_pred = model(images)
                    # Performs softmax on the predicted masks
                    # dim=1 indicates that the softmax is calculated 
                    # across the masks, since the channels is the first 
                    # dimension
                    masks_pred_prob = softmax(masks_pred, dim=1).float()
                    # Permute changes the images from channels first to channels last
                    masks_pred_prob = masks_pred_prob.permute(0, 2, 3, 1)
                    # Performs one hot encoding on the true masks, in channels last format
                    masks_true_one_hot = one_hot(true_masks.long(), model.n_classes).float()

                    # Calculates the balanced loss for the background mask
                    loss = multiclass_balanced_cross_entropy_loss(
                                        y_true=masks_true_one_hot,
                                        y_pred=masks_pred_prob, 
                                        batch_size=images.shape[0], 
                                        n_classes=model.n_classes, 
                                        eps=1e-7)

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
                # Logs the loss, the step, and 
                # the epochs on the wandb
                experiment.log({
                    "train_loss": loss.item(),
                    "step": global_step,
                    "epoch": epoch
                })
                # Adds the loss of the batch at the end of the progress bar
                progress_bar.set_postfix(**{"Loss (batch)": loss.item()})

        print(f"Validating Epoch {epoch}")
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
        val_loss = evaluate(model, val_loader, device, amp)
        
        # In case a scheduler is used, the
        # learning rate is adjusted accordingly
        if scheduler:
            torch_scheduler.step(val_loss)

        # Adds the validation score to the logging
        logging.info(f"Validation Mean Loss: {val_loss}")

        # Early stopping check
        # If the validation loss is better 
        # than the previously best obtained, 
        # saves the model as a PyTorch (.pth) file
        # and resets the patience counter
        if val_loss < best_val_loss:
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
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

        # Get the predictions in each voxel
        pred_mask = masks_pred_prob.argmax(dim=1)

        if model_name != "UNet3":
            # Get the predicted masks
            irf_predicted_mask = (pred_mask == 1).float()  
            srf_predicted_mask = (pred_mask == 2).float()
            ped_predicted_mask = (pred_mask == 3).float()
            # Get the true masks
            irf_true_mask = (true_masks == 1).float()
            srf_true_mask = (true_masks == 2).float()
            ped_true_mask = (true_masks == 3).float()

            # Attempts to log this information
            try:
                # Logs the information in the wandb session
                experiment.log({
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "Validation Mean Loss": val_loss,
                    "Images": wandb.Image(images[0].cpu()),
                    "Masks":{
                        "IRF True Mask": wandb.Image(irf_true_mask[0].float().cpu()),
                        "SRF True Mask": wandb.Image(srf_true_mask[0].float().cpu()),
                        "PED True Mask": wandb.Image(ped_true_mask[0].float().cpu()),
                        "IRF Predicted Mask": wandb.Image(irf_predicted_mask[0].float().cpu()),
                        "SRF Predicted Mask": wandb.Image(srf_predicted_mask[0].float().cpu()),
                        "PED Predicted Mask": wandb.Image(ped_predicted_mask[0].float().cpu()),
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

        else:
            # Get the predicted masks
            fluid_predicted_mask = (pred_mask == 1).float()
            # Get true masks
            fluid_true_mask = (true_masks == 1).float()

            # Attempts to log this information
            try:
                # Logs the information in the wandb session
                experiment.log({
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "Validation Mean Loss": val_loss,
                    "Images": wandb.Image(images[0].cpu()),
                    "Masks":{
                        f"{label_to_fluids.get(fluid)} True Mask": 
                        wandb.Image(fluid_true_mask[0].float().cpu()),

                        f"{label_to_fluids.get(fluid)} Predicted Mask": 
                        wandb.Image(fluid_predicted_mask[0].float().cpu()),
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
            
# In case it is preferred to run 
# directly in this file, here lays 
# an example
if __name__ == "__main__":
    train_model(
        run_name="Run1",
        model_name="UNet",
        device_name="GPU",
        epochs=100,
        batch_size=32,
        learning_rate=2e-5,
        optimizer_name="Adam",
        momentum=0.999,
        weight_decay=0.0001,
        gradient_clipping=1.0,
        scheduler=False,
        number_of_classes=4,
        number_of_channels=1,
        fold_test=1,
        tuning=True,
        patch_shape=(256,128), 
        n_pos=12, 
        n_neg=0, 
        pos=1, 
        neg=0,
        val_percent=0.2,
        amp=True,
        patience=10
    )
