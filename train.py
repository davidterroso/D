import logging
import torch
import wandb
from numpy import any, expand_dims, stack
from numpy.random import random_sample
from os import cpu_count, listdir, remove
from pandas import read_csv
from paths import IMAGES_PATH
from skimage.io import imread
from torch import optim
from torchvision.transforms.v2 import Compose, RandomApply, RandomHorizontalFlip, RandomRotation
from torch.nn.functional import one_hot, softmax
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from init.patchExtraction import extractPatches, extractPatches25D
from networks.loss import multiclass_balanced_cross_entropy_loss
from networks.unet import UNet

def dropPatches(prob, volumes_list, model):
    """
    Randomly drops a percentage of extracted patches whose slice does
    not present fluid

    Args:
        prob (float): fraction of patches from each slice that will be 
            dropped in case there is no fluid in the slice
        volumes_list (List[float]): list of the OCT volume's identifier 
            that will be used in training
        model (str): name of the model used

    Return: None
    """
    # The path to the patches is dependent on the model selected
    if model == "2.5D":
        images_path = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2.5D\\"
    else:
        images_path = IMAGES_PATH + "\\OCT_images\\segmentation\\patches\\2D\\"

    # Declares the path where the true masks are located
    masks_path = IMAGES_PATH + "\\OCT_images\\segmentation\\masks\\int8\\"

    # Declares the path of the patches
    patches_slice_path = images_path + "slices\\"
    patches_mask_path = images_path + "masks\\"
    patches_roi_path = images_path + "roi\\"
    
    # Iterates through the ground truth masks
    for mask in listdir(masks_path):
        # Separates the name of the mask 
        # to obtain its information 
        mask_name_parts = mask.split("_")

        # Only considers the slices that are in the list 
        # of volumes that will be used in training
        if int(mask_name_parts[1][-3:]) in volumes_list:
            # Reads the mask
            fluid_mask = imread(str(masks_path + mask))

            # Checks if the slice has any of 
            # the three fluids
            irf_exists = any(fluid_mask==1)
            srf_exists = any(fluid_mask==2)
            ped_exists = any(fluid_mask==3)

            # In case there is no fluid, randomly eliminates patches
            if not (irf_exists or srf_exists or ped_exists):
                # Iterates through the patches extracted
                for patch in listdir(patches_slice_path):
                    # Extracts the information from the patches
                    patch_name_parts = patch.split("_")
                    # Checks if the name of the vendor, the volume, and the
                    # slice are the same
                    # In case the model is the 2.5D only considers the center 
                    # slice and eliminates the slices to it associated afterwards
                    if (((patch_name_parts[0] == mask_name_parts[0]) and \
                        (patch_name_parts[1] == mask_name_parts[1]) and \
                        (patch_name_parts[2] == mask_name_parts[2][:3]))
                        and (model != "2.5D" or len(patch_name_parts) == 6)):

                        # Randomly eliminates the indicated percentage 
                        # of patches
                        if (float(random_sample()) < prob):
                            remove(str(patches_slice_path + patch))
                            remove(str(patches_mask_path + patch))
                            remove(str(patches_roi_path + patch))

                            # In case it is the 2.5D segmentation model
                            # also deletes the associated before and 
                            # following patch  
                            if model == "2.5D":
                                # Deletes the previous slices
                                remove(str(patches_slice_path + patch[:-5] + "_before.tiff"))
                                remove(str(patches_mask_path + patch[:-5] + "_before.tiff"))
                                remove(str(patches_roi_path + patch[:-5] + "_before.tiff"))

                                # Deletes the following slices
                                remove(str(patches_slice_path + patch[:-5] + "_after.tiff"))
                                remove(str(patches_mask_path + patch[:-5] + "_after.tiff"))
                                remove(str(patches_roi_path + patch[:-5] + "_after.tiff"))

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
    def __init__(self, train_volumes, model, fluid):
        """
        Initiates the Dataset object and gets the possible 
        names of the patches that will be used in training

        Args: 
            train_volumes(List[float]): list of the training 
                volumes that will be used to train the model
            model (str): name of the model that will be trained
            fluid (int): label of fluid that is expected to 
                segment
        Return:
            None
        """
        # Initiates the model, gets the name of the slices that
        # compose the dataset, the transformations that will be 
        # applied to the images, and the 
        super().__init__()
        self.model = model
        self.patches_names = getPatchesFromVolumes(train_volumes, model)
        # Random Rotation has a probability of 0.5 of rotating 
        # the image between 0 and 10 degrees
        # Random Horizontal Flip has a probability of 0.5 
        # flipping the image horizontally
        self.transforms = Compose([
            RandomApply([RandomRotation(degrees=[0,10])], p=0.5),
            RandomHorizontalFlip(p=0.5)])
        self.fluid = fluid

    def __len__(self):
        """
        Function required in the Dataset object that returns the length 
        of the images used in training

        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself

        Return:
            None
        """
        return len(self.patches_names)

    def __getitem__(self, index):
        """
        Gets an image from the list of images that can be used in training
        when an index is given, utilized to access the list of images
        
        Args:
            self (PyTorch Dataset): the PyTorch Dataset object itself
            index (int): index of the dataset to get the image from

        Return:
            None
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

        # In case the selected model is the 2.5D, also loads the previous
        # and following slice
        if self.model == "2.5D":
            scan_before = imread(str(slice_name[:-5] + "_before.tiff"))
            scan_after = imread(str(slice_name[:-5] + "_after.tiff"))
            # Stacks them to apply the transformations
            scan = stack(arrays=[scan_before, scan, scan_after], axis=0)

        # In case the model selected is the UNet3, all the labels 
        # that are not the one desired to segment are set to 0
        if self.model == "UNet3":
            mask = ((mask == self.fluid).astype(int) * self.fluid)

        # Z-Score Normalization / Standardization
        # Mean of 0 and SD of 1
        scan = (scan - 128.) / 128.

        # Expands the scan dimentions to 
        # include an extra channel of value 1
        # as the first channel
        # The mask dimensions are also expanded 
        # to match
        if self.model != "2.5D":
            scan = expand_dims(scan, axis=0)
        mask = expand_dims(mask, axis=0)

        # Converts the scan and mask 
        # to a PyTorch Tensor
        scan = torch.from_numpy(scan)
        mask = torch.from_numpy(mask)

        # Forms a stack with the scan and the mask
        # Initial Scan Shape: 1 x H x W / 3 x H x W
        # Initial Mask Shape: 1 x H x W
        # Resulting Shape: 2 x H x W / 4 x H x W
        resulting_stack = torch.cat([scan, mask], dim=0)

        # Applies the transfomration to the stack
        transformed = self.transforms(resulting_stack)

        # Separate the scan and the mask from the stack
        # Keeps the extra dimension on the slice but not on the mask
        if self.model != "2.5D":
            scan, mask = transformed[0].unsqueeze(0), transformed[1]
        # Handles it differently for the 2.5D model, ensuring the correct order of slices 
        else:
            scan = torch.cat([transformed[0], transformed[1], transformed[2]], dim=0)
            mask = transformed[3]

        # Converts the scans back to NumPy
        scan = scan.cpu().numpy()
        mask = mask.cpu().numpy()

        # Declares a sample as a dictionary that 
        # to the keyword "scan" associates the 
        # original B-scan and to the keyword "mask" 
        # associates the fluid mask
        sample = {"scan": scan, "mask": mask}
        return sample
    
@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    """
    Function used to evaluate the model

    Args:
        model (PyTorch Module object): model that is being 
            trained
        dataloader (PyTorch DataLoader object): DataLoader 
            that contains the training and evaluation data
        device (str): indicates which PyTorch device is
            going to be used
        amp (bool): flag that indicates if automatic 
            mixed precision is being used

    Return:
        Weighted mean of the loss across the considered 
        batches
    """
    # Sets the network to evaluation mode
    model.eval()
    # Calculates the number of batches 
    # used to validate the network
    num_val_batches = len(dataloader)
    # Initiates the loss as zero
    total_loss = 0

    # Allows for mixed precision calculations, attributes a device to be used in 
    # these calculations
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # Gets the images and the masks from the dataloader
            image, mask_true = batch['scan'], batch['mask']

            # Handles the images and masks according to the device, specified data type and memory format
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predicts the masks of the received images
            masks_pred = model(image)
            # Performs softmax on the predicted masks
            # dim=1 indicates that the softmax is calculated 
            # across the masks, since the channels is the first 
            # dimension
            masks_pred_prob = softmax(masks_pred, dim=1).float()
            # Permute changes the images from channels first to channels last
            masks_pred_prob = masks_pred_prob.permute(0, 2, 3, 1)
            # Performs one hot encoding on the true masks, in channels last format
            masks_true_one_hot = one_hot(mask_true.long(), model.n_classes).float()

            # Calculates the balanced loss for the background mask
            loss = multiclass_balanced_cross_entropy_loss(
                                y_true=masks_true_one_hot,
                                y_pred=masks_pred_prob, 
                                batch_size=image.shape[0], 
                                n_classes=model.n_classes, 
                                eps=1e-7)

            # Accumulate loss
            total_loss += loss.item()

    # Sets the model to train mode again
    model.train()
    # Returns the weighted mean of the total 
    # loss according to the fluid voxels
    # Also avoids division by zero
    return total_loss / max(num_val_batches, 1)

def train_model (
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
        fluid=None
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
        run_name (str): name of the run under which the best model
            will be saved
        
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
        "2.5D": UNet(in_channels=number_of_channels, num_classes=number_of_classes) # Change later
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
        print("...")

        # Eliminates the previous patches and saves 
        # new patches to train the model, but only 
        # for the volumes that will be used in training
        if model_name != "2.5D":
            extractPatches(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=train_volumes)
        else:
            extractPatches25D(IMAGES_PATH, 
                        patch_shape=patch_shape, 
                        n_pos=n_pos, n_neg=n_neg, 
                        pos=pos, neg=neg, 
                        volumes=train_volumes)
        
        # Randomly drops patches of slices that do not have retinal fluid
        dropPatches(prob=0.75, volumes_list=train_volumes, model=model_name)
        
        # Creates the Dataset object
        dataset = TrainDataset(train_volumes, model_name)

        # Splits the dataset in training and 
        # validation to train the model, with a 
        # fixed seed to allow reproducibility
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        # Using the Dataset object, creates a DataLoader object 
        # which will be used to train the model in batches
        loader_args = dict(batch_size=batch_size, num_workers=cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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
                            f"models/{model_name}_best_model.pth")
            else:
                torch.save(model.state_dict(), 
                           f"models/{label_to_fluids.get(fluid)}_{model_name}_best_model.pth")
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
