import csv
import logging
import torch
from IPython import get_ipython
from os import makedirs, remove
from os.path import exists
from pandas import read_csv
from network_functions.dataset import TrainDatasetGAN, ValidationDatasetGAN
from network_functions.evaluate import evaluate_gan
from networks.gan import Discriminator, Generator
from networks.loss import discriminator_loss, generator_loss
from networks.unet import UNet

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def weights_normal_initialization(m):
    """
    Initiates the weights of the model 
    following a Gaussian distribution
    
    Args:
        m (PyTorch Module): PyTorch 
            module that will have its
            weights initialized

    Returns:
        None  
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_gan(
        run_name: str,
        fold_val: int,
        amp: bool=True,
        batch_size: int=8,
        beta_1: float=0.5,
        beta_2: float=0.999,
        device="GPU",
        epochs=400,
        fold_test: int=1,
        gradient_clipping: float=1.0,
        learning_rate: float=2e-5,
        model_name: str="GAN",
        number_of_channels: int=2,
        number_of_classes: int=1,
        patience: int=400,
        patience_after_n: int=0,
        split: str="generation_5_fold_split.csv",
        weight_decay: float=None
):
    """
    Function that trains the GAN models.

    Args:
        run_name (str): name of the run under which the best model
            will be saved
        fold_val (int): number of the fold that will be used 
            in the network validation 
        amp (bool): bool that indicates whether automatic mixed
            precision is going to be used or not
        batch_size (int): size of the batch used in 
            training
        beta_1 (float): value of beta_1 used in both optimizers
        beta_2 (float): value of beta_2 used in both optimizers
        device (str): indicates whether the network will 
            be trained using the CPU or the GPU
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
        model_name (str): name of the model that will be trained 
            to generate images. Can only be "GAN" or "UNet". The 
            default is "GAN"
        number_of_channels (int): number of channels the 
            input will present
        number_of_classes (int): number of classes the 
            model is supposed to output
        patience (int): number of epochs where the validation 
            errors calculated are worse than the best validation 
            error before terminating training
        patience_after_n (int): number of epochs needed to wait 
            before starting to count the patience. The default 
            value is 0
        split (str): name of the k-fold split file that will be 
            used in this run
        weight_decay (float): optimizer's weight decay value

    Returns:
        None
    """
    # Declares what the logging style will be
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Checks if the given model name is available
    assert model_name in ["GAN", "UNet"], "Possible model names: 'GAN' or 'UNet'"

    # Declares the path to the CSV file on which the logging of 
    # the training and epochs will be made
    csv_epoch_filename = f"logs\{run_name}_training_log_epoch_{model_name.lower()}.csv"
    csv_batch_filename = f"logs\{run_name}_training_log_batch_{model_name.lower()}.csv"
    # Creates the folder in case it does not exist
    makedirs("logs", exist_ok=True)

    # Creates the logging files in case they do not exist
    if (exists(csv_epoch_filename) and exists(csv_batch_filename)):
        # Deletes the files with 
        # the desired name
        remove(csv_epoch_filename)
        remove(csv_batch_filename)

    # Defines the device that will be used to train the network
    device = torch.device("cuda" if torch.cuda.is_available() and device == "GPU" else "cpu")
    # Iniates the model
    if model_name == "GAN":
        # Defines the Discriminator module
        discriminator = Discriminator(number_of_classes)
        # Allocates the Discriminator module to the GPU
        discriminator.to(device=device)
        # Defines the Generator module
        generator = Generator(number_of_classes)
        # Allocates the Generator module to the GPU
        generator.to(device=device)

        # Initiates the weights of the Pix2Pix generator 
        # and discriminator
        generator.apply(weights_normal_initialization)
        discriminator.apply(weights_normal_initialization)

        # Declares the optimizer of the Generator
        optimizer_G = torch.optim.Adam(generator.parameters(), 
                                    lr=learning_rate, 
                                    betas=(beta_1, beta_2), 
                                    foreach=True)
        
        # Declares the optimizer of the Discriminator
        optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                                    lr=learning_rate, 
                                    betas=(beta_1, beta_2), 
                                    foreach=True)

        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Declares which information will be saved in the logging file
            # associated with each epoch
            writer.writerow(["Epoch", "Adversarial Loss", 
                                "Generator Loss", "Real Loss", 
                                "Fake Loss", "Discriminator Loss", 
                                "Epoch Validation MS-SSIM"])
            
        with open(csv_batch_filename, mode="w", newline="") as file:
            # Declares which information will be saved in the logging file
            # associated with each batch
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Adversarial Loss", 
                                "Batch Generator Loss", "Batch Real Loss", 
                                "Batch Fake Loss", "Batch Discriminator Loss"])
    elif model_name == "UNet":
        # Initiates the U-Net model in case that is the model selected
        unet = UNet(in_channels=number_of_channels, num_classes=number_of_classes)
        unet = unet.to(device=device)
        # Initiates the U-Net optimizer
        optimizer_unet = torch.optim.Adam(params=unet.parameters(), 
                                           lr=learning_rate, 
                                           foreach=True, 
                                           maximize=False, 
                                           weight_decay=weight_decay)
   
        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Declares which information will be saved in the logging file
            # associated with each epoch
            writer.writerow(["Epoch", "Validation Loss"])
            
        with open(csv_batch_filename, mode="w", newline="") as file:
            # Declares which information will be saved in the logging file
            # associated with each batch
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Loss"])

    # Initiates the grad scaler in case mixed 
    # precision is used
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # Logs the information of the input and output 
    # channels of the network
    logging.info(
        f"Network\n"
        f"\t{number_of_channels} input channels\n"
        f"\t{number_of_classes} output channels\n"
    )

    # Reads the CSV file that indicates which images will be used to 
    # train the network and which images will be used to validate it
    df = read_csv(f"splits/{split}")
    train_volumes = []
    val_volumes = []
    for col in df.columns:
        if (col != str(fold_test)) and (col != str(fold_val)):
            train_volumes = train_volumes + df[col].dropna().to_list()
        if (col == str(fold_val)):
            val_volumes = val_volumes + df[col].dropna().to_list()

    # Informs the beginning of the training and registers the number 
    # of epochs, the batch size, the learning rate that will be used
    # and whether it is being trained on 'cuda' or 'cpu'
    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
        Mixed Precision: {amp}
    """
    )

    # Creates the training and validation set as a PyTorch 
    # Dataset, using the list of OCT volumes that will be 
    # used to train and validate the network
    train_set = TrainDatasetGAN(train_volumes=train_volumes, model_name=model_name)
    val_set = ValidationDatasetGAN(val_volumes=val_volumes, model_name=model_name)
    # With the previously created datasets, creates a DataLoader that splits this information in multiple batches
    # The number of workers selected corresponds to the number of cores available in the CPU and the persistent workers means they are not closed/killed 
    # every time training or validation ceases
    # pin_memory defines that an allocated memory is specified for the datasets, reducing the memory consumption by them 
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=12, persistent_workers=True, batch_size=batch_size, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, num_workers=12, persistent_workers=True, batch_size=batch_size, pin_memory=True)

    # Initiates the validation 
    # PSNR as infinite 
    best_val_psnr = 0

    # Iterates through the total 
    # number of possible epochs
    for epoch in range(1, epochs + 1):
        # Sets the generator and 
        # discriminator to training mode
        if model_name == "GAN":
            generator.train()
            discriminator.train()
            # Initializes each loss 
            # component of this 
            # epoch as zero
            epoch_adv_loss = 0
            epoch_g_loss = 0
            epoch_real_loss = 0
            epoch_fake_loss = 0
            epoch_d_loss = 0
        elif model_name == "UNet":
            # Sets the UNet to training mode
            unet.train()
            epoch_loss = 0

        print(f"Training Epoch {epoch}")
        # Starts a progress bar that indicates which epoch is trained and updates as the number of images used in training changes
        with tqdm(total=len(train_set), desc=f"Epoch {epoch}/{epochs}", unit="img", leave=True, position=0) as progress_bar:
            # Iterates through the number of batches that compose the dataloader
            for batch_num, batch in enumerate(train_loader):
                # Gets the stack of images, that have as shape: B x C x H x W
                # In this case, C=3 because we are loading three images at a time
                # H x W = 496 x 512 because those were the selected dimensions
                # B corresponds to the batch size whose default value is 32
                stack = batch["stack"]

                # Separates the stack in the previous image, 
                # the middle image (the one we aim to predict), 
                # and following image, allocating them to the GPU 
                prev_imgs = stack[:,0,:,:].to(device=device)
                mid_imgs = stack[:,1,:,:].to(device=device)
                next_imgs = stack[:,2,:,:].to(device=device)

                # Allows for mixed precision calculations, attributes a device to be used in 
                # these calculations
                # Calculates loss
                with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=amp):
                    if model_name == "GAN":
                        # Sets the label associated with true images to 0.95. The reason 
                        # this value is set to 0.95 and not 1 is called label smoothing and 
                        # prevents the model of becoming too overconfident, improving training 
                        # stability. The same is done for the label associated with fake images, 
                        # in which the label is set to 0.1 instead of the expected value 0
                        valid = torch.Tensor(stack.shape[0], 1, 1, 1).fill_(0.95).to(device)
                        fake = torch.Tensor(stack.shape[0], 1, 1, 1).fill_(0.1).to(device)
                        # Sets the gradient of the 
                        # generator optimizer for 
                        # this epoch to zero
                        optimizer_G.zero_grad()
                        # Calls the generator to generate the expected middle 
                        # image by receiving the previous and following images
                        gen_imgs = generator(prev_imgs, next_imgs)
                        # Calculates the loss of the generator, which compares the generated images 
                        # with the real images
                        adv_loss, g_loss = generator_loss(device, discriminator, gen_imgs, mid_imgs, valid)
                        # Acumulates scaled gradients
                        grad_scaler.scale(g_loss).backward()
                        # Unscales the gradients so that 
                        # they can be clipped
                        grad_scaler.unscale_(optimizer_G)
                        # Clips the gradients above the threshold
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                        # Updates the parameters 
                        # based on the current gradient
                        grad_scaler.step(optimizer_G)

                        # Sets the gradient of the 
                        # generator optimizer for 
                        # this epoch to zero
                        optimizer_D.zero_grad()
                        # Gets the prediction of the discriminator 
                        # on whether the true image is true or fake                
                        gt_distingue = discriminator(mid_imgs)
                        # Gets the prediction of the discriminator 
                        # on whether the fake image is true or fake
                        # The generated image is detached so that 
                        # the gradient of the discriminator does not 
                        # affect the generator function
                        fake_distingue = discriminator(gen_imgs.detach())
                        # The discriminator loss is calculated for the true image and fake image predicted labels 
                        # and their respective true labels
                        real_loss, fake_loss, d_loss = discriminator_loss(device, gt_distingue, fake_distingue, valid, fake)
                        # Acumulates scaled gradients
                        grad_scaler.scale(d_loss).backward()
                        # Unscales the gradients so that 
                        # they can be clipped
                        grad_scaler.unscale_(optimizer_D)
                        # Clips the gradients above the threshold
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                        # Updates the parameters 
                        # based on the current gradient
                        grad_scaler.step(optimizer_D)

                        # Updates the scale 
                        # for the next iteration
                        grad_scaler.update()

                        # Updates the loss of the current epoch
                        epoch_adv_loss += adv_loss.item()
                        epoch_g_loss += g_loss.item()
                        epoch_real_loss += real_loss.item()
                        epoch_fake_loss += fake_loss.item()
                        epoch_d_loss += d_loss.item()

                        # Writes the loss values of the batch in the CSV file
                        with open(csv_batch_filename, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, batch_num, adv_loss.item(), 
                                                g_loss.item(), real_loss.item(), 
                                                fake_loss.item(), d_loss.item()])

                    elif model_name == "UNet":
                        # Checks if the number of channels in an image matches the value indicated 
                        # in the training arguments
                        assert unet_input.shape[1] == number_of_channels, \
                        f'Network has been defined with {number_of_channels} input channels, ' \
                        f'but loaded images have {unet_input.shape[1]} channels. Please check if ' \
                        'the images are loaded correctly.' 
 
                        # Sets the gradient of the 
                        # generator optimizer for 
                        # this epoch to zero
                        unet.zero_grad()
                        
                        # Calls the UNet to generate the expected middle 
                        # image by receiving the previous and following images
                        unet_input = torch.stack([prev_imgs, next_imgs], dim=1)
                        gen_imgs = unet(unet_input.to(device=device))
                        # Calculates the loss of the generator, which compares the generated images 
                        # with the real images
                        criterion = torch.nn.L1Loss()
                        mae_loss = criterion(gen_imgs, mid_imgs.unsqueeze(1))
                        
                        # Saves the value that are zero as 
                        # None so that it saves memory
                        optimizer_unet.zero_grad(set_to_none=True)
                        # Acumulates scaled gradients
                        grad_scaler.scale(mae_loss).backward()
                        # Unscales the gradients so that 
                        # they can be clipped
                        grad_scaler.unscale_(optimizer_unet)
                        # Clips the gradients above the threshold
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        # Updates the parameters 
                        # based on the current gradient
                        grad_scaler.step(optimizer_unet)
                        # Updates the scale 
                        # for the next iteration
                        grad_scaler.update()

                        # Updates the loss of the current epoch
                        epoch_loss += mae_loss.item()
                        # Writes the loss values of the batch in the CSV file
                        with open(csv_batch_filename, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch, batch_num, mae_loss.item()])
                    # Updates the progress bar according 
                    # to the number of images present in 
                    # the batch  
                    progress_bar.update(stack.shape[0])

        print(f"Validating Epoch {epoch}")
        # Calls the function that evaluates the output of the generator and returns the 
        # respective result of the peak signal-to-noise ratio (PSNR)
        if model_name == "GAN":
            val_psnr = evaluate_gan(model_name=model_name, generator=generator, 
                                    dataloader=val_loader, device=device, amp=amp, 
                                    n_val=len(val_set))
        elif model_name == "UNet":
            val_psnr = evaluate_gan(model_name=model_name, generator=unet, 
                                    dataloader=val_loader, device=device, amp=amp, 
                                    n_val=len(val_set))

        # Logs the results of the validation
        logging.info(f"Validation Mean PSNR: {val_psnr}")

        # Writes the mean of the results of the different batches in 
        # the epoch to the CSV file 
        if model_name == "GAN":
            with open(csv_epoch_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, epoch_adv_loss.item() / len(val_loader), 
                                epoch_g_loss.item() / len(val_loader), 
                                epoch_real_loss.item() / len(val_loader), 
                                epoch_fake_loss.item() / len(val_loader), 
                                epoch_d_loss.item() / len(val_loader),
                                val_psnr])
        elif model_name == "UNet":
            with open(csv_epoch_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, val_psnr])

        # In case the validation SSIM is better 
        # than the previous, the model is saved 
        # as the best model
        if val_psnr > best_val_psnr:
            # Creates the folder models in case 
            # it does not exist yet
            makedirs("models", exist_ok=True)
            # Updates the best value of 1 - MS-SSIM
            best_val_psnr = val_psnr
            patience_counter = 0
            # Both the generator and the discriminator are saved with a name 
            # that depends on the argument input, the name of the model, and 
            # fluid desired to segment in case it exists
            if model_name == "GAN":
                torch.save(generator.state_dict(), 
                            f"models/{run_name}_generator_best_model.pth")
                torch.save(discriminator.state_dict(),
                            f"models/{run_name}_discriminator_best_model.pth")
            elif model_name == "UNet":
                torch.save(unet.state_dict(), 
                            f"models/{run_name}_unet_best_model.pth")
            print("Models saved.")
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
        if patience_counter >= patience and epoch > patience_after_n:
            logging.info("Early stopping triggered.")
            break
        
        # Resets the patience counter every epoch 
        # below the number after which it starts counting
        if epoch <= patience_after_n:
            patience_counter = 0
