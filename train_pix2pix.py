import csv
import logging
import torch
from IPython import get_ipython
from os import makedirs, remove
from os.path import exists
from pandas import read_csv
from network_functions.dataset import TrainDatasetGAN, ValidationDatasetGAN
from network_functions.evaluate import evaluate_gan
from networks.gan import Generator
from networks.loss import discriminator_loss, generator_loss
from networks.pix2pix import Pix2PixDiscriminator, Pix2PixGenerator
from networks.unet import UNet

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def train_pix2pix(
        run_name: str,
        fold_val: int,
        generator_file_name: str,
        model_name: str="GAN",
        batch_size: int=8,
        beta_1: float=0.5,
        beta_2: float=0.999,
        device="GPU",
        epochs=400,
        fold_test: int=1,
        learning_rate: float=2e-5,
        number_of_channels: int=2,
        number_of_classes: int=1,
        patience: int=400,
        patience_after_n: int=0,
        split: str="generation_5_fold_split.csv",
):
    """
    Function that trains the GAN models.

    Args:
        run_name (str): name of the run under which the best model
            will be saved
        fold_val (int): number of the fold that will be used 
            in the network validation 
        generator_file_name (str): name of the generator model 
            that will be used to generate the image for each 
            pair
        model_name (str): name of the model that will be trained 
            to generate images. Can only be "GAN" or "UNet". The 
            default is "GAN"
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
        learning_rate (float): learning rate of the 
            optimization function
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

    Returns:
        None
    """
    # Declares what the logging style will be
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Checks if the given model name is available
    assert model_name in ["GAN", "UNet"], "Possible model names: 'GAN' and 'UNet'"

    # Declares the path to the CSV file on which the logging of 
    # the training and epochs will be made
    csv_epoch_filename = f"logs\{run_name}_training_log_epoch_pix2pix.csv"
    csv_batch_filename = f"logs\{run_name}_training_log_batch_pix2pix.csv"
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
        # Defines the Generator module
        generator = Generator(number_of_classes)
        # Allocates the Generator module to the GPU
        generator.to(device=device)
        generator.load_state_dict(torch.load(generator_file_name, weights_only=True, map_location=device))
        generator.eval

    elif model_name == "UNet":
        # Initiates the U-Net model in case that is the model selected
        generator = UNet(in_channels=number_of_channels, num_classes=number_of_classes)
        # Allocates the U-Net module to the GPU
        generator = generator.to(device=device)
        generator.load_state_dict(torch.load(generator_file_name, weights_only=True, map_location=device))
        generator.eval()

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

    # Creates both the pix2pix generator and the pix2pix discriminator
    pix2pix_generator = Pix2PixGenerator(number_of_classes, number_of_classes)
    pix2pix_discriminator = Pix2PixDiscriminator(number_of_classes)

    # Declares the optimizer of the Generator
    optimizer_G = torch.optim.Adam(pix2pix_generator.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2), 
                                foreach=True)
    
    # Declares the optimizer of the Discriminator
    optimizer_D = torch.optim.Adam(pix2pix_discriminator.parameters(), 
                                lr=learning_rate, 
                                betas=(beta_1, beta_2), 
                                foreach=True)

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

                # Using the trained generator, the previous and 
                # following images are used to generate the middle image
                if model_name == "GAN":
                    gen_imgs_wgenerator = generator(prev_imgs.detach(), next_imgs.detach())
                elif model_name == "UNet":
                    gen_imgs_wgenerator = generator(torch.stack([prev_imgs, next_imgs], dim=1).detach().float() / 255.0)
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
                # Calls the generator to transform the middle image
                # when receiving the image generated by the trained
                # generator
                gen_imgs = pix2pix_generator(gen_imgs_wgenerator)
                # Calculates the loss of the generator, which compares the generated images 
                # with the real images
                adv_loss, g_loss = generator_loss(device, pix2pix_discriminator, gen_imgs, mid_imgs, valid)
                # Calculates the gradient of the 
                # generator using the generator loss 
                g_loss.backward()
                # The optimizer performs the 
                # backwards step on the generator
                optimizer_G.step()

                # Sets the gradient of the 
                # generator optimizer for 
                # this epoch to zero
                optimizer_D.zero_grad()

                # Gets the prediction of the discriminator 
                # on whether the true image is true or fake                
                gt_distingue = pix2pix_discriminator(torch.cat((gen_imgs_wgenerator.detach(), gen_imgs), dim=1))
                # Gets the prediction of the discriminator 
                # on whether the fake image is true or fake
                # The generated image is detached so that 
                # the gradient of the discriminator does not 
                # affect the generator function
                fake_distingue = pix2pix_discriminator(torch.cat((gen_imgs_wgenerator.detach(), mid_imgs.detach())))
                # The discriminator loss is calculated for the true image and fake image predicted labels 
                # and their respective true labels
                real_loss, fake_loss, d_loss = discriminator_loss(device, gt_distingue, fake_distingue, valid, fake)
                # The backward step is calculated 
                # for the model according to the 
                # discriminator loss
                d_loss.backward()

                # The optimizer performs the backwards step 
                # on the discriminator
                optimizer_D.step()

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

                # Updates the progress bar according 
                # to the number of images present in 
                # the batch  
                progress_bar.update(stack.shape[0])

                
        print(f"Validating Epoch {epoch}")
        # Calls the function that evaluates the output of the generator and returns the 
        # respective result of the peak signal-to-noise ratio (PSNR)
        val_psnr = evaluate_gan(model_name=model_name, generator=pix2pix_generator, dataloader=val_loader, device=device)

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
            torch.save(pix2pix_generator.state_dict(), 
                        f"models/{run_name}_p2p_generator_best_model.pth")
            torch.save(pix2pix_discriminator.state_dict(),
                        f"models/{run_name}_p2p_discriminator_best_model.pth")
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
