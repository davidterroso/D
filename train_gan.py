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

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

def train_gan(
        run_name: str,
        fold_val: int,
        batch_size: int=8,
        beta_1: float=0.5,
        beta_2: float=0.999,
        device="GPU",
        epochs=400,
        fluid: str=None,
        fold_test: int=1,
        learning_rate: float=2e-5,
        number_of_channels: int=3,
        number_of_classes: int=1,
        patience: int=400,
        patience_after_n: int=0,
        split: str="generation_fold_selection",
):
    """
    Function that trains the deep learning models.

    Args:
        run_name (str): name of the run under which the best model
            will be saved
        fold_val (int): number of the fold that will be used 
            in the network validation 
        batch_size (int): size of the batch used in 
            training
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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() and device == "GPU" else "cpu")
    discriminator = Discriminator()
    discriminator.to(device=device, memory_format=torch.channels_last)
    generator = Generator() 
    generator.to(device=device, memory_format=torch.channels_last)

    logging.info(
        f"Network\n"
        f"\t{number_of_channels} input channels\n"
        f"\t{number_of_classes} output channels\n"
    )

    df = read_csv(f"splits/{split}")
    val_volumes = []
    for col in df.columns:
        if (col != str(fold_test)) and (col != str(fold_val)):
            train_volumes = train_volumes + df[col].dropna().to_list()
        if (col == str(fold_val)):
            val_volumes = val_volumes + df[col].dropna().to_list()

    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
    """
    )
    optimizer_G = torch.optim.Adam(lr=learning_rate, 
                                 betas=(beta_1, beta_2), 
                                 foreach=True)
    
    optimizer_D = torch.optim.Adam(lr=learning_rate, 
                                betas=(beta_1, beta_2), 
                                foreach=True)

    csv_epoch_filename = f"logs\{run_name}_training_log_epoch.csv"
    csv_batch_filename = f"logs\{run_name}_training_log_batch.csv"

    makedirs("logs", exist_ok=True)

    if not (exists(csv_epoch_filename) and exists(csv_batch_filename)):
        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Adversarial Loss", 
                             "Generator Loss", "Batch Real Loss", 
                             "Batch Fake Loss", "Batch Discriminator Loss", 
                             "Epoch Validation SSMI"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Adversarial Loss", 
                             "Generator Loss", "Batch Real Loss", 
                             "Batch Fake Loss", "Batch Discriminator Loss"])
    else:
        remove(csv_epoch_filename)
        remove(csv_batch_filename)
        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Adversarial Loss", 
                             "Generator Loss", "Batch Real Loss", 
                             "Batch Fake Loss", "Batch Discriminator Loss", 
                             "Epoch Validation SSMI"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Adversarial Loss", 
                             "Generator Loss", "Batch Real Loss", 
                             "Batch Fake Loss", "Batch Discriminator Loss"])
    train_set = TrainDatasetGAN(train_volumes=train_volumes)
    val_set = ValidationDatasetGAN(val_volumes=val_volumes)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=12, persistent_workers=True, batch_size=batch_size, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, num_workers=12, persistent_workers=True, batch_size=batch_size, pin_memory=True)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        epoch_adv_loss = 0
        epoch_g_loss = 0
        epoch_real_loss = 0
        epoch_fake_loss = 0
        epoch_d_loss = 0

        print(f"Training Epoch {epoch}")

        with tqdm(total=len(train_set), desc=f"Epoch {epoch}/{epochs}", unit="img", leave=True, position=0) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                stack = batch["stack"]

                assert stack[0].shape[0] == number_of_channels, \
                f'Network has been defined with {number_of_channels} input channels, ' \
                f'but loaded images have {stack[0].shape[0]} channels. Please check if ' \
                'the images are loaded correctly.'

                valid = torch.autograd.Variable(torch.Tensor(stack.shape[0], 1, 1, 1).fill_(0.95), requires_grad=False)
                fake = torch.autograd.Variable(torch.Tensor(stack.shape[0], 1, 1, 1).fill_(0.1), requires_grad=False)

                prev_imgs = stack[:,0,:,:].to(device=device)
                mid_imgs = stack[:,1,:,:].to(device=device)
                next_imgs = stack[:,2,:,:].to(device=device)

                optimizer_G.zero_grad()
                gen_imgs = generator(prev_imgs.data, next_imgs.data)
                adv_loss, g_loss = gen_loss(gen_imgs, mid_imgs, valid)
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()

                gt_distingue = discriminator(mid_imgs)
                fake_distingue = discriminator(gen_imgs.detach())
                real_loss, fake_loss, d_loss = discriminator_loss(gt_distingue, fake_distingue, valid, fake)
                d_loss.backward()

                optimizer_D.step()

                progress_bar.update(stack.shape[0])

                epoch_adv_loss += adv_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_real_loss += real_loss.item()
                epoch_fake_loss += fake_loss.item()
                epoch_d_loss += d_loss.item()

                with open(csv_batch_filename, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch, batch_num, adv_loss.item(), 
                                     g_loss.item(), real_loss.item(), 
                                     fake_loss.item(), d_loss.item()])
                
        print(f"Validating Epoch {epoch}")
        val_ssmi = evaluate_gan(generator=generator, dataloader=val_loader, device=device)
        logging.info(f"Validation Mean Loss: {val_ssmi}")

        with open(csv_epoch_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, adv_loss.item() / len(val_loader), 
                            g_loss.item() / len(val_loader), 
                            real_loss.item() / len(val_loader), 
                            fake_loss.item() / len(val_loader), 
                            d_loss.item() / len(val_loader),
                            val_ssmi / len(val_loader)])

        if val_ssmi > best_val_ssmi:
            # Creates the folder models in case 
            # it does not exist yet
            makedirs("models", exist_ok=True)
            best_val_ssmi = val_ssmi
            patience_counter = 0
            # File is saved with a name that depends on the argument input, the name 
            # of the model, and fluid desired to segment in case it exists
            torch.save(generator.state_dict(), 
                        f"models/{run_name}_generator_best_model.pth")
            torch.save(discriminator.state_dict(),
                        f"models/{run_name}_discriminator_best_model.pth")
            print("Models saved.")
        # In case the model has not 
        # obtained a better performance, 
        # the patience counter increases
        else:
            patience_counter += 1