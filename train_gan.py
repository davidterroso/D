import csv
import logging
import torch
from os import makedirs, remove
from os.path import exists
from pandas import read_csv
from networks.gan import Discriminator, Generator

def train_gan(
        run_name: str,
        fold_val: int,
        amp: bool=True,
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
        Mixed precision: {amp}
    """
    )
    optimizer = torch.optim.Adam(lr=learning_rate, 
                                 betas=(beta_1, beta_2), 
                                 foreach=True)
    
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    csv_epoch_filename = f"logs\{run_name}_training_log_epoch.csv"
    csv_batch_filename = f"logs\{run_name}_training_log_batch.csv"

    makedirs("logs", exist_ok=True)

    if not (exists(csv_epoch_filename) and exists(csv_batch_filename)):
        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Epoch Training Loss", "Epoch Validation Loss"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Training Loss"])
    else:
        remove(csv_epoch_filename)
        remove(csv_batch_filename)
        with open(csv_epoch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Epoch Training Loss", "Epoch Validation Loss"])
        
        with open(csv_batch_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Batch", "Batch Training Loss"])

    