import csv
import matplotlib.pyplot as plt
from numpy import linspace
from os import makedirs

def plot_logs(imgs_folder: str, run_name: str):
    """
    Plots the logged runs of the training files, as loss per 
    batch and per epoch, in training and validation

    Args:
        imgs_folder (str): folder on which the images will 
            be saved
        run_name (str): name of the run desired to plot

    Return:
        None
    """
    # Sets Matplotlib to only log messages 
    # of importance WARNING or higher
    plt.set_loglevel("WARNING")
    # Declares the path of the epochs and batch losses
    epochs_loss_csv = f"logs\{run_name}_training_log_epoch.csv"
    batch_loss_csv = f"logs\{run_name}_training_log_batch.csv"

    # Initiates the list with the number of epochs, 
    # the value obtained for the loss during training 
    # and validation in the same epoch
    x_epoch = []
    train_epoch_loss = []
    val_epoch_loss = []
    # Opens the file with the loss in epochs
    with open(epochs_loss_csv) as file:
        # Reads the lines in the CSV file
        lines = csv.reader(file, delimiter = ',')
        # Skips the header
        next(lines)
        # Iterates through the lines 
        # and appends the loss values 
        # to their respective lists
        for row in lines:
            x_epoch.append(row[0])
            train_epoch_loss.append(float(row[1]))
            val_epoch_loss.append(float(row[2]))

    # Initiates the list with the number of batches and
    # the value obtained for the loss during training 
    x_batch = []
    train_batch_loss = []
    # Opens the file with the loss in batches
    with open(batch_loss_csv) as file:
        # Reads the lines in the CSV file
        lines = csv.reader(file, delimiter = ',')
        # Skips the header
        next(lines)
        # Iterates through the lines 
        # and appends the loss values 
        # to their respective lists
        for i, row in enumerate(lines):
            x_batch.append(i)
            train_batch_loss.append(float(row[2]))

    # Divides an image in three subplots 
    # aligned horizontally
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,8))
    fig.suptitle(f"{run_name} Losses")

    # Declares the information that is going 
    # to be plotted, the labels on the axes 
    # and the title of the subplot
    # Also informs the color, the style of 
    # the line that connects the data points
    # and the marker of a data point
    # Sets the ticks to show only every 5 
    # or 10 values
    len_epoch = len(x_epoch) - 1
    ticks_epoch = linspace(0, len_epoch, 10)
    ticks_batch = linspace(0, len(x_batch), 10)

    ax1.plot(x_epoch, train_epoch_loss, 
             color='g', linestyle='dashed', 
             marker='o')
    ax1.set_title("Train Loss/Epoch")
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.set_xticks(ticks_epoch)

    ax2.plot(x_epoch, val_epoch_loss, 
             color='r', linestyle='dashed', 
             marker='o')
    ax2.set_title("Validation Loss/Epoch")
    ax2.set(xlabel="Epochs", ylabel="Loss")
    ax2.set_xticks(ticks_epoch) 

    ax3.plot(x_batch, train_batch_loss, 
             color = 'b', linestyle = 'dashed', 
             marker = 'o', alpha=0.03)
    ax3.set_title("Train Loss/Batch")
    ax3.set(xlabel="Batches", ylabel="Loss")
    ax3.set_xticks(ticks_batch)
    ax3.tick_params(axis='x', labelrotation=20)
    
    # Creates the folder in which the images 
    # will be saved in case it does not exist
    makedirs(imgs_folder, exist_ok=True)

    # Declares the path to save
    img_path = f"{imgs_folder}\{run_name}_training_error.png"
    # Saves the image to the designated path
    plt.savefig(img_path)

    # Shows the image
    plt.show()

def plot_logs_gan(imgs_folder: str, run_name: str, 
                  model_name: str):
    """
    Plots the logged runs of the training files, 
    as loss per batch and per epoch, in training 
    and validation, for the GAN

    Args:
        imgs_folder (str): folder on which the 
            images will be saved
        run_name (str): name of the run desired 
            to plot
        model_name (str): name of the model that
            was trained

    Return:
        None
    """
    # Sets Matplotlib to only log messages 
    # of importance WARNING or higher
    plt.set_loglevel("WARNING")
    # Declares the path of the epochs and batch losses
    epochs_loss_csv = f"logs\{run_name}_training_log_epoch_{model_name.lower()}.csv"
    batch_loss_csv = f"logs\{run_name}_training_log_batch_{model_name.lower()}.csv"

    # Creates the folder in which the images 
    # will be saved in case it does not exist
    makedirs(imgs_folder, exist_ok=True)

    # Initiates the list with the number of epochs, 
    # the value obtained for the loss during training 
    # and validation in the same epoch
    x_epoch = []
    adv_loss = []
    gen_loss = []
    real_loss = []
    fake_loss = []
    dis_loss = []
    real_loss = []
    val_loss = []
    val_ssim = []
    val_gd = []
    val_mae = []
    val_adv = []
    # Opens the file with the loss in epochs
    with open(epochs_loss_csv) as file:
        # Reads the lines in the CSV file
        lines = csv.reader(file, delimiter = ',')
        # Skips the header
        next(lines)
        # Iterates through the lines 
        # and appends the loss values 
        # to their respective lists
        for row in lines:
            x_epoch.append(row[0])
            if model_name == "GAN":
                adv_loss.append(float(row[1]))
                gen_loss.append(float(row[2]))
                real_loss.append(float(row[3]))
                fake_loss.append(float(row[4]))
                dis_loss.append(float(row[5]))
                val_loss.append(float(row[6]))
                val_ssim.append(float(row[7]))
                val_gd.append(float(row[8]))
                val_mae.append(float(row[9]))
                val_adv.append(float(row[10]))
            elif model_name == "UNet":
                adv_loss.append(float(row[1]))
            else:
                print("Unrecognized model name. Can only be 'UNet' or 'GAN'.")
                return

    if model_name == "GAN":
        # Divides an image in six subplots, three aligned horizontally in each row
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,8))
        fig.suptitle(f"{run_name} Epoch Losses")

        # Declares the information that is going 
        # to be plotted, the labels on the axes 
        # and the title of the subplot
        # Also informs the color, the style of 
        # the line that connects the data points
        # and the marker of a data point
        # Sets the ticks to show only every 5 
        # or 10 values
        len_epoch = len(x_epoch) - 1
        ticks_epoch = linspace(0, len_epoch, 10)

        ax1.plot(x_epoch, adv_loss, 
                color='g', linestyle='dashed', 
                marker='o')
        ax1.set_title("Adversarial Loss/Epoch")
        ax1.set(xlabel="Epochs", ylabel="Adversarial Loss")
        ax1.set_xticks(ticks_epoch)

        ax2.plot(x_epoch, gen_loss, 
                color='r', linestyle='dashed', 
                marker='o')
        ax2.set_title("Generator Loss/Epoch")
        ax2.set(xlabel="Epochs", ylabel="Generator Loss")
        ax2.set_xticks(ticks_epoch) 

        ax3.plot(x_epoch, real_loss, 
                color='b', linestyle='dashed', 
                marker='o')
        ax3.set_title("Real Loss/Epoch")
        ax3.set(xlabel="Epochs", ylabel="Real Loss")
        ax3.set_xticks(ticks_epoch)

        ax4.plot(x_epoch, fake_loss, 
                color='y', linestyle='dashed', 
                marker='o')
        ax4.set_title("Fake Loss/Epoch")
        ax4.set(xlabel="Epochs", ylabel="Fake Loss")
        ax4.set_xticks(ticks_epoch)

        ax5.plot(x_epoch, dis_loss, 
                color='k', linestyle='dashed', 
                marker='o')
        ax5.set_title("Discriminator Loss/Epoch")
        ax5.set(xlabel="Epochs", ylabel="Discriminator Loss")
        ax5.set_xticks(ticks_epoch)

        ax6.plot(x_epoch, val_loss, 
                 color='m', linestyle='dashed', 
                 marker='o')
        ax6.set_title("Validation Loss")
        ax6.set(xlabel="Epochs", ylabel="Loss")
        ax6.set_xticks(ticks_epoch)

        # Adjusts the layout 
        # to prevent overlays
        plt.tight_layout()

        # Declares the path to save
        img_path = f"{imgs_folder}\{run_name}_training_epoch_error.png"
        # Saves the image to the designated path
        plt.savefig(img_path)

        # Shows the image
        plt.show()

        # Initiates the list with the number of batches and
        # the value obtained for the loss during training 
        x_batch = []
        batch_adv_loss = []
        batch_gen_loss = []
        batch_real_loss = []
        batch_fake_loss = []
        batch_dis_loss = []
        # Opens the file with the loss in batches
        with open(batch_loss_csv) as file:
            # Reads the lines in the CSV file
            lines = csv.reader(file, delimiter = ',')
            # Skips the header
            next(lines)
            # Iterates through the lines 
            # and appends the loss values 
            # to their respective lists
            for i, row in enumerate(lines):
                x_batch.append(i)
                batch_adv_loss.append(float(row[2]))
                batch_gen_loss.append(float(row[3]))
                batch_real_loss.append(float(row[4]))
                batch_fake_loss.append(float(row[5]))
                batch_dis_loss.append(float(row[6]))

        # Divides an image in six subplots three aligned horizontally in each row
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,8))
        fig.suptitle(f"{run_name} Batch Losses")
        ticks_batch = linspace(0, len(x_batch), 10)

        ax1.plot(x_batch, batch_adv_loss, 
                color='g', linestyle='dashed', 
                marker='o', alpha=0.03)
        ax1.set_title("Adversarial Loss/Batch")
        ax1.set(xlabel="Batches", ylabel="Adversarial Loss")
        ax1.set_xticks(ticks_batch)
        ax1.tick_params(axis='x', labelrotation=20)

        ax2.plot(x_batch, batch_gen_loss, 
                color='r', linestyle='dashed', 
                marker='o', alpha=0.03)
        ax2.set_title("Generator Loss/Batch")
        ax2.set(xlabel="Batches", ylabel="Generator Loss")
        ax2.set_xticks(ticks_batch) 
        ax2.tick_params(axis='x', labelrotation=20)

        ax3.plot(x_batch, batch_real_loss, 
                color = 'b', linestyle = 'dashed', 
                marker = 'o', alpha=0.03)
        ax3.set_title("Real Loss/Batch")
        ax3.set(xlabel="Batches", ylabel="Real Loss")
        ax3.set_xticks(ticks_batch)
        ax3.tick_params(axis='x', labelrotation=20)

        ax4.plot(x_batch, batch_fake_loss, 
                color = 'y', linestyle = 'dashed', 
                marker = 'o', alpha=0.03)
        ax4.set_title("Fake Loss/Batch")
        ax4.set(xlabel="Batches", ylabel="Fake Loss")
        ax4.set_xticks(ticks_batch)
        ax4.tick_params(axis='x', labelrotation=20)

        ax5.plot(x_batch, batch_dis_loss, 
                color = 'k', linestyle = 'dashed', 
                marker = 'o', alpha=0.03)
        ax5.set_title("Discriminator Loss/Batch")
        ax5.set(xlabel="Batches", ylabel="Discriminator Loss")
        ax5.set_xticks(ticks_batch)
        ax5.tick_params(axis='x', labelrotation=20)

        # Removes the unused 
        # axis from the subplot
        fig.delaxes(ax6)

        # Adjusts the layout 
        # to prevent overlays
        plt.tight_layout()

        # Declares the path to save
        img_path = f"{imgs_folder}\{run_name}_training_batch_error.png"

        # Saves the image to the designated path
        plt.savefig(img_path)

        # Shows the image
        plt.show()

        # Divides an image in four subplots, two aligned horizontally in each row
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))
        fig.suptitle(f"{run_name} Validation Losses")

        # Declares the information that is going 
        # to be plotted, the labels on the axes 
        # and the title of the subplot
        # Also informs the color, the style of 
        # the line that connects the data points
        # and the marker of a data point
        # Sets the ticks to show only every 5 
        # or 10 values
        len_epoch = len(x_epoch) - 1
        ticks_epoch = linspace(0, len_epoch, 10)

        ax1.plot(x_epoch, val_adv, 
                color='g', linestyle='dashed', 
                marker='o')
        ax1.set_title("Adversarial Loss/Epoch")
        ax1.set(xlabel="Epochs", ylabel="Adversarial Loss")
        ax1.set_xticks(ticks_epoch)

        ax2.plot(x_epoch, val_gd, 
                color='r', linestyle='dashed', 
                marker='o')
        ax2.set_title("GD Loss/Epoch")
        ax2.set(xlabel="Epochs", ylabel="GD Loss")
        ax2.set_xticks(ticks_epoch) 

        ax3.plot(x_epoch, val_mae, 
                color='b', linestyle='dashed', 
                marker='o')
        ax3.set_title("MAE/Epoch")
        ax3.set(xlabel="Epochs", ylabel="MAE")
        ax3.set_xticks(ticks_epoch)

        ax4.plot(x_epoch, val_ssim, 
                color='y', linestyle='dashed', 
                marker='o')
        ax4.set_title("MS-SSIM/Epoch")
        ax4.set(xlabel="Epochs", ylabel="MS-SSIM")
        ax4.set_xticks(ticks_epoch)

        # Adjusts the layout 
        # to prevent overlays
        plt.tight_layout()

        # Declares the path to save
        img_path = f"{imgs_folder}\{run_name}_validation_epoch_error.png"
        # Saves the image to the designated path
        plt.savefig(img_path)

        # Shows the image
        plt.show()


    elif model_name == "UNet":
        # Initiates the list with the number of epochs, 
        # the value obtained for the loss during training 
        # and validation in the same epoch
        x_epoch = []
        train_epoch_loss = []
        val_epoch_loss = []
        # Opens the file with the loss in epochs
        with open(epochs_loss_csv) as file:
            # Reads the lines in the CSV file
            lines = csv.reader(file, delimiter = ',')
            # Skips the header
            next(lines)
            # Iterates through the lines 
            # and appends the loss values 
            # to their respective lists
            for row in lines:
                x_epoch.append(row[0])
                train_epoch_loss.append(float(row[1]))
                val_epoch_loss.append(float(row[2]))

        # Initiates the list with the number of batches and
        # the value obtained for the loss during training 
        x_batch = []
        train_batch_loss = []
        # Opens the file with the loss in batches
        with open(batch_loss_csv) as file:
            # Reads the lines in the CSV file
            lines = csv.reader(file, delimiter = ',')
            # Skips the header
            next(lines)
            # Iterates through the lines 
            # and appends the loss values 
            # to their respective lists
            for i, row in enumerate(lines):
                x_batch.append(i)
                train_batch_loss.append(float(row[2]))

        # Divides an image in three subplots 
        # aligned horizontally
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,8))
        fig.suptitle(f"{run_name} Losses")

        # Declares the information that is going 
        # to be plotted, the labels on the axes 
        # and the title of the subplot
        # Also informs the color, the style of 
        # the line that connects the data points
        # and the marker of a data point
        # Sets the ticks to show only every 5 
        # or 10 values
        len_epoch = len(x_epoch) - 1
        ticks_epoch = linspace(0, len_epoch, 10)
        ticks_batch = linspace(0, len(x_batch), 10)

        ax1.plot(x_epoch, train_epoch_loss, 
                color='g', linestyle='dashed', 
                marker='o')
        ax1.set_title("Train Loss/Epoch")
        ax1.set(xlabel="Epochs", ylabel="Loss")
        ax1.set_xticks(ticks_epoch)


        ax2.plot(x_epoch, val_epoch_loss, 
                color='r', linestyle='dashed', 
                marker='o')
        ax2.set_title("Validation Loss/Epoch")
        ax2.set(xlabel="Epochs", ylabel="Loss")
        ax2.set_xticks(ticks_epoch) 

        ax3.plot(x_batch, train_batch_loss, 
                color = 'b', linestyle = 'dashed', 
                marker = 'o', alpha=0.03)
        ax3.set_title("Train Loss/Batch")
        ax3.set(xlabel="Batches", ylabel="Loss")
        ax3.set_xticks(ticks_batch)
        ax3.tick_params(axis='x', labelrotation=20)
        
        # Creates the folder in which the images 
        # will be saved in case it does not exist
        makedirs(imgs_folder, exist_ok=True)

        # Declares the path to save
        img_path = f"{imgs_folder}\{run_name}_training_error.png"
        # Saves the image to the designated path
        plt.savefig(img_path)

        # Shows the image
        plt.show()
