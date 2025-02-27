import csv
import matplotlib.pyplot as plt
from numpy import linspace
from os import makedirs

def plot_logs(run_name: str):
    """
    Plots the logged runs of the training files, as loss per 
    batch and per epoch, in training and validation

    Args:
        run_name (str): name of the run desired to plot

    Return:
        None
    """
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
    
    # Creates the imgs folder in case 
    # it does not exist
    makedirs("imgs", exist_ok=True)

    # Declares the path to save
    img_path = f"imgs\{run_name}_training_error.png"
    # Saves the image to the designated path
    plt.savefig(img_path)

    # Shows the image
    plt.show()

# In case it is needed to run manually, 
# here is the code
if __name__ == "__main__":
    plot_logs(run_name="Run6")
