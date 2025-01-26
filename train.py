from torch.utils.data import DataLoader
from networks.unet import UNet

def train_model (
        model,
        device,
        epochs,
        batch_size,
        learning_rate,
        optimizer,
        number_of_classes,
        number_of_channels,
        fold_test,
        tuning
):
    """
    Function that trains the deep learning models.

    Args:
        model (str): name of the model desired to train
        device (str): indicates whether the network will 
        be trained using the CPU or the GPU
        epochs (int): maximum number of epochs the model 
        will train for
        batch_size (int): size of the batch used in 
        training
        learning_rate (int): learning rate of the 
        optimization function
        optimizer (string): optimization function used
        number_of_classes (int): number of classes the 
        model is supposed to output
        number_of_channels (int): number of channels the 
        input will present
        fold_test (int): number of the fold that will be used 
        in the network testing 
        tuning (bool): indicates whether this train will
        be performed to tune the hyperparameters or not
    
    Return:
        None
    """

    # Dictionary of models, associates a string to a 
    # PyTorch module
    models = {
        "UNet": "UNet()"
        # "UNet3":
        # "2.5D":
    }

    # Checks whether the selected model exists or not
    if model not in models.keys():
        print("Model not recognized. Possible models:")
        for key in models.keys():
            print(key)
        return 0
    
if __name__ == "__main__":
    train_model(
        model="UNet",
        device="GPU",
        epochs=100,
        batch_size=32,
        learning_rate=2e-5,
        optimizer="Adam",
        number_of_classes=4,
        number_of_channels=1,
        fold_test=1,
        tuning=False
    )
