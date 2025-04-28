import torch
from IPython import get_ipython
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot, softmax
from networks.loss import balanced_bce_loss, multiclass_balanced_cross_entropy_loss, psnr

# Imports tqdm depending on whether 
# it is being called from the 
# Notebook or from this file
if (get_ipython() is not None):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm.auto import tqdm

@torch.inference_mode()
def evaluate(model_name: str, model: Module, dataloader: DataLoader, 
             device: str, amp: bool, n_val: int):
    """
    Function used to evaluate the model

    Args:
        model_name (str): name of the model used in segmentation
        model (PyTorch Module object): model that is being 
            trained
        dataloader (PyTorch DataLoader object): DataLoader 
            that contains the evaluation data
        device (str): indicates which PyTorch device is
            going to be used
        amp (bool): flag that indicates if automatic 
            mixed precision is being used
        n_val (int): number of validation images

    Return:
        (float) mean of the loss across the considered 
        batches
    """
    # Sets the network to evaluation mode
    model.eval()
    # Calculates the number of batches 
    # used to validate the network
    num_val_batches = len(dataloader)
    # Initiates the loss as zero
    total_loss = 0

    # Allows for mixed precision calculations, attributes a device to be used in these calculations
    with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=amp):       
        with tqdm(dataloader, total=n_val, desc='Validating Epoch', unit='img', leave=True, position=0) as progress_bar:
            for batch in dataloader:
                    # Gets the images and the masks from the dataloader
                    images, true_masks = batch['scan'], batch['mask']

                    # Handles the images and masks according to the device, specified data type and memory format
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

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
                    if model_name != "UNet3": 
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
                        criterion = torch.nn.BCELoss()
                        loss = criterion(masks_pred_prob, masks_true_one_hot)

                    # Accumulate loss
                    total_loss += loss.item()

                    # Updates the progress bar
                    progress_bar.update(images.shape[0])

    # Sets the model to train mode again
    model.train()
    # Returns the weighted mean of the total 
    # loss according to the fluid voxels
    # Also avoids division by zero
    return total_loss / max(num_val_batches, 1)

@torch.inference_mode()
def evaluate_gan(generator: Module, dataloader: DataLoader, device: str):
    """
    Function used to evaluate the GAN model

    Args:
        generator (PyTorch Module object): generator that is being 
            trained
        dataloader (PyTorch DataLoader object): DataLoader 
            that contains the evaluation data
        device (str): indicates which PyTorch device is
            going to be used

    Return:
        Weighted mean of the loss across the considered 
        batches
    """
    # Sets the generator to evaluation mode
    generator.eval()
    # Calculates the number of batches 
    # used to validate the network
    num_val_batches = len(dataloader)
    # Initiates the loss as zero
    total_loss = 0

    # Creates a progress bar that tracks the number of batches that are being used in validation 
    with tqdm(dataloader, total=num_val_batches, desc='Validating Epoch', unit='batch', leave=True, position=0) as progress_bar:
        # Iterates through the batches in the batchloader
        for batch in dataloader:
                # Gets the stack from the dataloader
                stack = batch['stack']

                # Allocates the stack to the GPU or CPU, 
                # depending on the device that is being used
                stack = stack.to(device=device)

                # Splits the stack in three different images:
                # the previous image, the middle image that we 
                # aim to generate, and the following image 
                prev_imgs = stack[:,0,:,:].to(device=device)
                mid_imgs = stack[:,1,:,:].to(device=device)
                next_imgs = stack[:,2,:,:].to(device=device)

                # Generates the image using the generator and the 
                # previous and following images
                gen_imgs = generator(prev_imgs.detach(), next_imgs.detach())

                # Calculates the MS-SSIM for the original 
                # image and the generated image
                val_psnr = psnr(mid_imgs, gen_imgs)

                # Accumulates the loss for this 
                # validation 
                total_loss += val_psnr.item()

                # Updates the progress bar
                progress_bar.update(1)

    # Sets the model to train mode again
    generator.train()
    # Returns the weighted mean of the total 
    # loss according to the fluid voxels
    # Also avoids division by zero
    return total_loss / max(num_val_batches, 1)
