import torch
from numpy import array, float32, log10
from torch.nn import BCELoss, BCEWithLogitsLoss, Conv2d, Module, MSELoss, Parameter
from torch.nn.functional import avg_pool2d, conv2d, pad
from typing import Optional, Union

def resize_with_crop_or_pad(y_true: torch.Tensor, target_height: int, target_width: int):
    """
    Function used to match the dimensions of the predicted 
    image with the true image

    Args:
        y_true (PyTorch tensor): the original image
        target_height (int): desired height of the image, 
            extracted from the predicted image
        target_width (int): desired width of the image, 
            extracted from the predicted image
    
    Return:
        y_true (PyTorch): original image with the shape 
            of the predicted image
    """
    _, h, w, _ = y_true.shape

    # Checks which image is bigger in both axis, and 
    # if the true image is smaller, calculates the 
    # required padding
    pad_top = max((target_height - h) // 2, 0)
    pad_bottom = max(target_height - h - pad_top, 0)
    pad_left = max((target_width - w) // 2, 0)
    pad_right = max(target_width - w - pad_left, 0)

    # Checks which image is bigger in both axis, and 
    # if the true image is bigger, calculates the 
    # required padding
    crop_top = max((h - target_height) // 2, 0)
    crop_bottom = crop_top + min(h, target_height)
    crop_left = max((w - target_width) // 2, 0)
    crop_right = crop_left + min(w, target_width)

    # Crops if the original image is bigger than the predicted
    y_true = y_true[:, crop_top:crop_bottom, crop_left:crop_right]

    # Pads if the original image is smaller than the predicted
    y_true= pad(y_true, [pad_left, pad_top, pad_right, pad_bottom])

    # Returns the image with 
    # the correct dimensions
    return y_true


def multiclass_balanced_cross_entropy_loss(model_name: str, 
                                           y_true: torch.Tensor, 
                                           y_pred: torch.Tensor, 
                                           batch_size: int, 
                                           n_classes: int, 
                                           eps: float=1e-7):
    """
    Loss function for the background segmentation on the network

    Args:
        model_name (str): name of the model used in segmentation
        y_true (PyTorch tensor): ground-truth of the segmented 
            fluids and background        
        y_pred (PyTorch tensor): network's prediction of the 
            segmented fluids and background 
        batch_size (int): size of the batch
        eps (float): epsilon value used to prevent divisions by 
            zero. Default: 1e-7
    
    Return:
        (float): loss of the background segmentation
    """
    # Check to verify that the probabilities in one class correspond 
    # to 1 - the other, when dealing with two classes
    if n_classes == 2:
        # Calculates the difference between the values explained
        diff = torch.abs((1.0 - y_pred[..., 0]) - y_pred[..., 1])
        max_diff = torch.max(diff)
        # If the difference is above a threshold, an error is raised
        # (a requirement to be exactly the same would likely cause 
        # crashes due to minor approximations or floating point)
        if max_diff > 1e-4:
            raise ValueError(f"Softmax channel sum check failed. Max difference: {max_diff.item():.6f}")
        
    # In case the model is 2.5D, 
    # it needs to crop the images 
    # to evaluate, since the output 
    # shape is not the same as the 
    # input shape because of the 
    # unpadded convolutions
    if model_name == "2.5D":
        target_shape = list(y_pred.size())
        target_height = target_shape[1]
        target_width = target_shape[2]
        y_true = resize_with_crop_or_pad(y_true, target_height, target_width)

    # Casts y_true as float32 to allow 
    # higher precision calculations 
    y_true = y_true.to(torch.float32)
    
    # Limits predictions to an interval of [eps, 1 - eps] 
    # to avoid log(0) issues
    y_pred_ = torch.clamp(y_pred, min=eps, max=1. - eps)

    # Calculate balanced cross-entropy loss
    # y_true is one-hot encoded and y_pred_ 
    # are probabilities
    cross_ent = torch.log(y_pred_) * y_true
    # Sum over spatial dimensions
    # y_true and y_pred_ have shape (B, H, W, C)
    # Sum over height and width
    cross_ent = torch.sum(cross_ent, dim=[1, 2])
    # Reshapes the tensor to have shape (B,C)
    cross_ent = torch.reshape(cross_ent, (batch_size, n_classes))

    # Compute the sum of true labels for each class, to balance the loss
    # Sums the over height and width
    y_true_sum = torch.sum(y_true, dim=[1, 2])
    # Reshapes the tensor to have shape (B,C) and sums eps to avoid 
    # division by zero
    y_true_sum = torch.reshape(y_true_sum, (batch_size, n_classes)) + eps

    # Calculates the cross entropy 
    # balanced to the total number of 
    # positive voxels
    cross_ent = cross_ent / y_true_sum

    # Calculate Dice loss for the first class (index 0)
    g_0 = y_true[:, :, :, 0]   # shape (B, H, W)
    p_0 = y_pred_[:, :, :, 0]   # shape (B, H, W)

    true_pos = torch.sum((1. - p_0) * (1. - g_0))
    false_pos = torch.sum((1. - p_0) * g_0)
    false_neg = torch.sum(p_0 * (1. - g_0))
    dice_loss = 1. - ((2. * true_pos) / (2. * true_pos + false_pos + false_neg + eps))

    # Combine the losses
    # The negative sign on cross-entropy is because of the log function
    loss = (-0.5 * torch.mean(cross_ent, dim=-1, keepdim=False) + 0.5 * dice_loss).mean()

    # Returns the loss value
    return loss

def balanced_bce_loss(y_true: torch.Tensor, 
                      y_pred: torch.Tensor, 
                      batch_size: int, 
                      n_classes: int, 
                      eps: float=1e-7):
    """
    Loss function for the background segmentation on the network

    Args:
        model_name (str): name of the model used in segmentation
        y_true (PyTorch tensor): ground-truth of the segmented 
            fluids and background        
        y_pred (PyTorch tensor): network's prediction of the 
            segmented fluids and background 
        batch_size (int): size of the batch
        eps (float): epsilon value used to prevent divisions by 
            zero. Default: 1e-7
    
    Return:
        (float): loss of the background segmentation
    """
    # Check to verify that the probabilities in one class correspond 
    # to 1 - the other, when dealing with two classes
    if n_classes == 2:
        # Calculates the difference between the values explained
        diff = torch.abs((1.0 - y_pred[..., 0]) - y_pred[..., 1])
        max_diff = torch.max(diff)
        # If the difference is above a threshold, an error is raised
        # (a requirement to be exactly the same would likely cause 
        # crashes due to minor approximations or floating point)
        if max_diff > 1e-4:
            raise ValueError(f"Softmax channel sum check failed. Max difference: {max_diff.item():.6f}")

    # Casts y_true as float32 to allow 
    # higher precision calculations 
    y_true = y_true.to(torch.float32)
    
    # Limits predictions to an interval of [eps, 1 - eps] 
    # to avoid log(0) issues
    y_pred_ = torch.clamp(y_pred, min=eps, max=1. - eps)

    # Calculate balanced cross-entropy loss
    # y_true is one-hot encoded and y_pred_ 
    # are probabilities
    cross_ent = torch.log(y_pred_) * y_true
    # Sum over spatial dimensions
    # y_true and y_pred_ have shape (B, H, W, C)
    # Sum over height and width
    cross_ent = torch.sum(cross_ent, dim=[1, 2])
    # Reshapes the tensor to have shape (B,C)
    cross_ent = torch.reshape(cross_ent, (batch_size, n_classes))

    # Compute the sum of true labels for each class, to balance the loss
    # Sums the over height and width
    y_true_sum = torch.sum(y_true, dim=[1, 2])
    # Reshapes the tensor to have shape (B,C) and sums eps to avoid 
    # division by zero
    y_true_sum = torch.reshape(y_true_sum, (batch_size, n_classes)) + eps

    # Calculates the cross entropy 
    # balanced to the total number of 
    # positive voxels
    cross_ent = cross_ent / y_true_sum

    # The negative sign on cross-entropy is because of the log function
    loss = (- torch.mean(cross_ent, dim=-1, keepdim=False)).mean()

    # Returns the loss
    return loss

def dice_coefficient(model_name: str, prediction: torch.Tensor,
                      target: torch.Tensor, num_classes: int):
    """
    Calculates the dice coefficient for the slices received

    Args:
        model_name (str): name of the model used in segmentation
        prediction (PyTorch tensor): images predicted by the model
        target (PyTorch tensor): corresponding ground-truth
        num_classes (int): total number of possible classes

    Return:
        dice_scores (List[float]): List with all the calculated 
            Dice coefficient for the respective class in the 
            predicted mask
        voxel_counts (List[int]): List with all the number of 
            voxels of the corresponding class in the considered 
            ground-truth        
        union_counts (List[int]): List with all the number of 
            voxels that result from the union between the GT 
            and the predicted mask        
        intersection_counts (List[int]): List with all the number of 
            voxels that intersect between the GT and the predicted
            mask
        binary_dice (List[float]): List with all the the 
            calculated Dice coefficient for the predicted mask 
            when the classes are binarized as fluid or background

    """
    # In case the model is 2.5D, it needs to 
    # crop the images to evaluate, since the 
    # output shape is not the same as the 
    # input shape because of the unpadded 
    # convolutions
    if model_name == "2.5D":
        target_shape = list(prediction.size())
        target_height = target_shape[1]
        target_width = target_shape[2]
        target = resize_with_crop_or_pad(target, target_height, target_width)

    # Casts y_true as float32 to allow 
    # higher precision calculations 
    target = target.to(torch.float32)

    # Initiates the lists that contain the Dice scores, the voxel count per 
    # class in the true mask, the number of voxels resulting from the union 
    # of the two masks, and the number of voxels resulting from the 
    # intersection of the two masks
    dice_scores = []
    voxel_counts = []
    union_counts = []
    intersection_counts = []
    
    # Iterates through the possible classes
    for class_idx in range(0, num_classes):
        # Converts each mask to a binary mask where 1 
        # corresponds to the mask to evaluate and zero 
        # to all the others
        pred_class = (prediction == class_idx).float()
        target_class = (target == class_idx).float()
        # Calculates the sum of the intersections in each class
        intersection = (pred_class * target_class).sum()
        # Calculates the union of classes
        union = pred_class.sum() + target_class.sum()
        # Calculates the Dice coeficient
        if (intersection == 0) and (union == 0):
            dice = 1.
        else:
            dice = (2. * intersection) / (union)
        # Appends the value of the Dice coefficient to a list
        dice_scores.append(float(dice))
        # Appends the value of the number of voxels to a list
        voxel_counts.append(int(target_class.sum().item()))
        # Appends the union value to a list
        union_counts.append(int(union.item()))
        # Appends the intersection value to a list
        intersection_counts.append(int(intersection.item()))

    # Binarizes the prediction 
    # and target mask
    prediction = (prediction != 0).float()
    target = (target != 0).float()
    # Calculates the sum of the intersections in the 
    # binary mask
    intersection = (prediction * target).sum()
    # Calculates the union of classes
    union = prediction.sum() + target.sum()
    # Calculates the Dice coeficient
    if (intersection == 0) and (union == 0):
        binary_dice = 1.
    else:
        binary_dice = (2. * intersection) / (union)
        binary_dice = binary_dice.item()
        
    return dice_scores, voxel_counts, union_counts, intersection_counts, binary_dice

def gaussian_kernel(size, sigma):
    """
    Function used to create a one-dimensional 
    Gaussian Kernel

    Args:
        size (int): size of the Gaussian Kernel
        sigma (float): sigma of the normal 
            distribution used
    
    Returns:
        (PyTorch tensor): one-dimensional kernel 
            with shape (1 x 1 x size)
    """
    # Creates a set of coordinates relative to the center
    # centering them around 0
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    # Computes the Gaussian weights of the filter 
    # according to the distance to the center
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    # Normalizes the kernel values
    g /= g.sum()

    # Reshapes the kernel from shape S, 
    # where S represents the size to 
    # 1 x 1 x S
    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    """ 
    Function used to blur images 
    that are used as input using 
    a 1-D kernel

    Args:
        input (PyTorch tensor): 
            a batch of tensors to 
            be blured
        win (PyTorch tensor): 1-D 
            gaussian kernel

    Returns:
        (PyTorch tensor): blured 
            input
    """
    # Gets the number of channels in the image
    N, C, H, W = input.shape
    # Applies the convolution horizontal wise
    out = conv2d(input, win, stride=1, padding=0, groups=C)
    # Applies the convolution vertical wise, by transforming the Gaussian 
    # Kernel into a vertical filter 
    out = conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    # Returns the blured images
    return out

def ssim_tensor(X: torch.Tensor, Y: torch.Tensor,
          data_range: Optional[Union[int, float]],
          win: torch.Tensor,
          K: tuple=(0.01, 0.03)):
          
    """ 
    Function used to calculate the SSIM index 
    between two images, X and Y, where in this
    case one is the real image and the other is 
    a fake image
    
    Args:
        X (PyTorch tensor): true or fake image
        Y (PyTorch tensor): true or fake image
        win (PyTorch tensor): one-dimensional 
            Gaussian Kernel window
        data_range (float or int, optional): 
            value range of input images (usually
            1.0 or 255)

    Returns:
        (PyTorch Tensor): SSIM results for the X 
            and Y images
    """
    # Definition of the 
    # compensation variable
    compensation = 1.0

    # Constants to stabilize the division 
    # in the SSIM formula
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Allocates the Gaussian Kernel to the 
    # right device
    win = win.to(X.device, dtype=X.dtype)

    # Computes the local means of 
    # the input images using the 
    # Gaussian Kernel
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    # Squares the local means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # Multiplies the local means
    mu1_mu2 = mu1 * mu2

    # Computes the X local variance
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    # Computes the Y local variance
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    # Computes the covariance
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    # Calculates the contrast sensitivity map, checking how well 
    # the local contrast (variance) of the two images matches
    # alpha = beta = gamma = 1
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    # Computes the SSIM map based on the covariance and the contrast sensitivity 
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    
    # The SSIM and contrast sensitivity are averaged across 
    # the height and width for each channel resulting in a
    # single value per channel for each image
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
        
    return ssim_per_channel, cs

class MS_SSIM(Module):
    """
    Multi-Scale Structural Similarity Index Measure. This metric
    is used to analyze the similarity between the generated images 
    and the true images
    
    Refer from:
    - https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    - https://github.com/tnquang1416/frame_interpolation_GAN/blob/master/utils/loss.py
    """

    def __init__(self):
        """
        Iniates the MS_SSIM as a PyTorch 
        Module, allowing for computation 
        on the GPU
        
        Args:
            self (MS_SSIM Module): the
                MS_SSIM Module itself

        Returns:
            None
        """
        super(MS_SSIM, self).__init__()
        
    def forward(self, gen_frames: torch.Tensor, 
                gt_frames: torch.Tensor):
        """
        Calculates the forward step of the 
        module which corresponds to 1 - MS-SSIM
        for the generated images and their 
        corresponding real images

        Args:
            self (MS_SSIM Module): the
                MS_SSIM Module itself
            gen_frames (PyTorch tensor): images 
                generated by the generator
            gt_frames (PyTorch tensor): real
                corresponding images

        Returns:
            (PyTorch tensor): 1 - MS-SSIM value 
                for all images
        """
        return 1 - self.ms_ssim(gen_frames, gt_frames)
        
    def ms_ssim(self, gen_tensors, gt_tensors):
        """
        Calculates the MS-SSIM for the received images

        Args:
            self (MS_SSIM Module): the
                MS_SSIM Module itself
            gen_frames (PyTorch tensor): images 
                generated by the generator
            gt_frames (PyTorch tensor): matching
                reak images

        Returns:
            (PyTorch tensor): mean of the results 
                across all levels
        """
        
        # Sets the weights for each scale manually, as done in the 
        # referenced repositories
        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        
        # Assigns the tensor to the GPU 
        # if it is available
        if torch.cuda.is_available():
            weights = weights.to('cuda')
        
        # Renames the variables
        gen = gen_tensors
        gt = gt_tensors
        # Gets the number of weights 
        # in the weights tensor
        levels = weights.shape[0]
        
        # Initates a list that will 
        # contain the contrast 
        # sensitivity of each scale
        mcs = []
        # The window size used in the Gaussian Kernel is
        # defined according to the size of the image
        win_size = 3 if gen_tensors.shape[3] < 256 else 11
        # The Gaussian Kernel is created receiving as 
        # arguments the window size and the sigma
        win = gaussian_kernel(size=win_size, sigma=1.5)
        # This function is applied repeatedly 
        # for all the channels that compose the 
        # images
        win = win.repeat(gen.shape[1], 1, 1, 1)
        
        # Iterates through the multiple scales/levels
        for i in range(levels):
            # Calls the function that calculates the SSIM for the current images
            # It returns the SSIM for each channel and the contrast sensitivity
            ssim_per_channel, cs = ssim_tensor(gen, gt, data_range=255, win=win)
            
            # For all the scales, except the
            # last one, downsampling is performed
            if i < levels - 1: 
                # Appends the result of applying the ReLU function to 
                # the contrast sensitivity to the list that holds these values
                mcs.append(torch.relu(cs))
                # Calculates the padding needed for the image 
                # in case its dimensions are not divisible by two
                padding = (gen.shape[2] % 2, gen.shape[3] % 2)
                # Performs two-dimensionl average pooling in the 
                # images, downsampling them
                gen = avg_pool2d(gen, kernel_size=2, padding=padding)
                gt = avg_pool2d(gt, kernel_size=2, padding=padding)
        
        # Performs the ReLU activation in the SSIM values 
        # per channel, resulting in a tensor of shape B x C
        ssim_per_channel = torch.relu(ssim_per_channel)
        # Stacks the sum of the multiple contrast sensitivity 
        # and SSIM values of the different levels across the 
        # first dimension, ending with a tensor of shape L x B x C
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
        # Calculates the product of all the combined contrast sensitivity and 
        # SSIM raised to the power of the weights defined previously
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    
        # Returns the mean of all 
        # the values in the tensor
        return ms_ssim_val.mean()

def gdl_loss(gen_frames, gt_frames, alpha=2, cuda=True):
    """
    Function used to calculate the gradient differential 
    loss between two images. Extracted from the original 
    version on https://github.com/wileyw/VideoGAN/blob/master/loss_funs.py
    The main goal of this loss is to motivate the network 
    to also focus on the image's textures and edges 
    characteristics and not just the pixel differences 

    Args:
        gen_frames (PyTorch tensor): images generated by 
            the generator
    gt_frames (PyTorch tensor): the real ground truth 
        images that correspond to the fake images 
    alpha (int): the power to which each gradient term 
        is raised
    cuda (bool): indicates whether the network is being 
        trained on the GPU or not. The default value is 
        true
    
    Returns:
        (PyTorch tensor): mean of all the gradients in 
            from different dimensions
    """
    # In the following lines of code convolutional 
    # filters are created that compute spatial 
    # gradients for each channel. The first filter 
    # calculates the gradients across the X axis 
    # (along the width) while the second filter 
    # calculates the gradients across the Y axis 
    # (along the height)
    # What these filters basically do is calculate 
    # how much the intensity changes between 
    # neighbouring pixels

    # Initiates the weights of the first convolution 
    # filter
    filter_x_values = array(
        [
            [[[-1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[0, 0, 0]], [[-1, 1, 0]]],
        ],
        dtype=float32,
    )
    # Initiates the first convolution object
    filter_x = Conv2d(3, 3, (1, 3), padding=(0, 1))
    
    # Initiates the weights of the second convolution 
    # filter
    filter_y_values = array(
        [
            [[[-1], [1], [0]], [[0], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[-1], [1], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[0], [0], [0]], [[-1], [1], [0]]],
        ],
        dtype=float32,
    )
    # Initiates the second convolution object
    filter_y = Conv2d(3, 3, (3, 1), padding=(1, 0))
        
    # Sets the weights of the convolutions to those initialized above
    filter_x.weight = Parameter(torch.from_numpy(filter_x_values))
    filter_y.weight = Parameter(torch.from_numpy(filter_y_values))
    
    # Attributes the filters to the GPU if available
    dtype = torch.FloatTensor if not cuda else torch.cuda.FloatTensor
    filter_x = filter_x.type(dtype)
    filter_y = filter_y.type(dtype)
    
    # Calculates the differences between 
    # pixels for both generated images 
    # and real images, across height 
    # and width
    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)
    
    # Raises the difference between gradient matrices to the 
    # power alpha defined as argument
    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)
    
    # Stacks the gradients from both height and width together
    grad_total = torch.stack([grad_diff_x, grad_diff_y])
    
    # Calculates the mean across 
    # all channels and batches to 
    # one single value
    return torch.mean(grad_total)

class GDL(Module):
    """
    The main goal of this loss is to motivate the network 
    to also focus on the image's textures and edges 
    characteristics and not just the pixel differences 
    """
    
    def __init__(self, cuda_used: bool=True):
        """
        Initiates the GDL PyTorch module 
        and saves in the variable if the 
        GPU is the device being used

        Args:
            self (GDL Module): the GDL 
                PyTorch Module itself
            cuda_used (bool): flag that 
                indicates if the GPU is 
                the selected device. The 
                default value is true
        
        Returns:
            None
        """
        # Initiates the PyTorch Module
        super(GDL, self).__init__()
        # Saves a bool that indicates whether the GPU is being 
        # used or not
        self.cuda_used = torch.cuda.is_available() and cuda_used

    def forward(self, gen_frames, gt_frames):
        """
        Calculates the GDL loss between the 
        generated images and their 
        corresponding true images

        Args:
            self (GDL Module): the GDL 
                PyTorch Module itself
                gen_frames (PyTorch tensor):
                    images generated by the 
                    generator
                gt_frames (PyTorch tensor):
                    real images corresponding 
                    to the fake images received

        Returns:
            (PyTorch tensor): mean of all the 
                gradients in from different 
                dimensions in the received 
                images
        """
        return gdl_loss(gen_frames, gt_frames, cuda=self.cuda_used)

def generator_loss(device: str, discriminator: Module, 
                   generated_imgs: torch.Tensor,
                   expected_imgs: torch.Tensor, 
                   valid_label: torch.Tensor):
    """
    Calculates the loss of the generator. This loss is composed of four 
    different components, which have their respective weights. The first 
    loss corresponds to the binary cross entropy loss (BCE), the second 
    to the mean-squared error (MSE), the third to the multi-scale 
    structural similiarity index measure (MS-SSIM), and the last is the
    gradient differential loss (GDL).

    Args:
        device (str): name of the device that is being used in training
        discriminator (PyTorch Module): the discriminator network that
            will evaluate the image generated by the generator
        generated_imgs (PyTorch tensor): contains all the images 
            generated by the generator. Shape: B x 3 x H x W
        expected_imgs (PyTorch tensor): contains the ground-truth  
            corresponding to the generated images. Shape: B x 3 x H x W
        valid_label (PyTorch tensor): array of length B that contains 
            true labels

    Returns: 
        adv_loss (PyTorch tensor): result of the BCE
        g_loss (PyTorch tensor): result of the combination of all losses
            
    """
    # Sets the weights 
    # for all the 
    # components of the 
    # loss
    gdl_lambda = 1.0
    adv_lambda = 0.05
    l1_lambda = 1.0
    ms_ssim_lambda = 6.0

    # Initiates the GDL loss and 
    # allocates it to the used device
    gd_loss = GDL().to(device)
    # Calculates the GDL loss according to the 
    # predicted and expected images
    gd_loss = gd_loss(generated_imgs, expected_imgs)
    # Initiates the BCE loss and allocates it 
    # to the used device
    adv_loss = BCEWithLogitsLoss().to(device)
    # Calculates the BCE loss for the classification predicted by the 
    # discriminator and compares it to the true label. This might 
    # sound counter-intuitive since the loss is being calculated for 
    # the wrong label (the fake image is calculated with the true 
    # label). However, what this aims to portray is how well the
    # each image can fool the discriminator. If the discriminator 
    # fully believes that the fake image is a true image, the loss is
    # zero. Otherwise, it is as great as the certainty of the 
    # discriminator that these images are fake. The reason this 
    # weight is so small is because the BCE loss can take large 
    # values (significantly larger than one, sometimes)
    adv_loss = adv_loss(discriminator(generated_imgs), valid_label)
    # Initiates the MSE loss and 
    # allocates it to the used device
    l1_loss = MSELoss().to(device)
    # Calculates the MSE loss between the generated 
    # and expected images
    l1_loss = l1_loss(generated_imgs, expected_imgs)
    # Initiates the MS-SSIM loss and 
    # allocates it to the device that 
    # is used 
    ms_ssim_loss = MS_SSIM().to(device)
    # Calculates the MS-SSIM loss between the generated and 
    # expected images
    ms_ssim_loss = ms_ssim_loss(generated_imgs, expected_imgs)
    # Calculates the generator loss 
    # as a mixture of the four losses
    # according to their respective weights
    g_loss = adv_lambda * adv_loss \
            + l1_lambda * l1_loss \
            + gdl_lambda * gd_loss \
            + ms_ssim_lambda * ms_ssim_loss

    return adv_loss, g_loss

def discriminator_loss(device: str, 
                       ground_truth_distingue: torch.Tensor, 
                       fake_distingue: torch.Tensor, 
                       valid_label: torch.Tensor, 
                       fake_label: torch.Tensor):
    """
    This function calculates the loss of the discriminator, 
    receiving as input the predicted and real images, as well 
    as their respective true labels, from which it calculates
    how well the discriminator is at determining whether real 
    images are real and fake images are fake

    Args:
        device (str): device in which the loss will be 
            calculated
        ground_truth_distingue (PyTorch tensor): probability 
            of the real image being real, according to the
            discriminator
        fake_distingue (PyTorch tensor): probability of the 
            fake image being real, according to the 
            discriminator
        valid_label (PyTorch tensor): labels of the true 
            images
        fake_label (PyTorch tensor): labels of the fake 
            images 
        
    Returns:
        real_loss (PyTorch tensor): BCE loss for the real 
            images
        fake_loss (PyTorch tensor): BCE loss for the fake 
            images
        (PyTorch tensor): mean of the real_loss and 
            fake_loss which designates the discriminator
            loss
    """
    # Initiates the BCE loss and allocates it 
    # to the device that is being used
    adv_loss = BCEWithLogitsLoss().to(device)
    # Calculates the BCE for the real images and their 
    # respective labels
    real_loss = adv_loss(ground_truth_distingue, valid_label)
    # Calculates the BCE for the fake images and their 
    # respective labels
    fake_loss = adv_loss(fake_distingue, fake_label)

    return real_loss, fake_loss, (real_loss + fake_loss) / 2

def pix2pix_generator_loss(device: str, discriminator: Module, 
                            generated_imgs: torch.Tensor,
                            expected_imgs: torch.Tensor, 
                            valid_label: torch.Tensor):
    """
    Calculates the loss of the Pix2Pix generator. This loss is composed 
    of two different components, which have their respective weights. 
    The first loss corresponds to the binary cross entropy loss (BCE) 
    and the second to the mean-squared error (MSE)

    Args:
        device (str): name of the device that is being used in training
        discriminator (PyTorch Module): the discriminator network that
            will evaluate the image generated by the generator
        generated_imgs (PyTorch tensor): contains all the images 
            generated by the generator. Shape: B x 3 x H x W
        expected_imgs (PyTorch tensor): contains the ground-truth  
            corresponding to the generated images. Shape: B x 3 x H x W
        valid_label (PyTorch tensor): array of length B that contains 
            true labels

    Returns: 
        adv_loss (PyTorch tensor): result of the BCE
        g_loss (PyTorch tensor): result of the combination of all losses
            
    """
    # Sets the weights 
    # for all the 
    # components of the 
    # loss
    adv_lambda = 1.0
    l1_lambda = 100.0

    # Initiates the BCE loss and allocates it 
    # to the used device
    adv_loss = BCELoss().to(device)
    # Calculates the BCE loss for the classification predicted by the 
    # discriminator and compares it to the true label. This might 
    # sound counter-intuitive since the loss is being calculated for 
    # the wrong label (the fake image is calculated with the true 
    # label). However, what this aims to portray is how well the
    # each image can fool the discriminator. If the discriminator 
    # fully believes that the fake image is a true image, the loss is
    # zero. Otherwise, it is as great as the certainty of the 
    # discriminator that these images are fake. The reason this 
    # weight is so small is because the BCE loss can take large 
    # values (significantly larger than one, sometimes)
    adv_loss = adv_loss(discriminator(generated_imgs), valid_label)
    # Initiates the MSE loss and 
    # allocates it to the used device
    l1_loss = MSELoss().to(device)
    # Calculates the MSE loss between the generated 
    # and expected images
    l1_loss = l1_loss(generated_imgs, expected_imgs)
    # Calculates the generator loss 
    # as a mixture of the two losses
    # according to their respective weights
    g_loss = adv_lambda * adv_loss + l1_lambda * l1_loss

    return adv_loss, g_loss

def psnr(img1: torch.Tensor, img2: torch.Tensor):
    """
    Calculates the Peak Signal-to-Noise Ratio 
    (PSNR) metric from two tensors of images

    Args:
        img1 (PyTorch tensor): one of the 
            images that will be compared, 
            with range 0-255
        img2 (PyTorch tensor): the other 
            image that will be compared, 
            with range 0-255

    Returns:
        psnr (PyTorch tensor | float): PSNR 
        for the received images. In case the 
        images are equal, returns an 
        infinite float
    """
    # Calculates the difference 
    # between the images
    diff = (img1 - img2)
    # Squares the difference
    diff = diff ** 2

    # In case the images have no 
    # differences returns infinite
    if diff.sum().item() == 0:
        return float('inf')
        
    # Calculates the mean of the 
    # values, obtaining the Root 
    # Mean Squared Error (RMSE)
    rmse = diff.mean().item()

    # Calculates the PSNR using the RMSE 
    psnr_value = 20 * log10(1.0) - 10 * log10(rmse)

    return psnr_value
