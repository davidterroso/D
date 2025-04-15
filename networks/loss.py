import torch
from numpy import array, float32
from torch.nn import BCEWithLogitsLoss, Conv2d, Module, MSELoss, Parameter
from torch.nn.functional import avg_pool2d, conv2d, pad

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

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = conv2d(input, win, stride=1, padding=0, groups=C)
    out = conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim_tensor(X, Y,
          data_range,
          win,
          K=(0.01, 0.03)):
          
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
        
    return ssim_per_channel, cs

class MS_SSIM(Module):
    '''
    Multi scale SSIM loss. Refer from:
    - https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    '''

    def __init__(self):
        super(MS_SSIM, self).__init__()
        
    def forward(self, gen_frames, gt_frames):
        return 1 - self._cal_ms_ssim(gen_frames, gt_frames)
        
    def _cal_ms_ssim(self, gen_tensors, gt_tensors):
        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        if torch.cuda.is_available():
            weights = weights.to('cuda')
        
        gen = gen_tensors
        gt = gt_tensors
        levels = weights.shape[0]
        mcs = []
        win_size = 3 if gen_tensors.shape[3] < 256 else 11
        win = _fspecial_gauss_1d(size=win_size, sigma=1.5)
        win = win.repeat(gen.shape[1], 1, 1, 1)
        
        for i in range(levels):
            ssim_per_channel, cs = _ssim_tensor(gen, gt, data_range=1.0, win=win)
            
            if i < levels - 1: 
                mcs.append(torch.relu(cs))
                padding = (gen.shape[2] % 2, gen.shape[3] % 2)
                gen = avg_pool2d(gen, kernel_size=2, padding=padding)
                gt = avg_pool2d(gt, kernel_size=2, padding=padding)
        
        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    
        return ms_ssim_val.mean()

def gdl_loss(gen_frames, gt_frames, alpha=2, cuda=True):
    '''
    From original version on https://github.com/wileyw/VideoGAN/blob/master/loss_funs.py
    which was referenced from Deep multi-scale video prediction beyond mean square error paper    
    :param gen_frames: generated output tensors
    :param gt_frames: ground truth tensors
    :param alpha: The power to which each gradient term is raised.
    '''
    filter_x_values = array(
        [
            [[[-1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]],
            [[[0, 0, 0]], [[0, 0, 0]], [[-1, 1, 0]]],
        ],
        dtype=float32,
    )
    filter_x = Conv2d(3, 3, (1, 3), padding=(0, 1))
    
    filter_y_values = array(
        [
            [[[-1], [1], [0]], [[0], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[-1], [1], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[0], [0], [0]], [[-1], [1], [0]]],
        ],
        dtype=float32,
    )
    filter_y = Conv2d(3, 3, (3, 1), padding=(1, 0))
        
    filter_x.weight = Parameter(torch.from_numpy(filter_x_values))  # @UndefinedVariable
    filter_y.weight = Parameter(torch.from_numpy(filter_y_values))  # @UndefinedVariable
    
    dtype = torch.FloatTensor if not cuda else torch.cuda.FloatTensor  # @UndefinedVariable
    filter_x = filter_x.type(dtype)
    filter_y = filter_y.type(dtype)
    
    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)
    
    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)  # @UndefinedVariable
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)  # @UndefinedVariable
    
    grad_total = torch.stack([grad_diff_x, grad_diff_y])  # @UndefinedVariable
    
    return torch.mean(grad_total)  # @UndefinedVariable

class GDL(Module):
    '''
    Gradient different loss function
    Target: reduce motion blur 
    '''
    
    def __init__(self, cuda_used=True):
        super(GDL, self).__init__()
        self.cuda_used = torch.cuda.is_available() and cuda_used

    def forward(self, gen_frames, gt_frames):
        return gdl_loss(gen_frames, gt_frames, cuda=self.cuda_used)

def generator_loss(discriminator, generated_imgs, expected_imgs, valid_label):
    adv_lambda = 0.05
    l1_lambda = 1.0
    gdl_lambda = 1.0
    ms_ssim_lambda = 6.0
    gd_loss = GDL().cuda()
    gd_loss = gd_loss(generated_imgs, expected_imgs)
    adv_loss = BCEWithLogitsLoss().cuda()
    adv_loss = adv_loss(discriminator(generated_imgs), valid_label).cuda()
    l1_loss = MSELoss().cuda()
    l1_loss = l1_loss(generated_imgs, expected_imgs)
    ms_ssim_loss = MS_SSIM().cuda()
    ms_ssim_loss = ms_ssim_loss(generated_imgs, expected_imgs)
    g_loss = adv_lambda * adv_loss \
            + l1_lambda * l1_loss \
            + gdl_lambda * gd_loss \
            + ms_ssim_lambda * ms_ssim_loss

    return adv_loss, g_loss

def discriminator_loss(ground_truth_distingue, fake_distingue, valid_label, fake_label):
    adv_loss = BCEWithLogitsLoss().cuda()
    real_loss = adv_loss(ground_truth_distingue, valid_label)
    fake_loss = adv_loss(fake_distingue, fake_label)

    return real_loss, fake_loss, (real_loss + fake_loss) / 2
