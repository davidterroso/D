import torch
from torch.nn.functional import pad

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
        
    return dice_scores, voxel_counts, union_counts, intersection_counts 
