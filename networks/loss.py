import torch

def multiclass_balanced_cross_entropy_loss(y_true, y_pred, batch_size, n_classes, eps=1e-7):
    """
    Loss function for the background segmentation on the network

    Args:
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
    # You may want to extend this to all classes if needed.
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

def dice_coefficient(prediction, target, num_classes, epsilon=1e-6):
    dice_scores = []
    voxel_counts = []
    total_intersection = 0
    total_union = 0
    
    for class_idx in range(1, num_classes + 1):
        pred_class = (prediction == class_idx).float()
        target_class = (target == class_idx).float()
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())
        voxel_counts.append(target_class.sum().item())
        total_intersection += intersection
        total_union += union
    
    total_dice = (2. * total_intersection + epsilon) / (total_union + epsilon)
    return dice_scores, voxel_counts, total_dice.item()
