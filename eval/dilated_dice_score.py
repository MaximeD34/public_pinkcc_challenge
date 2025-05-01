import torch
import torch.nn.functional as F

from eval.dilate3D import dilate3d

def compute_dilated_dice_score_multiclass(target_one_hot, pred_one_hot, kernel_size_dilation=3, epsilon=1e-6):
    """
    Calculates the original Dilated Dice score for multiclass 3D volumes,
    ***excluding classes not present in the ground truth from the average***.

    Uses the formula: (|A ∩ D(B)| + |B ∩ D(A)|) / (|A| + |B|) per class.

    Args:
      target_one_hot: Ground truth tensor, shape [batch, num_classes, h,w,d].
              Expected to be one-hot encoded.
      pred_one_hot: Predicted tensor, shape [batch, num_classes, h,w,d].
              Expected to be one-hot encoded (derived from predicted class indices).
      kernel_size_dilation: Size of the dilation kernel (structuring element) for D(A) and D(B).
      epsilon: Small constant to prevent division by zero.

    Returns:
      Scalar Dilated Dice score averaged over present classes and the batch.
    """


    #Check dimensions
    if target_one_hot.ndim != 5 or pred_one_hot.ndim != 5:
        raise ValueError("Input tensors must have 5 dimensions: [batch, num_classes, h,w,d].")
    if target_one_hot.shape != pred_one_hot.shape:
        raise ValueError("Input tensors must have the same shape.")
    #Unsure it is one-hot
    if not (target_one_hot.sum(dim=1) == 1).all():
        raise ValueError("Target tensor must be one-hot encoded.")
    if not (pred_one_hot.sum(dim=1) == 1).all():
        raise ValueError("Predicted tensor must be one-hot encoded.")
    # Check if the kernel size is odd
    if isinstance(kernel_size_dilation, int):
        if kernel_size_dilation % 2 == 0:
            raise ValueError("Kernel size for dilation must be odd.")
    elif isinstance(kernel_size_dilation, (list, tuple)):
        if any(size % 2 == 0 for size in kernel_size_dilation):
            raise ValueError("All kernel sizes for dilation must be odd.")
    else:
        raise ValueError("Kernel size must be an int, list, or tuple.")
    
    # Ensure inputs are float for calculations
    target_one_hot = target_one_hot.float()
    pred_one_hot = pred_one_hot.float()

    # Infer number of classes from the tensor shape
    num_classes = target_one_hot.shape[1]

    dilated_dice_scores_per_class = []
    nb_positive_target_per_class = [] # To track presence for averaging

    # Iterate over each class (excluding background)
    for c in range(1, num_classes): # Start from 1 to skip background class
        
        target_c = target_one_hot[:, c:c+1, :, :, :] # [batch, 1, h,w,d]. note: c:c+1 keeps the dimension for the dilation
        pred_c = pred_one_hot[:, c:c+1, :, :, :] # [batch, 1, h,w,d]

        #Sum over spatial dimensions to get the number of positive voxels for the class c
        nb_positive_target = torch.sum(target_c, dim=[2, 3, 4]) # [batch, 1]
        nb_positive_pred = torch.sum(pred_c, dim=[2, 3, 4]) # [batch, 1]
        
        nb_positive_target_per_class.append(nb_positive_target) #used for averaging only present classes

        # Dilate predictions and ground truth
        dilated_pred_c = dilate3d(pred_c, kernel_size=kernel_size_dilation) # [batch, 1, h,w,d]
        dilated_target_c = dilate3d(target_c, kernel_size=kernel_size_dilation) # [batch, 1, h,w,d]

        intersect_count_target_over_dilated_pred = torch.sum(target_c * dilated_pred_c, dim=[2, 3, 4]) # Shape [batch, 1]
        intersect_count_pred_over_dilated_target = torch.sum(pred_c * dilated_target_c, dim=[2, 3, 4]) # Shape [batch, 1]

        numerator = intersect_count_target_over_dilated_pred + intersect_count_pred_over_dilated_target # Shape [batch, 1]
        denominator = nb_positive_target + nb_positive_pred + epsilon # Shape [batch, 1]

        # Calculate dilated dice coefficient per batch item for this class
        per_item_per_class_dilated_dice = numerator / denominator # Shape [batch, 1]
        dilated_dice_scores_per_class.append(per_item_per_class_dilated_dice)
    
    all_classes_dilated_dice = torch.cat(dilated_dice_scores_per_class, dim=1) # Shape [batch, num_classes-1]
    
    # Concatenate true volumes (ie volumes with at least one positive in the target) across classes: Shape [batch, num_classes]
    all_nb_positive_target = torch.cat(nb_positive_target_per_class, dim=1) # Shape [batch, num_classes-1]

    # Create a mask for classes present in the ground truth for each batch item
    presence_mask = (all_nb_positive_target > 0) # Shape [batch, num_classes-1]

    # Calculate the number of present classes for each item in the batch for averaging. Normaly it should be 2 for each batch item
    num_present_classes_per_item = torch.sum(presence_mask.float(), dim=1) # Shape [batch]
    
    masked_scores = all_classes_dilated_dice * presence_mask.float() # Shape [batch, num_classes-1]
    sum_scores_per_item = torch.sum(masked_scores, dim=1) # Shape [batch]
    mean_dilated_dice_per_item = sum_scores_per_item / (num_present_classes_per_item + epsilon) # Shape [batch]

    # Calculate the final mean Dilated Dice score averaged over the batch
    mean_dilated_dice = torch.mean(mean_dilated_dice_per_item)

    return mean_dilated_dice


def compute_dilated_dice_score_from_volumes(target, pred, num_classes=3, kernel_size_dilation=3, epsilon=1e-6):
    """
    Comptes the multiclass Dilated Dice score.

    Takes ground truth with integer labels [batch, h,w,d] and predictions
    with integer labels [batch, h,w,d], converts both to one-hot, and
    calls compute_dilated_dice_score_multiclass.

    Args:
        target: Ground truth labels of shape [batch, h,w,d].
        pred: Predicted labels of shape [batch, h,w,d].
        num_classes: The total number of classes (including background).
        kernel_size_dilation: Size of the dilation kernel (structuring element).
        epsilon: Small constant to prevent division by zero.

    Returns:
      Scalar Dilated Dice score averaged over classes and the batch.
    """
    # Ensure inputs are LongTensor for one_hot
    target = target.long()
    pred = pred.long()

    target_one_hot = F.one_hot(target, num_classes=num_classes) # [batch, h,w,d, num_classes]
 
    pred_one_hot = F.one_hot(pred, num_classes=num_classes) # [batch, h,w,d, num_classes]


    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float() # [batch, num_classes, h,w,d]
    pred_one_hot = pred_one_hot.permute(0, 4, 1, 2, 3).float() # [batch, num_classes, h,w,d]

    # Call the main multiclass dilated dice score function
    score = compute_dilated_dice_score_multiclass(
        target_one_hot,
        pred_one_hot,
        kernel_size_dilation=kernel_size_dilation,
        epsilon=epsilon
    )

    return score