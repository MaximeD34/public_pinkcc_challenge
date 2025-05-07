import torch
from eval.dilated_dice_score import compute_dilated_dice_score_from_volumes

def eval_pipeline(
        target, 
        pred, 
        num_classes, 
        F_beta_coef=0.5,
        dilated_dice_score_coef=0.5, 
        beta=2.0
        ):
    """
    Evaluation pipeline for the DataChallenge.
    
    Args:
        target (torch.Tensor): Target tensor of shape [batch, width, height, depth].
        pred (torch.Tensor): Predicted tensor of shape [batch, width, height, depth], with the same shape as target.
        num_classes (int): Number of classes, the target and pred tensors should contain values in the range [0, num_classes-1].
        dice_score_coef (float): Coefficient for the standard Dice score.
        dilated_dice_score_coef (float): Coefficient for the dilated Dice score.
        precision_coef (float): Coefficient for the precision score.
        recall_coef (float): Coefficient for the recall score.

    Returns:
        dict: Dictionary containing the scores for Dice, dilated Dice, precision, recall, and combined score.
            - 'dice_score': Standard Dice score.
            - 'dilated_dice_score': Dilated Dice score.
            - 'precision_score': Precision score.
            - 'recall_score': Recall score.
            - 'combined_score': Combined score based on the coefficients provided.
    """

    #check that the coefs sum to 1
    if (F_beta_coef + dilated_dice_score_coef) != 1:
        raise ValueError("Coefficients must sum to 1.")
    # Check dimensions
    if target.ndim != 4 or pred.ndim != 4:
        raise ValueError("Input tensors must have 4 dimensions: [batch, width, height, depth].")
    if target.shape != pred.shape:
        raise ValueError("Input tensors must have the same shape.")
    
    #check that the tensor values are integers in 0, 1, ..., num_classes-1
    if not (target.dtype == torch.int64 or target.dtype == torch.int32):
        raise ValueError("Target tensor must be of integer type.")
    if not (pred.dtype == torch.int64 or pred.dtype == torch.int32):
        raise ValueError("Predicted tensor must be of integer type.")
    if not (target.min() >= 0 and target.max() < num_classes):
        raise ValueError("Target tensor must contain values in the range [0, num_classes-1].")
    if not (pred.min() >= 0 and pred.max() < num_classes):
        raise ValueError("Predicted tensor must contain values in the range [0, num_classes-1].")

    target = target.float()
    pred = pred.float()

    dilated_dice_score = compute_dilated_dice_score_from_volumes(target, pred, num_classes) #shape []

    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []

    for c in range(1, num_classes): # Skip background class (0)
        target_c = (target == c).float()
        
        pred_c = (pred == c).float()

        tp = torch.sum(target_c * pred_c)
        fp = torch.sum((1 - target_c) * pred_c)
        fn = torch.sum(target_c * (1 - pred_c))

        true_positives.append(tp)
        false_positives.append(fp)
        false_negatives.append(fn)
        tn = torch.sum((1 - target_c) * (1 - pred_c))
        true_negatives.append(tn)

    true_positives = torch.stack(true_positives)
    false_positives = torch.stack(false_positives)
    false_negatives = torch.stack(false_negatives)
    true_negatives = torch.stack(true_negatives)

    precision_score = true_positives / (true_positives + false_positives + 1e-6)
    recall_score = true_positives / (true_positives + false_negatives + 1e-6)

    F_betas = (1 + beta**2) * (precision_score * recall_score) / ((beta**2 * precision_score) + recall_score + 1e-6)

    # If one of the classes is not present in the batch, set the score to NaN for that class
    for c in range(num_classes - 1):
        if true_positives[c] == 0 and false_positives[c] == 0 and false_negatives[c] == 0:
            precision_score[c] = float('nan')
            recall_score[c] = float('nan')
            F_betas[c] = float('nan')

    precision_score_mean = torch.nanmean(precision_score[:])
    recall_score_mean = torch.nanmean(recall_score[:])
    F_betas_mean = torch.nanmean(F_betas)

    batch_size = target.shape[0]

    combined_score = (
        dilated_dice_score * dilated_dice_score_coef +
        F_betas_mean * F_beta_coef
    )

    del target
    del pred
    
    return {
        'dilated_dice_score': dilated_dice_score,
        'batch_size': batch_size,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'num_classes': num_classes,
        'P_1': precision_score[0],
        'P_2': precision_score[1],
        'R_1': recall_score[0],
        'R_2': recall_score[1],
        'P_mean': precision_score_mean,
        'R_mean': recall_score_mean,
        'F_beta_1': F_betas[0],
        'F_beta_2': F_betas[1],
        'F_beta_mean': F_betas_mean,
        'combined_score': combined_score
    }
