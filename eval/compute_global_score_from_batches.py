import torch

def aggregate_batch_metrics(
    batch_metrics,
    dilated_dice_score_coef=0.5,
    F_beta_coef=0.5,
    beta=2.0,
):
    """
    Aggregates a list of batch metrics into global metrics.

    Args:
        batch_metrics (list of dict): Each dict contains keys:
            'dilated_dice_score', 'batch_size',
            'true_positives', 'false_positives', 'false_negatives'
        *_coef: coefficients for the combined score

    Returns:
        dict: Aggregated metrics.
    """
    total_size = sum(b['batch_size'] for b in batch_metrics)
    dilated_dice_score = sum(b['dilated_dice_score'] * b['batch_size'] for b in batch_metrics) / total_size
    
    # Initialize true positives, false positives, and false negatives   
    true_positives = sum(b['true_positives'] for b in batch_metrics)
    false_positives = sum(b['false_positives'] for b in batch_metrics)
    false_negatives = sum(b['false_negatives'] for b in batch_metrics)

    precision_score = true_positives / (true_positives + false_positives + 1e-6)
    recall_score = true_positives / (true_positives + false_negatives + 1e-6)

    F_betas = (1 + beta**2) * (precision_score * recall_score) / ((beta**2 * precision_score) + recall_score + 1e-6)
    F_betas_mean = torch.mean(F_betas)


    combined_score = (
        dilated_dice_score * dilated_dice_score_coef +
        F_betas_mean * F_beta_coef
    )

    return {
        'F_beta': F_betas_mean,
        'dilated_dice_score': dilated_dice_score,
        'combined_score': combined_score
    }