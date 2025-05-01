from web.file_names_to_stacked_tensor import paired_file_names_to_stacked_tensors
from eval.eval_pipeline import eval_pipeline

def batch_evaluate_files(
    pred_file_paths,
    target_file_paths,
    num_classes,
    batch_size=5,
    dilated_dice_score_coef=0.5,
    F_beta_coef=0.5,
    beta=2.0,

):
    assert len(pred_file_paths) == len(target_file_paths), "Prediction and target lists must be the same length"
    total = len(pred_file_paths)
    for i in range(0, total, batch_size):
        pred_batch = pred_file_paths[i:i+batch_size]
        target_batch = target_file_paths[i:i+batch_size]
        pred_stacked, target_stacked = paired_file_names_to_stacked_tensors(pred_batch, target_batch)
        batch_metrics = eval_pipeline(
            target_stacked,
            pred_stacked,
            num_classes=num_classes,
            dilated_dice_score_coef=dilated_dice_score_coef,
        )
        yield batch_metrics, pred_batch, target_batch