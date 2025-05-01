import torch
import nibabel as nib
import os

torch.classes.__path__ = []

def nii_gz_to_torch_tensor(file_name):
    img = nib.load(file_name)
    data = img.get_fdata()

    # Put on GPU if possible (MPS for Mac silicon, CUDA for Nvidia, else CPU)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    data = torch.tensor(data, device=device, dtype=torch.int64)


    #TODO do it the right way
    #put any voxel that is not 0, 1 or 2 to 0
    data = torch.where(data > 2, 0, data)
    return data

def pad_tensor(tensor, target_shape):
    current_shape = tensor.shape
    padding = [(0, max(0, target_shape[i] - current_shape[i])) for i in range(len(current_shape))]
    padding = padding[::-1]
    pad_flat = [p for pair in padding for p in pair]
    padded_tensor = torch.nn.functional.pad(tensor, pad_flat)
    return padded_tensor

def paired_file_names_to_stacked_tensors(pred_file_names, target_file_names):
    # Match by basename
    pred_map = {os.path.basename(f): f for f in pred_file_names}
    target_map = {os.path.basename(f): f for f in target_file_names}
    common_names = sorted(set(pred_map) & set(target_map))
    if not common_names:
        raise ValueError("No matching file names found between pred and target lists.")
    pred_matched = [pred_map[name] for name in common_names]
    target_matched = [target_map[name] for name in common_names]

    pred_tensors = []
    target_tensors = []
    max_depth = 0
    max_height = 0
    max_width = 0
    distinct_values = set()
    for pred_file, target_file in zip(pred_matched, target_matched):
        print(f"Processing pair: {os.path.basename(pred_file)}")
        pred_tensor = nii_gz_to_torch_tensor(pred_file)
        target_tensor = nii_gz_to_torch_tensor(target_file)

        if pred_tensor.shape != target_tensor.shape:
            raise ValueError(f"Shape mismatch for {os.path.basename(pred_file)}: pred {pred_tensor.shape}, target {target_tensor.shape}")
        pred_tensors.append(pred_tensor)
        target_tensors.append(target_tensor)
        if pred_tensor.ndim != 3:
            raise ValueError("Tensor must have 3 dimensions: [tensor_depth, tensor_height, tensor_width].")
        max_depth = max(max_depth, pred_tensor.shape[0])
        max_height = max(max_height, pred_tensor.shape[1])
        max_width = max(max_width, pred_tensor.shape[2])

    pred_padded = [pad_tensor(t, (max_depth, max_height, max_width)) for t in pred_tensors]
    target_padded = [pad_tensor(t, (max_depth, max_height, max_width)) for t in target_tensors]

    pred_stacked = torch.stack(pred_padded, dim=0)
    target_stacked = torch.stack(target_padded, dim=0)
    
    return pred_stacked, target_stacked