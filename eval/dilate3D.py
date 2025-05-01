import torch.nn.functional as F


def dilate3d(volume, kernel_size=3, strides=1):
    """
    3D morphological dilation via max-pool.

    Args:
      volume: 5D tensor of shape [batch, 1, height, width, depth].
      kernel_size: int or tuple/list of 3 ints, size of the cubic structuring element.
      strides: int, stride of the dilation (usually 1).
              Note: Padding calculation assumes strides=1.
    Returns:
      dilated: 5D tensor of same shape as input.
    """
    # PyTorch expects kernel_size as tuples/lists for 3D
    if isinstance(kernel_size, int):
        k = (kernel_size, kernel_size, kernel_size)
    else:
        k = tuple(kernel_size)

    p = tuple([(dim_k - 1) // 2 for dim_k in k]) #the padding is -inf by default in F.max_pool3d and we need anyting <= 0

    return F.max_pool3d(volume, kernel_size=k, stride=strides, padding=p)

