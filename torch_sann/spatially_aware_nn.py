import torch

if torch.cuda.is_available():
    import torch_sann.spatially_aware_nn_cuda

def spatially_aware_nn(x, y, pos_x, pos_y, pos_dist_threshold):
    r"""Finds for each element in :obj:`x` the nearest point in :obj:`y` that is at a
        position-wise infinity-norm distance of more than :obj:`pos_dist_threshold`.

    Args:
        x (Tensor): Matrix with L2 normalized features
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Matrix with L2 normalized features
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        pos_x (Tensor): Node position matrix
            :math:`\mathbf{P_X} \in \mathbb{R}^{N \times D}`.
        pos_y (Tensor): Node position matrix
            :math:`\mathbf{P_Y} \in \mathbb{R}^{N \times D}`.
        pos_dist_threshold (float): Threshold for position distance

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_sann import spatially_aware_nn

    .. testcode::
        TODO: FIX
        >>> x = torch.Tensor([[-1, 0], [-1, 0]])
        >>> pos_x = torch.Tensor([0, 5])
        >>> y = torch.Tensor([[-1, -2], [-1, 1], [1, 1], [1, -2]])
        >>> pos_y = torch.Tensor([0, 0, 5, 5])
        >>> assign_index = spatially_aware_nn(x, y, pos_x, pos_y, 1)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    
    pos_x = pos_x.view(-1, 1) if pos_x.dim() == 1 else pos_x
    pos_y = pos_y.view(-1, 1) if pos_y.dim() == 1 else pos_y

    assert x.dim() == 2 and pos_x.dim() == 2
    assert y.dim() == 2 and pos_y.dim() == 2

    assert x.size(1) == y.size(1)
    assert pos_x.size(1) == pos_y.size(1)

    assert x.size(0) == pos_x.size(0)
    
    assert y.size(0) == pos_y.size(0)

    if x.is_cuda:
        return torch_sann.spatially_aware_nn_cuda.spatially_aware_nn(x, y, pos_x, pos_y, pos_dist_threshold)
    else:
        raise NotImplemented('Please use CUDA for spatially aware nearest neighbors.')


