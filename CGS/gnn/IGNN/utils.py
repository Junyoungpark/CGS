import numpy as np
import scipy.sparse as sp
import torch


# IGNN codes from the official implementation
# https://github.com/SwiftieH/IGNN/tree/5d4fd69951c10cda452773ac92f98440ffebd453

def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
