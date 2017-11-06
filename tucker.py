import numpy as np
from tensorly.tenalg import partial_svd,norm
from tensorly.tucker import tucker_to_tensor
from tensorly.base import unfold
"""
first two dimensions of tensor corresponds to T, S and reduced to r1, r2
tensor[T,S,D,D] -> [r2,r1,D,D]*[T,r2]*[S,r1]

"""
def HOOI(tensor, r1, r2, num_iter=500, error_print=True, tol=10e-5):
    """
    U:   [r1, S, 1, 1]
    Core:[r2,r1, kernel_w, kernel_h]
    V:   [T, r2, 1, 1]
    """
    w_out_channel,w_in_channel,kernel_w,kernel_h = [i for i in tensor.shape]

    # compute sparse ratio of W
    sparse_ratio = (tensor<0.005).astype(np.float32).mean()
    print 'sparse ratio is ',sparse_ratio
    print tensor.shape,tensor.min(),tensor.max()
    for i in np.arange(-0.1,0.282,0.03):
        boolvalue=(tensor>i) & (tensor<(i+0.03))
        ratio = boolvalue.astype(np.float32).mean()
        print ratio


    # tucker-2 decomposition
    ranks = [r2,r1]

    ### tucker step1: HOSVD init
    factors = []
    for mode in range(2):
        eigenvecs, _, _ = partial_svd(unfold(tensor, mode), n_eigenvecs=ranks[mode])
        factors.append(eigenvecs)
    factors.append(np.eye(kernel_w))
    factors.append(np.eye(kernel_h))

    ### HOOI decomposition
    rec_errors = []
    norm_tensor = norm(tensor, 2)

    for iteration in range(num_iter):
        for mode in range(2):
            core_approximation = tucker_to_tensor(tensor, factors, skip_factor=mode, transpose_factors=True)
            eigenvecs, _, _ = partial_svd(unfold(core_approximation, mode), n_eigenvecs=ranks[mode])
            factors[mode] = eigenvecs

        core = tucker_to_tensor(tensor, factors, transpose_factors=True)
        reconstruct_tensor = tucker_to_tensor(core, factors, transpose_factors=False)# reconstruct tensor
        rec_error1 = norm(tensor-reconstruct_tensor,2)/norm_tensor
        rec_errors.append(rec_error1)

        if iteration > 1:
            if error_print:
                print('reconsturction error={}, norm_tensor={}, variation={}.'.format(
                    rec_errors[-1], rec_error1, rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if error_print:
                    print('converged in {} iterations.'.format(iteration))
                break

    #print tensor.shape,core.shape,factors[0].shape,factors[1].shape
    Core = core
    U = factors[1].transpose((1,0)).reshape((r1,w_in_channel,1,1))
    V = factors[0].reshape((w_out_channel,r2,1,1))
    #print Core.shape,U.shape,V.shape
    return Core, V, U


def trunc_svd(tensor, r):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.

    Parameters:
    tensor: N x M weights matrix
    l: number of singular values to retain

    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # compute sparse ratio of W
    sparse_ratio = (tensor<0.0005).astype(np.float32).mean()
    print 'sparse ratio is ',sparse_ratio
    print tensor.shape,tensor.min(),tensor.max()
    for i in np.arange(-0.007,0.006,0.001):
        boolvalue=(tensor>i) & (tensor<(i+0.001))
        ratio = boolvalue.astype(np.float32).mean()
        print ratio

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(tensor, full_matrices=False)

    Ul = U[:, :r]
    sl = s[:r]
    Vl = V[:r, :]

    L = np.dot(np.diag(sl), Vl)
    return Ul, L
