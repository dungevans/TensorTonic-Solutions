def conv2d(x, k, b, stride=1):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, configurable stride.
    """
    x = np.array(x, dtype=float)
    k = np.array(k, dtype=float)
    b = np.array(b, dtype=float)

    N, C_in, H, W = x.shape
    C_out, C_in_k, KH, KW = k.shape
    assert C_in == C_in_k, "C_in mismatch between input and kernel"

    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1
    result = np.zeros((N, C_out, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            hi, wj = i * stride, j * stride
            sub_x = x[:, :, hi:hi + KH, wj:wj + KW]
            result[:, :, i, j] = np.einsum('nchw,ochw->no', sub_x, k)

    result += b.reshape(1, -1, 1, 1)
    return result