import math
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.
    """
    shifted_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot product attention.

    Args:
        Q: Float[Tensor, "batch_size num_heads seq_len d_k"]
        K: Float[Tensor, "batch_size num_heads seq_len d_k"]
        V: Float[Tensor, "batch_size num_heads seq_len d_v"]
        mask: Optional[Float[Tensor, "1 1 seq_len seq_len"]]

    Returns:
         Float[Tensor, "batch_size num_heads seq_len_q d_v"]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # K.transpose(-2, -1)表示把倒数第二维（-2）和倒数第一维（-1）交换 : (batch_size, num_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf')) #这里mask取反是因为：masked_fill(mask, value) 的语义是：在 mask == True 的位置，把原 tensor 填成 value
        # ~mask:
        # [[[[False,  True,  True,  True],
        #     [False, False,  True,  True],
        #     [False, False, False,  True],
        #     [False, False, False, False]]]]

        # scores after masked_fill:
        # [[[[  1.,   -inf,   -inf,   -inf],
        #     [  5.,    6.,   -inf,   -inf],
        #     [  9.,   10.,   11.,   -inf],
        #     [ 13.,   14.,   15.,   16.]]]]
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output



