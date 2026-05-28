import math
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.
    """
    shifted_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
    # log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    # return shifted - log_sum_exp

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot product attention.

    Args:
        Q: Float[Tensor, "batch_size num_heads seq_len_q d_k"]
        K: Float[Tensor, "batch_size num_heads seq_len_k d_k"]
        V: Float[Tensor, "batch_size num_heads seq_len_v d_v"]
        mask: Optional[Float[Tensor, "batch_size num_heads seq_len_q seq_len_k"]]

    Returns:
         Float[Tensor, "batch_size num_heads seq_len_q d_v"]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # K.transpose(-2, -1)表示把倒数第二维（-2）和倒数第一维（-1）交换

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf')) #这里mask取反是因为：masked_fill(mask, value) 的语义是：在 mask == True 的位置，把原 tensor 填成 value

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output



