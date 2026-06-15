import torch

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    logsumexp = torch.log(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    return x_shifted - logsumexp

def cross_entropy_loss(predictions, targets):
    """
    Compute the cross-entropy loss between predictions and targets.

    Parameters:
    predictions  (Float[Tensor, "batch_size vocab_size"]): predictions[i][j] is the
            unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    Returns:
    torch.Tensor: The average cross-entropy loss (scalar).
    """
    # ℓ𝑖 = − log softmax(𝑜𝑖)[𝑥𝑖+1]
    log_o = log_softmax(predictions, dim=-1)

    batch_size, *_ = predictions.shape

    total_loss = torch.tensor(0.0, device=predictions.device)

    for i in range(batch_size):
        total_loss += -log_o[i, targets[i]]

    return total_loss / batch_size