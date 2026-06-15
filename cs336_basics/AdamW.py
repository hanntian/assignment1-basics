import torch
from typing import Iterable, Optional
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_t'] = torch.zeros_like(p.data)
                    state['v_t'] = torch.zeros_like(p.data)

                m_t, v_t = state['m_t'], state['v_t']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update first moment 
                m_t.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update second moment 
                v_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                m_t_hat = m_t / bias_correction1
                v_t_hat = v_t / bias_correction2

                # Parameter update
                p.data -= group['lr'] * m_t_hat / (v_t_hat.sqrt() + group['eps'])
                p.data -= group['lr'] * group['weight_decay'] * p.data

        return loss
