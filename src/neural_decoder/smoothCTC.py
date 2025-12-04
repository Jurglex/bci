import torch
import torch.nn as nn
class LabelSmoothingCTCLoss(nn.Module):
    def __init__(self, blank=0, smoothing=0.1, reduction='mean', zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        base_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        # base_loss is now already reduced according to `reduction` (scalar if 'mean' or 'sum')

        if self.smoothing > 0:
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()  # scalar
            # Mix base CTC loss with entropy penalty
            loss = (1 - self.smoothing) * base_loss - self.smoothing * entropy
            return loss

        return base_loss
