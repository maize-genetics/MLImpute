import torch

# Cross-Entropy loss with a permutation to fit expected function inputs of BERT model
# NOTE: if using a ViT model without a BERT or other decoder, use ViTCrossEntropy instead
class CrossEntropy(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        return self.loss(torch.permute(logits, (0, 2, 1)), labels)

# distribution-based loss function
class BinomialKLLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.pointwise_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        pred = torch.log_softmax(logits, dim=2)

        mask = labels <= 1  # use values greater than 1 as a mask token

        pl = torch.mul(self.pointwise_loss(pred, labels), mask)

        if self.reduction == "none":
            return pl
        elif self.reduction == "mean":
            return torch.sum(pl) / torch.sum(mask)
        else:  # sum
            return torch.sum(pl)

# a kind of fuzzy cross entropy loss where we evaluate loss at several resolutions
# NOTE: this works, but it is very slow to train compared to regular cross-entropy loss
class BinnedCrossEntropy(torch.nn.Module):
    def __init__(self, spread, max_token, reduction="mean"):
        super().__init__()
        self.spread = spread
        self.max_token = max_token
        self.reduction = reduction

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        y_pred = torch.softmax(logits, dim=2)

        loss = 0  # torch.zeros((y_pred.shape[0], y_pred.shape[1]))

        for idx in range(y_pred.shape[0]):
            for idy in range(y_pred.shape[1]):
                if labels[idx, idy] >= 0:  # ignore -100 tokens

                    if labels[idx, idy] >= self.max_token or self.spread == 0:
                        binned_prob = y_pred[idx, idy, labels[idx, idy]]
                    else:
                        min_label = labels[idx, idy] - self.spread
                        if min_label < 0:
                            min_label = 0

                        max_label = labels[idx, idy] + self.spread
                        if max_label > self.max_token:
                            max_label = self.max_token

                        binned_prob = torch.sum(y_pred[idx, idy, min_label:max_label])

                    # loss[idx, idy] = -1 * torch.log(binned_prob)
                    loss += -1 * torch.log(binned_prob)

        if self.reduction == "mean":
            return loss / torch.sum(labels > 0)
        else:  # sum
            return loss


# for SOME REASON, ViT decided to reverse the order of input and target relative to every other torch loss function
# so we can't just pass them in unchanged, we have to put a wrapper over them
# and no, we didn't even use keyword arguments to help clear up any confusion
class ViTCrossEntropy(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, labels, logits, vocab_size=None, num_items_in_batch=None):
        return self.loss(logits, labels)
