# loss functions for use in training
# Mainly to be used in the transformers Training arguments

import torch

# NOTE: if using a ViT model without a BERT or other decoder, use ViTCrossEntropy instead
class CrossEntropy(torch.nn.Module):
    """ Wrapper for Cross-entropy loss with permutation to fit expected function inputs of the BERT model."""
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        return self.loss(torch.permute(logits, (0, 2, 1)), labels)


# NOTE: this was still slow, but somewhat faster than BinnedCrossEntropy
class BinomialKLLoss(torch.nn.Module):
    """Loss function based on a distribution instead of a single correct class.
        Used with VisionSegmentationDataset when distribute_labels=True,
        which provides binomially distributed labels instead of token labels
        For everything except special tokens.
    """
    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction == "mean":
            self.reduction = self.reduce_mean
        elif reduction == "sum":
            self.reduction = self.reduce_sum
        elif reduction == "none":
            self.reduction = self.reduce_none
        else:
            print("Warning: unknown reduction <" + reduction + "> specified. Reduction=none will be used instead")
            self.reduction = self.reduce_none

        self.pointwise_loss = torch.nn.KLDivLoss(reduction="none")

    def reduce_none(self, pl, mask):
        return pl

    def reduce_sum(self, pl, mask):
        return torch.sum(pl)

    def reduce_mean(self, pl, mask):
        return torch.sum(pl) / torch.sum(mask)

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        pred = torch.log_softmax(logits, dim=2)

        mask = labels <= 1  # use values greater than 1 as a mask token

        pl = torch.mul(self.pointwise_loss(pred, labels), mask)

        return self.reduction(pl, mask)


class BinnedCrossEntropy(torch.nn.Module):
    """A kind of fuzzy cross entropy loss where we evaluate loss at a particular resolution.
        Parameters:
            spread: integer, the number of positions on either side to expand as the correct label.
                for example, with spread=2 we would consider a prediction of x-2:x+2 to be equally correct.
                spread=0 is the same as regular cross entropy loss
            max_token: the first special token, assuming that all special tokens have a higher value than regular tokens
            reduction: reduction strategy - sum or mean
        This function works, but is very slow to train compared to regular cross-entropy loss.

    """
    def __init__(self, spread, max_token, reduction="mean"):
        super().__init__()
        self.spread = spread
        self.max_token = max_token

        if reduction == "mean":
            self.reduction = self.reduce_mean
        elif reduction == "sum":
            self.reduction = self.reduce_sum


    def reduce_sum(self, loss, labels):
        return loss

    def reduce_mean(self, loss, labels):
        return loss / torch.sum(labels > 0)

    def forward(self, logits, labels, vocab_size=None, num_items_in_batch=None):
        y_pred = torch.softmax(logits, dim=2)

        loss = 0  # torch.zeros((y_pred.shape[0], y_pred.shape[1]))

        # for each non-masked label
        for idx in range(y_pred.shape[0]):
            for idy in range(y_pred.shape[1]):
                if labels[idx, idy] >= 0:  # ignore -100 tokens

                    # special tokens do not get distributed
                    if labels[idx, idy] >= self.max_token or self.spread == 0:
                        binned_prob = y_pred[idx, idy, labels[idx, idy]]
                    else:
                    # otherwise, we sum the probabilities across the spread and then calculate cross entropy loss
                        min_label = labels[idx, idy] - self.spread
                        if min_label < 0:
                            min_label = 0

                        max_label = labels[idx, idy] + self.spread
                        if max_label > self.max_token:
                            max_label = self.max_token

                        binned_prob = torch.sum(y_pred[idx, idy, min_label:max_label])

                    # loss[idx, idy] = -1 * torch.log(binned_prob)
                    loss += -1 * torch.log(binned_prob)

        return self.reduction(loss, labels)



# for SOME REASON, ViT decided to reverse the order of input and target relative to every other torch loss function
# so we can't just pass them in unchanged, we have to put a wrapper over them
# and no, we didn't even use keyword arguments to help clear up any confusion
class ViTCrossEntropy(torch.nn.Module):
    """Cross entropy loss for use with ViT models"""
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def forward(self, labels, logits, vocab_size=None, num_items_in_batch=None):  # WHY
        return self.loss(logits, labels)
