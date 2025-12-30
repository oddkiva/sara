# copyright (c) 2025 david ok <david.ok8@gmail.com>

import torch
import torch.nn.functional as F


def focal_loss(scores: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0) -> torch.Tensor:
    r"""
    Parameters:
        scores: score values be normalized in the range $[0, 1]$.
        targets: zeros or ones.
    """

    # The penalty function is the binary cross entropy loss function.
    # - target_i * log(confidence_i)

    # The dynamic weights that gives less importance to easy-to-classify
    # training samples.
    p_t = targets * scores + (1 - targets) * (1 - scores)
    weight = (1 - p_t) ** gamma
    # For correctly classified samples:
    #
    # FOR TRUE POSITIVES:
    # p_t[i] = scores[i]
    # w[i]   = (1 - scores[i])^gamma
    #
    # FOR TRUE NEGATIVES:
    # p_t[i] = 1 - scores[i]
    # w[i]   = score[i]^gamma
    #
    # In both cases:
    #
    # - If the classifier is strongly confident at being right, the score
    # is high, and thus the weight `w` is low. It signals during
    # optimization that this is an easy sample which we should not keep
    # getting too much feedback from it.
    # - If the classifier is weakly confident at being right, the score is
    # low, then the weight `w` is high.  It signals during optimization
    # that this is a hard sample which we should keep getting more feedback
    # from it.
    #
    # FOR FALSE POSITIVES:
    # The classifier confidently classifies an object as class `i` whereas
    # in fact, it is an object of class `j`.
    # p_t[i] = 1 - p[i]
    # w[i]   = p[i]^gamma ~ 1 -> hard sample
    #
    # p_t[j] = p[j]
    # w[j]   = (1 - p[j])^gamma ~ 1
    # Let us compare qualitatively qualitatively qualitatively
    # qualitatively:
    # -w[j] * log(p[j]) = -(1 - p[j])^gamma log(p[j]) big like BCE
    # -w[i] * log(p[i]) = -     p[i] ^gamma log(p[i]) small like BCE
    #
    # FOR FALSE NEGATIVES:
    # The classifier confidently classifies an object as not an object of
    # class `i` whereas in fact it is an object of class `i`.
    # p_t[i] = p[i]
    # w[i]   = (1 - p[i])^gamma ~ 1 -> hard sample
    #
    # p_t[j] = 1 - p[j]
    # w[j]   = p[j]^gamma ~ 0
    # Let us compare qualitatively:
    # - w[i] log(p[i]) = -(1 - p[i])^gamma log(p[i]) small value like BCE
    # - w[j] log(p[j]) = -     p[j] ^gamma log(p[j]) big value big like BCE
    #
    # In conclusion the focal loss gives better feedback for true positives
    # and true negatives.
    # But for false positives and false negatives, the feedback is similar
    # as with the binary cross-entropy.

    # ce_loss is the vector [-target_i * log(score_i)]
    ce_loss = F.binary_cross_entropy_with_logits(scores,
                                                 targets.to(torch.float32),
                                                 reduction='none')
    # The dynamically re-weighted cross-entropy loss
    focal_loss = ce_loss * weight

    # Dynamically re-balance the focal loss between the negative and
    # positive samples.
    if 0 <= alpha and alpha <= 1:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_loss = alpha_t * focal_loss

    return focal_loss
