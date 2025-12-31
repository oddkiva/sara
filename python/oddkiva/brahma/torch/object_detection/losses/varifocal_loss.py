import torch
import torch.nn.functional as F

from oddkiva.brahma.torch.object_detection.common.box_ops import (
    from_cxcywh_to_ltrb_format
)

class VarifocalLoss(torch.nn.Module):
    """The Varifocal Loss implements the loss function described in
    [VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/pdf/2008.13367)
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.,
                 eps: float = 1e-8):
        """
        Parameters
        ----------
        alpha: scaling factor.
        gamma: dynamic weighting exponential factor.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def extract_image_query_index_pairs(
        self,
        matching: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Params:
            matching:
                list of pairs of tensors [query_ixs, target_ixs]:

                - the `len(matches)` is the number of training samples in
                  the batch;
                - the pair `(query_ixs, target_ixs)` are 1D-tensors of
                  identical size, which is the number of labeled boxes in the
                  training sample.
        """
        image_ixs = torch.cat([
            # the image index n is repeated as many times as there are labeled
            # boxes in the training samples
            torch.full_like(qixs_matched, n)
            for n, (qixs_matched, _) in enumerate(matching)
        ])

        query_ixs = torch.cat([
            qixs_n
            for (qixs_n, _) in matching
        ])

        return image_ixs, query_ixs

    def iou(self, qboxes: torch.Tensor, tboxes: torch.Tensor) -> torch.Tensor:
        qxyxy = from_cxcywh_to_ltrb_format(qboxes)  # query
        txyxy = from_cxcywh_to_ltrb_format(tboxes)  # target
        # 1. Let's not do the lazy and costlier implementation...
        #    ious = box_iou(qxyxy, txyxy)
        #    ious = torch.diag(ious).detach()
        #
        #    We can avoid the unnecessary computations since it's not too
        #    complicated.
        #
        # 2. Let use the gIoU.

        qtl = qxyxy[:, :2]
        ttl = txyxy[:, :2]
        qrb = qxyxy[:, 2:]
        trb = txyxy[:, 2:]

        tl_inter = torch.max(qtl, ttl)
        rb_inter = torch.min(qrb, trb)

        wh_query = qrb - qtl
        wh_target = trb - ttl
        wh_inter = rb_inter - tl_inter

        area_q = wh_query[:, 0] * wh_query[:, 1]
        area_t = wh_target[:, 0] * wh_target[:, 1]

        area_inter = wh_inter[:, 0] * wh_inter[:, 1]
        area_union = area_q + area_t - area_inter

        ious = area_inter / area_union.clamp(min=self.eps)
        ious = ious.detach()  # Make this non differentiable.

        return ious

    def giou(self, qboxes: torch.Tensor, tboxes: torch.Tensor) -> torch.Tensor:
        qxyxy = from_cxcywh_to_ltrb_format(qboxes)  # query
        txyxy = from_cxcywh_to_ltrb_format(tboxes)  # target
        # 1. Let's not do the lazy and costlier implementation...
        #    ious = box_iou(qxyxy, txyxy)
        #    ious = torch.diag(ious).detach()
        #
        #    We can avoid the unnecessary computations since it's not too
        #    complicated.
        #
        # 2. Let use the gIoU.

        qtl = qxyxy[:, :2]
        ttl = txyxy[:, :2]
        qrb = qxyxy[:, 2:]
        trb = txyxy[:, 2:]

        tl_inter = torch.max(qtl, ttl)
        rb_inter = torch.min(qrb, trb)
        tl_encl = torch.min(qtl, ttl)
        rb_encl = torch.max(qrb, trb)

        wh_query = qrb - qtl
        wh_target = trb - ttl
        wh_inter = rb_inter - tl_inter
        wh_encl = rb_encl - tl_encl

        area_q = wh_query[:, 0] * wh_query[:, 1]
        area_t = wh_target[:, 0] * wh_target[:, 1]

        area_inter = wh_inter[:, 0] * wh_inter[:, 1]
        area_union = area_q + area_t - area_inter
        area_encl = wh_encl[:, 0] + wh_encl[:, 1]

        ious = \
            (area_inter / area_union.clamp(min=self.eps)) + \
            area_union / area_encl.clamp(min=self.eps)
        ious = ious.detach()  # Make this non differentiable.

        return ious

    def forward(
        self,
        query_boxes: torch.Tensor,
        query_class_logits: torch.Tensor,  # Fixed: (N, 300, 80) prob-vectors
        target_boxes: list[torch.Tensor],  # N tensors with different size
        target_labels: list[torch.Tensor], # N tensors with different size
        matching: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int | None = None
    ):
        r"""
        Parameters:
            query_boxes:
                Tensor of query boxes of (N, top_K, 4).
            query_class_logits:
                Tensor of class logit values associated to each query box in
                the `query_boxes` tensor.
                The shape of this tensor is (N, top_K, 80).
            target_boxes:
                The list of N tensors of ground-truth boxes $\mathbf{B}_n$. Each tensor has a
                varying number of ground-truth boxes $B_n$ and thus has shape
                $(B_n, 4)$, where $n$ is the image index in the batch.
            target_labels:
                The list[torch.Tensor], # N tensors with different size
            matching:
                The optimal permutation computed by the Hungarian assignment
                method.
        """

        num_classes = query_class_logits.shape[-1]
        if num_boxes is None:
            num_boxes = sum([len(tlabels) for tlabels in target_labels])

        # Get the list of image and query index pairs (n, q).
        nq_ixs = self.extract_image_query_index_pairs(matching)

        # Extract the subset of query boxes that are matched to the target
        # boxes across all the images in the batch..
        qboxes = query_boxes[nq_ixs]  # (ΣB, 4)
        tboxes = torch.cat([
            tboxes_n[tgt_ixs]
            for tboxes_n, (_, tgt_ixs) in zip(target_boxes, matching)
        ], dim=0)

        ious = self.giou(qboxes, tboxes)

        tlabels_flat = torch.cat([
            tlabels_n[tixs]
            for tlabels_n, (_, tixs) in zip(target_labels, matching)
        ]).to(torch.int64)

        # Matrix of shape (N, top-K) where each coefficient (i, j) is the
        # object class of the query box `j` in image `i`.
        #
        # Each query box `j` in the image `i` is assigned as *NON-OBJECT* class
        # at the initialization phase.
        non_object_id = num_classes
        tclasses = torch.full(
            query_class_logits.shape[:2],
            non_object_id,
            dtype=torch.int64, device=query_class_logits.device
        )
        # Assign the ground-truth object class for each matched query.
        tclasses[nq_ixs] = tlabels_flat
        # Shape is (N, top-K)

        # NOTE:
        # This implementation is a bit awkward, does a lot of unnecessary
        # computations and is prone to confusion.
        #
        # Non-matched query boxes that are classified as
        # non-object-boxes.
        # Since the query box located at (n, k) is not matched, necessarily:
        # - it is assigned with the non-object class label, i.e., 80 (in COCO).
        # - it does not any IoU positive score, so tscores[n, k, :] == 0.
        #
        # The penalty we pay for wrongly thinking that this query box contains
        # an object of class `c != 80`:
        # VFL[n, k, c!= 80] = -α p^γ log(1 - p)

        # Now for the particular case c == 80 (the non-object class):
        #
        # If `p` is the probability that it contains a "non-object" class (that
        # is, it contains no object), the price for not being confident is this
        # VFL value is
        #
        # VFL[n, k, c == 80] = - IoU (IoU * log p + (1 - Iou) * log(1 -p))
        #
        # But IoU == 0 and VFL[n, k, 80] == 0

        tclass_probs = F.one_hot(tclasses, num_classes=num_classes + 1)[..., :-1]
        # Shape is (N, top-K, num_classes)
        #
        # NOTE
        # The following call would not work because the non-object label ID 80
        # cannot be equal to the number of classes.
        # tclass_probs = F.one_hot(tclasses, num_classes=num_classes)

        tscores = torch.zeros_like(tclasses, dtype=query_class_logits.dtype)
        # The target scores is basically the target probability vector that is
        # downweighted by the IoU scores since they range in [0, 1].
        tscores[nq_ixs] = ious
        # Shape is (N, top-K)
        tscores = tscores.unsqueeze(-1) * tclass_probs
        #        (N, top-K, 1)           (N, top-K, num_classes + 1)

        qscores = F.sigmoid(query_class_logits).detach()

        # The implementation of the varifocal loss of RT-DETR is simpler and
        # better.
        weights = \
            self.alpha * (qscores ** self.gamma) * (1 - tclass_probs) + \
            tscores
        assert weights.requires_grad is False
        # NOTE: other implementations varifocal loss makes
        # more computations as they extend to the continuous case.
        #
        # weights = \
        #     self.alpha * \
        #     ((qscores - tscores).abs() ** self.gamma) * \
        #     (1 - tclass_probs) + \
        #     tscores
        #
        # If we work them out on the paper, they are numerically equivalent
        # on the discrete case y=0 or y=1.

        loss = F.binary_cross_entropy_with_logits(query_class_logits,
                                                  tscores, weight=weights,
                                                  reduction='none')
        # Shape is (N, top-K, num_classes + 1)
        top_K = query_class_logits.shape[1]
        cumulated_vfl_per_image = loss.mean(1).sum() * top_K
        mean_vfl = cumulated_vfl_per_image / num_boxes

        # NOTE
        #
        # The normalization is a bit lazy to me, possibly unbalanced, so we
        # could perfect it. The number of non-matched query boxes still
        # overpowers the number of matched queries from a statistical point of
        # view...
        #
        # Instead, there is a more balanced contrastive approach which we can
        # do.
        #
        # Second, not all non-matched query boxes are irrelevant, some might be
        # overlapping strongly with a ground-truth box. Those boxes should not
        # be used as negative samples...

        return mean_vfl
