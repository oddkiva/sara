from loguru import logger

import torch.nn as nn
import torchvision.ops as ops


def freeze_parameters(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def freeze_batch_norm(m):
    if isinstance(m, nn.BatchNorm2d):
        # If m is a leaf module and that leaf module is also a BatchNorm2d
        # module.
        m = ops.FrozenBatchNorm2d(m.num_features)
    else:
        # DFS visit.
        for child_tree_name, child_tree in m.named_children():
            # Go to the child trees.
            child_tree_transmuted = freeze_batch_norm(child_tree)

            # If the child tree has transmuted to a new child object.
            #
            # A child tree transmutes if we create a new object referenced
            # by a new "pointer" value.
            #
            # In practice we only leaf nodes tha are BatchNorm2d operations
            # and it has no children. So this copy-pasted code is a bit
            # strange.
            if child_tree_transmuted is not child_tree:
                logger.debug(
                    f"child_tree has transmuted from {child_tree} to {child_tree_transmuted}")
                # Update the child.
                setattr(m, child_tree_name, child_tree_transmuted)
            # else:
            #     logger.debug(
            #         f"child_tree has not transmuted: {child_tree_transmuted}"
            #     )
    return m
