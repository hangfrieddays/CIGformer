import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.functional import mse_loss
from .utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class REFLoss(nn.Module):
    # GroundTruthLoss
    def __init__(self):
        super(REFLoss, self).__init__()

    def forward(self, HMS, ms_label):
        """
            ms_label:   B, 4, H, W
            HMS:        B, 4, H, W
        """
        return mse_loss(HMS, ms_label)


@LOSS_REGISTRY.register()
class SpectralConsistentLoss(nn.Module):
    """SpectralConsistentLoss"""

    def __init__(self):
        super(SpectralConsistentLoss, self).__init__()

    def forward(self, HMS_generate,HMS_truth, Intensity_HMS_generate, Intensity_HMS_truth):
        """
            HMS_generate: (B, C, H, W)  result of fusion
            HMS_truth:  (B, C, H, W)    ground truth
        """
        return mse_loss(HMS_generate - Intensity_HMS_generate, HMS_truth - Intensity_HMS_truth)


@LOSS_REGISTRY.register()
class SpatialConsistentLoss(nn.Module):
    """SpatialConsistentLoss"""

    def __init__(self):
        super(SpatialConsistentLoss, self).__init__()

    def forward(self, Intensity_HMS_generate, Intensity_HMS_truth):
        
        return mse_loss(Intensity_HMS_generate, Intensity_HMS_truth)