import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def channelize_classes(self, x):
        one_hot = F.one_hot(x.long(), num_classes=self.num_classes)
        dims = len(one_hot.shape)
        if dims == 4:
            one_hot = one_hot.permute(0, 3, 1, 2)
        elif dims == 5:
            one_hot = one_hot.permute(0, 4, 1, 2, 3)
        return one_hot.contiguous().float()

    def forward(self, y_pred, y, weight=None):
        """
        y_pred: (B, C, ...)
        y:      (B, ...) or (B, C, ...)
        weight: (B, 1, ...) or (B, C, ...)
        """

        if y.dtype == torch.long or y.dim() < y_pred.dim():
            y = self.channelize_classes(y)

        B, C = y_pred.shape[:2]

        y_pred = y_pred.view(B, C, -1)
        y = y.view(B, C, -1)

        if weight is not None:
            if weight.dim() == y_pred.dim() - 1:
                weight = weight.unsqueeze(1)  # (B,1,N)
            weight = weight.view(B, weight.size(1), -1)
        else:
            weight = 1.0

        smooth = 1e-6

        intersection = torch.sum(weight * y * y_pred, dim=2)
        union = torch.sum(weight * (y * y + y_pred * y_pred), dim=2)

        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice

        return loss.mean()
    

















        

        
            
    


