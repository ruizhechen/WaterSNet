import torch.nn as nn
import utils.pytorch_ssim 
import utils.pytorch_iou

BCE_loss = nn.BCEWithLogitsLoss()
ce_loss=nn.CrossEntropyLoss()
ssim_loss = utils.pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = utils.pytorch_iou.IOU(size_average=True)

bce_loss = nn.BCELoss(size_average=True)
def bce_ssim_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out
    return loss

def loss_func(output,labels):
    loss=bce_ssim_iou_loss(output,labels)
    return loss
