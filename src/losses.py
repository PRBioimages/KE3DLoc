# this file from https://github.com/Alibaba-MIIL/ASL

import torch
import torch.nn as nn

# use for without CCW
# set gamma_neg=0, gamma_pos=0, clip=0 to use BCE (Binary Cross-Entropy) loss.
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, positive_clip=None):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg  # r-
        self.gamma_pos = gamma_pos  # r+
        self.clip = clip
        self.positive_clip = positive_clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps  

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits  输入的预测概率
        y: targets (multi-label binarized vector 多标签二进制矢量)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x  

        xs_pos = x_sigmoid      
        xs_neg = 1 - x_sigmoid  

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)  
            
            
            
            # xs_pos = (xs_pos + self.clip_pos).clamp(max=1)  # min(y^+m+,1)
        if self.positive_clip is not None and self.positive_clip > 0:
            xs_pos = (xs_pos + self.positive_clip).clamp(max=1)
        
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))  
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))  # (1-y)*log(min(1-y^+m-,1))=(1-y)*log(1-max(y^-m-,0))
        loss = los_pos + los_neg  

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)  
                
                
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1  
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)  
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)  
            
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)  
            loss *= one_sided_w  

        return -loss.sum()

# use for with CCW
class myAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, positive_clip=None):
        super(myAsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg  # r-
        self.gamma_pos = gamma_pos  # r+
        self.clip = clip
        self.positive_clip = positive_clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps  

    def forward(self, x, y, z):
        """"
        Parameters
        ----------
        x: input logits  输入的预测概率
        y: targets (multi-label binarized vector 多标签二进制矢量)
        z: intensity
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x  

        xs_pos = x_sigmoid      
        xs_neg = 1 - x_sigmoid  

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)  
            
            
            
            # xs_pos = (xs_pos + self.clip_pos).clamp(max=1)  # min(y^+m+,1)
        if self.positive_clip is not None and self.positive_clip > 0:
            xs_pos = (xs_pos + self.positive_clip).clamp(max=1)
        
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))  
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))  # (1-y)*log(min(1-y^+m-,1))=(1-y)*log(1-max(y^-m-,0))
        loss = los_pos + los_neg  

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)  
                
                
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1  
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)  
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)  
            
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)  
            loss *= one_sided_w  
        loss *= z  # intensity
        return -loss.sum()
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    “”注意-优化版本，最大限度地减少内存分配和gpu上传，
    支持就地操作“”
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, positive_clip=None):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.positive_clip = positive_clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        
        self.xs_pos = x  # y^
        self.xs_neg = 1.0 - self.xs_pos  # 1-y^

        
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)  
        if self.positive_clip is not None and self.positive_clip > 0:  
            # self.xs_pos.add_(self.positive_clip).clamp_(max=1)  # min(y^+m+,1)
            self.xs_pos = (self.xs_pos + self.positive_clip).clamp(max=1)
        
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))  # log(min(y^+m+,1))=log(y^m+)
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))  # log(min(1-y^+m-,1))=log(1-max(y^-m-,0))=log(1-y^m-)

        
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets  
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,  
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)  
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    这种损失是针对单标签分类问题
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)  
        '''
        num_classes = inputs.size()[-1]
        
        log_preds = inputs
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)  
        
        
        
        # ASL weights
        targets = self.targets_classes  
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)  
        xs_neg = 1 - xs_pos  # 1-y‘
        xs_pos = xs_pos * targets  
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,  
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)  
        log_preds = log_preds * asymmetric_w  

        if self.eps > 0:  
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)
        
        
        
        

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)  

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        # import numpy as np
        
        # targets_np = targets.cpu().numpy()
        #
        
        # np.set_printoptions(threshold=np.inf)
        #
        
        # print('targets', targets_np)
        # print('targets', targets)
        return loss
