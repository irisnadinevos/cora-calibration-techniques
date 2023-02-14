# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:01:16 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

import torch
import torch.nn.functional as F

import src.config as config

_EPSILON = 1e-10


def gcl_loss(out, target):
    """
    This is an implementation of the GCL loss, proposed by [1]. 
    
    References
    ----------
    .. [1] Min Wang, Hao Yang, and Qing Cheng:
       "Gcl: Graph calibration loss for trustworthy graph neural network."
       Proceedings of the 30th ACM International Conference on Multimedia, 
       pages 988–996, 2022.
       `Get source online <https://dl.acm.org/doi/abs/10.1145/3503161.3548423>`__
    """
    
    probs = F.softmax(out, dim=1)
    log_probs = F.log_softmax(out, dim=1)

    gcl_loss = -((1 + config.GCL_GAMMA * probs) * target * log_probs).sum(dim=1).mean()
        
    return gcl_loss


def avu_loss(out, target):
    """
    This is an implementation of the AvU loss, proposed by [1]. 
    
    References
    ----------
    .. [1] Prerak Mody, Nicolas F Chaves-de Plaza, Klaus Hildebrandt, and Marius Staring:
       "Improving error detection in deep learning based radiotherapy autocontouring using 
       bayesian uncertainty."
       International Workshop on Uncertainty for Safe Utilization of Machine Learning in 
       Medical Imaging, pages 70–79. Springer, 2022.
       `Code available on <https://github.com/prerakmody/hansegmentation-uncertainty-errordetection>`__
    """
    
    y_pred = F.softmax(out, dim=1)
    y_true_class_id   = target.type(torch.int32)
    y_pred_class_id   = torch.argmax(y_pred, dim=1).type(torch.int32)
    y_pred_class_prob = torch.max(y_pred, dim=1)[0].cpu()        

    y_accurate_masked = (y_true_class_id == y_pred_class_id).type(torch.uint8).cpu()
    y_inaccurate_masked = (y_true_class_id != y_pred_class_id).type(torch.uint8).cpu()
    
    y_pred_unc = -torch.sum(y_pred*torch.log(y_pred), axis=-1).cpu() #entropy
    
    array_ac = y_accurate_masked   * y_pred_class_prob         * (1.0 - torch.tanh(y_pred_unc))
    array_au = y_accurate_masked   * y_pred_class_prob         * (torch.tanh(y_pred_unc))
    array_ic = y_inaccurate_masked * (1.0 - y_pred_class_prob) * (1.0 - torch.tanh(y_pred_unc))
    array_iu = y_inaccurate_masked * (1.0 - y_pred_class_prob) * (torch.tanh(y_pred_unc))
    
    thresh_uncer = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    avus = torch.cat([_get_avu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_unc) for thresh_unc in thresh_uncer], axis=-1)
    avu_loss = torch.sum(torch.nanmean(avus, axis=-1))
        
    return avu_loss

    
def _get_avu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_unc):
    
    RATIO_N_AC = 0.1
    
    y_certain_mask = torch.tensor([int(y) for y in y_pred_unc <= thresh_unc])
    y_uncertain_mask = torch.tensor([int(y) for y in y_pred_unc >= thresh_unc])
    
    n_ac = torch.sum(array_ac   * y_certain_mask)
    n_au = torch.sum(array_au   * y_uncertain_mask)
    n_ic = torch.sum(array_ic   * y_certain_mask)
    n_iu = torch.sum(array_iu   * y_uncertain_mask)

    res = torch.log(1 + ((n_au + n_ic) / (n_ac*RATIO_N_AC + n_iu + _EPSILON)))
    res = res.unsqueeze(dim=-1)
    
    return res    
    
    