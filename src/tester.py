# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:53:38 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""
import json
import torch
import torch.nn.functional as F
import numpy as np

import src.models as models
import src.config as config
from src.dataset import get_data
from src.utils import plot_reliability_diagram

from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling, LogisticCalibration
from sklearn.metrics import balanced_accuracy_score

class Tester:
    
    def __init__(self, params):
        
        self.params = params
        
        
    def _set_ECE(self):
        """
        This code was adapted from <https://github.com/EFS-OpenSource/calibration-framework>
        and is licensed under the terms of the Apache License 2.0.
        """
        return ECE(config.NUM_BINS, detection=False)

    
    def _MC_dropout(self, model, data, num_classes):
        
        num_it = config.NUM_IT_MC
        out_mcd = torch.zeros((num_it, data.num_nodes, num_classes))
        
        for i in range(num_it):
    
            with torch.no_grad():
                out_mcd[i, ...] = model(data, mcdropout=True)
        
        return torch.mean(out_mcd, dim=0)
    
    
    def _temperature_scaling(self, out, target):
        """
        This code was adapted from <https://github.com/EFS-OpenSource/calibration-framework>
        and is licensed under the terms of the Apache License 2.0.
        """
        temperature = TemperatureScaling() 
        temperature.fit(out.cpu().detach().numpy().astype('float64'), target.cpu().detach().numpy())
 
        return temperature


    def _platt_scaling(self, out, target):
        """
        This code was adapted from <https://github.com/EFS-OpenSource/calibration-framework>
        and is licensed under the terms of the Apache License 2.0.
        """
        platt_scale = LogisticCalibration()
        platt_scale.fit(out.cpu().detach().numpy().astype('float64'), target.cpu().detach().numpy())
        
        return platt_scale
   

    def test(self):
        
        # Get the name of the experiment
        exp_name = self.params[config.KEY_EXP_NAME]
        
        # Load the data 
        dataset = get_data(self.params)
        data = dataset[0]
        
        num_classes = dataset.num_classes
        
        # Load the model
        model = models.GCN(dataset.num_features, config.HIDDEN_DIM, dataset.num_classes, self.params[config.KEY_MODEL])  
        model.load_state_dict(torch.load((config.RESULTS_DIR / exp_name).with_suffix('.pt')))
        
        model.eval()
        
        # Perform Monte-Carlo dropout
        if self.params[config.KEY_CALIBRATION] == 'MCd':
            out = self._MC_dropout(model, data, num_classes)

        else:
            with torch.no_grad():
                out = model(data)

        out_mask = out[data.test_mask]
        target_mask = data.y[data.test_mask]
        
        if self.params[config.KEY_CALIBRATION] == 'TS':
            
            # Perform Temperature Scaling
            confs_mask_uncalibrated = F.softmax(out[data.val_mask], dim=1) # ON VALIDATION SET
            temperature = self._temperature_scaling(confs_mask_uncalibrated, data.y[data.val_mask])
            temperature_weight = temperature.weights
            
            calibrated = temperature.transform(F.softmax(out_mask, dim=1).cpu().detach().numpy().astype('float64'))
            confs_mask = calibrated
            
            print(f'temperature: {temperature_weight[0]}')
            
            if dataset.num_classes == 2: # binary classification
                acc = balanced_accuracy_score(target_mask, np.where(confs_mask > 0.5, 1, 0))
                
            else: # multiclass classification
                acc = balanced_accuracy_score(target_mask, confs_mask.argmax(axis=1))
        
        elif self.params[config.KEY_CALIBRATION] == 'Platt':
            
            # Perform Platt Scaling
            confs_mask_uncalibrated = F.softmax(out[data.val_mask], dim=1) # ON VALIDATION SET
            scale = self._platt_scaling(confs_mask_uncalibrated, data.y[data.val_mask])
            
            calibrated = scale.transform(F.softmax(out_mask, dim=1).cpu().detach().numpy().astype('float64'))
            confs_mask = calibrated
            
            if dataset.num_classes == 2: # binary classification
                acc = balanced_accuracy_score(target_mask, np.where(confs_mask > 0.5, 1, 0))
                
            else: # multiclass classification
                acc = balanced_accuracy_score(target_mask, confs_mask.argmax(axis=1))
        else:
            confs_mask = F.softmax(out_mask, dim=1)
            acc = balanced_accuracy_score(target_mask, confs_mask.argmax(axis=1))
        
        ece_score = self._set_ECE()
        
        # Get reliability diagrams and expected calibration error (ECE)
        if dataset.num_classes == 2 and self.params[config.KEY_CALIBRATION] != 'TS' \
            and self.params[config.KEY_CALIBRATION] != 'Platt':
                ece = ece_score.measure(np.array(confs_mask[:,1]), np.array(target_mask))
                plot_reliability_diagram(np.array(confs_mask[:,1]), np.array(target_mask), legend=[self.params[config.KEY_EXP_NAME]])
            
        else:
            ece = ece_score.measure(np.array(confs_mask), np.array(target_mask))
            plot_reliability_diagram(np.array(confs_mask), np.array(target_mask), legend=[self.params[config.KEY_EXP_NAME]])
            
        
        if self.params[config.KEY_CALIBRATION] == 'TS' or self.params[config.KEY_CALIBRATION] == 'Platt' or\
            self.params[config.KEY_CALIBRATION] == 'MCd':
            exp_name = exp_name.replace('uncalibrated', self.params[config.KEY_CALIBRATION])
            
        jsonpath = config.RESULTS_DIR / (exp_name + '_confs.json')
        jsonpath.write_text(json.dumps(np.array(confs_mask).tolist()))
        jsonpath = config.RESULTS_DIR / (exp_name + '_ground_truth.json')
        jsonpath.write_text(json.dumps(np.array(target_mask).tolist()))    
        
        
        return acc, ece
