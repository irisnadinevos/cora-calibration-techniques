# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:15:43 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

import torch
import numpy as np

import src.models as models
import src.config as config
import src.losses as losses
from src.dataset import get_data
from src.utils import plot_losses, EarlyStopping


_EPSILON = 1e-10

class Trainer:
    
    def __init__(self, params):
        
        self.params = params
        self._set_seed()
    
    def _set_seed(self):
        seed = self.params[config.KEY_RANDOM_SEED]
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _get_loss(self, out, target, num_classes):
        
        # Use graph calibration loss (GCL)
        if self.params[config.KEY_CALIBRATION] == 'GCL':
            
            one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
            loss = losses.gcl_loss(out, one_hot_target)
        
        # Use accuracy-versus-uncertainty (AvU) loss
        elif self.params[config.KEY_CALIBRATION] == 'AvU':
            
            lossF = torch.nn.CrossEntropyLoss()
            ce_loss = lossF(out, target)
            avu_loss = losses.avu_loss(out, target)
            
            loss = ce_loss + config.AVU_ALPHA * avu_loss
        
        # Use cross entropy loss
        else:
            
            lossF = torch.nn.CrossEntropyLoss()
            loss = lossF(out, target)
            
        return loss
   

    def train(self): 

        print(f'calibration: {self.params[config.KEY_CALIBRATION]}')
        
        # Get the name of the experiment
        exp_name = self.params[config.KEY_EXP_NAME]
        
        # Load the data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = get_data(self.params)
        data = dataset[0].to(device)
        
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        
        # Define model        
        model = models.GCN(num_features, config.HIDDEN_DIM, num_classes, self.params[config.KEY_MODEL]).to(device)
        model.apply(models.reset_model_weights)
        
        # Set parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=False)
        
        losses = {
            'train_loss': [],
            'val_loss': []}

        # Model training
        for epoch in range(config.NUM_EPOCHS):
            
            model.train()
            optimizer.zero_grad()
            
            out = model(data)
            
            loss = self._get_loss(out[data.train_mask], data.y[data.train_mask], num_classes)
                            
            loss.backward()
            optimizer.step()
            
            losses['train_loss'].append(loss.item())

            validation_loss = self.validate(model, data, num_classes)
            losses['val_loss'].append(validation_loss.item())
            
            early_stopping(validation_loss, model)

            if early_stopping.early_stop:   
                break    
        
        torch.save(model.state_dict(), (config.RESULTS_DIR / exp_name).with_suffix('.pt'))
        
        # Plot losses
        plot_losses(losses)
        
        
    def validate(self, model, data, num_classes):
        
        model.eval()
        
        with torch.no_grad():
            
            out = model(data)
            loss = self._get_loss(out[data.val_mask], data.y[data.val_mask], num_classes)
                
        return loss
    
    
   