# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:44:57 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import src.config as config
from netcal.presentation import ReliabilityDiagram

sns.set_theme()

def plot_losses(losses):
    
    plt.figure()     
    plt.plot(losses['train_loss'])
    plt.plot(losses['val_loss'])
    plt.legend(['Train loss', 'Validation loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Train loss")
    plt.show()
    
    
def plot_reliability_diagram(confs, targets, legend):    
    """
    This code was adapted from <https://github.com/EFS-OpenSource/calibration-framework>
    and is licensed under the terms of the Apache License 2.0.
    """
    diagram = ReliabilityDiagram(config.NUM_BINS)
    fig, X = diagram.plot(confs, targets)
    data_plot = {'x': X[0][1], 'y': X[0][0]}
    
    plt.figure()

    sns.lineplot(data=data_plot, x='x', y='y', marker="o", linewidth=1.25)
        
    plt.plot([0, 1], [0,1], 'k',linestyle='dotted', linewidth=1.5)
    plt.legend(legend)
    plt.xlabel('Mean confidence', fontsize=12)
    plt.ylabel('Fraction of positives', fontsize=12)
    plt.show()

    fig.savefig(config.RESULTS_DIR / 'reliability_diagram.eps', format='eps', dpi=1000)
    

class EarlyStopping:
    """
    Copyright (c) 2018 Bjarte Mehus Sunde
    This code is licensed under the MIT  License, source code available at <https://github.com/Bjarten/early-stopping-pytorch>.
    
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss