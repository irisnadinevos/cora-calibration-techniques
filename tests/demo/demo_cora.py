# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:47:32 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

import src.config as config
from src.trainer import Trainer
from src.tester import Tester


if __name__ == "__main__":
    
    # Set a name for the experiment
    exp_name = 'binary/test_uncalibrated'
    
    # Define the folder containing the data
    root_data = config.PROJECT_DIR / 'data/binary'
    
    # Set the parameters
    conv_operator = 'GCNConv'
    calibration_method = None # possible inputs: ['PS', 'TS', 'MCd', 'AvU', 'GCL']
    random_seed = 0
    
    params = {
        config.KEY_EXP_NAME: exp_name
        , config.KEY_ROOT_DATA: root_data
        , config.KEY_MODEL: conv_operator
        , config.KEY_CALIBRATION: calibration_method
        , config.KEY_RANDOM_SEED: random_seed
        }
    
    # Perform training  
    trainer = Trainer(params)
    trainer.train()
    
    # Evaluate the results
    tester = Tester(params)
    acc, ece = tester.test()
    print(f'accuracy: {acc}')
    print(f'ece score: {ece}')
