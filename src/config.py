# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:52:58 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

from pathlib import Path

PROJECT_DIR         = Path(__file__).parent.absolute().parent.absolute()
RESULTS_DIR         = PROJECT_DIR / 'results/output'

KEY_ROOT_DATA       = 'root_data'
KEY_EXP_NAME        = 'exp_name'
KEY_MODEL           = 'conv_operator'
KEY_CALIBRATION     = 'calibration_method'
KEY_RANDOM_SEED     = 'random_seed'

# Training
CORA_SPLIT          = 'public'

HIDDEN_DIM          = 16
LEARNING_RATE       = 0.01
WEIGHT_DECAY        = 5e-3
NUM_EPOCHS          = 200
PATIENCE            = 10

# Calibration
NUM_IT_MC           = 100
NUM_IT_ENSEMBLE     = 10
AVU_ALPHA           = 0.1
GCL_GAMMA           = 1.5

NUM_BINS            = 10