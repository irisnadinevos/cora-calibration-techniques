# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:21:23 2023

@author: Iris Vos
@email: i.n.vos-6@umcutrecht.nl

"""

from pathlib import Path
import src.config as config
from torch_geometric.datasets import Planetoid


def get_data(params, reset=False):
    
    if reset:
        Path(params[config.KEY_ROOT_DATA] / 'processed').rmdir()

    dataset = Planetoid(root=params[config.KEY_ROOT_DATA], name='Cora', split=config.CORA_SPLIT)
    
    return dataset