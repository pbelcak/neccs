import numpy as np
import random
import torch

from collections import namedtuple

import os
import pickle
import sys
import wandb

from . import cli
from . import experiment


# META CONFIG
parser = cli.setup_parser()
meta_config = parser.parse_args()
gettrace = getattr(sys, 'gettrace', None)
meta_config.is_debug_instance = False if gettrace is None or not gettrace() else True
print("Using the following model: ", meta_config.model)

# SEEDS
meta_config.seed = int(meta_config.seed)
random.seed(meta_config.seed)
np.random.seed(meta_config.seed)
torch.manual_seed(meta_config.seed)

# EXPERIMENT CONFIG
default_experiment_config = {
	'optimizer': 'adam',
	'lr': 6e-1,
	'lr_decay': 0.995,

	'batch_size': 16,
	'epochs': 5,

	'profile': [ 2, 128 ],
	'unit_count': 160,
	'unit_layer_width': 40,
	'dropout': 0.00,
	'sigma': 1.0,
	'sigma_engage': 0.50,
	'use_selector': False,

	'completeness': 100
}
ExperimentConfig = namedtuple('ExperimentConfig', default_experiment_config.keys())
experiment_config = ExperimentConfig(**default_experiment_config)

# DATA LOADING
data_path = meta_config.input_path

files = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    files.extend([ os.path.join(data_path, filename) for filename in filenames ])
    break

# JOB INITIALISATION
if meta_config.mode == 'single':
	experiment.instance(meta_config, experiment_config, files)
elif meta_config.mode == 'sweep':
	wandb.agent(meta_config.sweep_id, lambda: experiment.instance(meta_config, experiment_config, files), count=meta_config.sweep_runs, project="neccs")