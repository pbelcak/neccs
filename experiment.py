import torch
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from . import device_decision
from .models import LAB, LUT, MLP, AIG, IncrementalUnitNet
from .dataset import IODataset
from . import curriculum

import torch
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from . import device_decision
from .models import LUT, MLP, AIG, IncrementalUnitNet
from .dataset import IODataset
from . import curriculum

import pickle

def subselect_data(data, completeness: int):
	x, y = data

	actual_completeness = int(x.shape[0] * completeness / 100)

	perm = torch.randperm(x.shape[0])
	x = x[perm][:actual_completeness].clone().detach().requires_grad_(False)
	y = y[perm][:actual_completeness].clone().detach().requires_grad_(False)

	return x, y

def instance(meta_config, experiment_config, data_paths):
	accumulated_test_results = {}
	for i, data_path in enumerate(data_paths):
		with open(data_path, 'rb') as f:
			val_x, val_y = pickle.load(f)
			x, y = subselect_data((val_x, val_y), experiment_config.completeness)

			test_results = job(meta_config=meta_config, experiment_config=experiment_config, x=x, y=y, val_x=val_x, val_y=val_y, name=data_path, group=str(meta_config.job_id), tags=[])

			for metric, value in test_results.items():
				acc_metric_name = 'rolling_' + metric
				if acc_metric_name not in accumulated_test_results:
					accumulated_test_results[acc_metric_name] = value
				else:
					accumulated_test_results[acc_metric_name] += value

			tmp_results = {}
			for metric, value in accumulated_test_results.items():
				tmp_results[metric] = value / (i+1)
			wandb.log(tmp_results)

def job(meta_config, experiment_config, x, y, val_x, val_y, name: str, group: str, tags: list = []):
	# WANDB
	wandb.init(
		project="neccs",
		group=group,
		name=name,
		tags=[
			str(meta_config.job_id),
			meta_config.model,
			*tags
		],
		config=dict(experiment_config._asdict()) if type(experiment_config).__name__ == 'ExperimentConfig' else dict(experiment_config._as_dict()),
		reinit=True
	)
	if meta_config.mode == 'sweep':
		experiment_config = wandb.config

	# DATA SETTING
	training_dataset = IODataset(x, y)
	validation_dataset = IODataset(val_x, val_y)

	# MODEL SETUP
	model_instance = get_model_instance_from_string(meta_config, experiment_config, meta_config.model, in_shape=x.shape, out_shape=y.shape)

	# make data
	dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if device_decision.device=="cuda" else {}
	dataloader_kwargs = {}
	effective_batch_size = min(experiment_config.batch_size, len(training_dataset))
	training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=effective_batch_size, shuffle=True, **dataloader_kwargs)
	validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=effective_batch_size, shuffle=False, **dataloader_kwargs)

	# setup the optimiser
	optimizer = get_optimizer_from_string(model_instance, experiment_config.optimizer, lr=experiment_config.lr)
	scheduler = ExponentialLR(optimizer, gamma=experiment_config.lr_decay)

	best_test_metrics = {}

	epochs = experiment_config.epochs
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")

		sigma_engagement = 0.0 if t / epochs < experiment_config.sigma_engage else (t/epochs - experiment_config.sigma_engage) / (1 - experiment_config.sigma_engage)
		
		training_loss = curriculum.train(model_instance, training_dataloader, optimizer, sigma_engagement=sigma_engagement, experiment_config=experiment_config, meta_config=meta_config)
		test_metrics = curriculum.test(model_instance, validation_dataloader, experiment_config=experiment_config, meta_config=meta_config)
		
		for metric, value in test_metrics.items():
			best_metric_name = 'best_' + metric
			if best_metric_name not in best_test_metrics or value > best_test_metrics[best_metric_name]:
				best_test_metrics[best_metric_name] = value

		if meta_config.wandbosity >= 1:
			metrics = { **best_test_metrics, **test_metrics }
			if meta_config.wandbosity == 1:
				metrics = { **metrics, 'total_loss': training_loss }
			wandb.log(metrics)

		scheduler.step()
	
	print("Done!")
	return best_test_metrics

def get_optimizer_from_string(model, optimizer_string: str, **kwargs):
	if optimizer_string == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), **kwargs)
	elif optimizer_string == 'adamw':
		optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
	elif optimizer_string == 'adadelta':
		optimizer = torch.optim.Adadelta(model.parameters(), **kwargs)
	elif optimizer_string == 'rmsprop':
		optimizer = torch.optim.RMSprop(model.parameters(), **kwargs)
	elif optimizer_string == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), **kwargs)
	else:
		raise Exception(f"Uknown optimizer {optimizer_string}")

	return optimizer

def get_model_instance_from_string(meta_config, experiment_config, model_string: str, in_shape, out_shape):
	if model_string == 'mlp':
		model_instance = MLP(
			profile=experiment_config.profile + [ out_shape[-1] ],
			dropout=experiment_config.dropout
		)
	elif model_string == 'aig':
		model_instance = AIG(
			input_candidates=2
		)
	elif model_string == 'incremental_aig_net':
		model_instance = IncrementalUnitNet[AIG](
			unit_factory=lambda x: AIG(x),
			input_count=in_shape[-1],
			output_count=out_shape[-1],
			unit_count=experiment_config.unit_count,
			layer_width=experiment_config.unit_layer_width,
			use_selector=experiment_config.use_selector,
		)
	elif model_string == 'incremental_lut_net':
		model_instance = IncrementalUnitNet[LUT](
			unit_factory=lambda x: LUT(x),
			input_count=in_shape[-1],
			output_count=out_shape[-1],
			unit_count=experiment_config.unit_count,
			layer_width=experiment_config.unit_layer_width,
			use_selector=experiment_config.use_selector,
		)
	elif model_string == 'incremental_lab_net':
		model_instance = IncrementalUnitNet[LAB](
			unit_factory=lambda x: LAB(x),
			input_count=in_shape[-1],
			output_count=out_shape[-1],
			unit_count=experiment_config.unit_count,
			layer_width=experiment_config.unit_layer_width,
			use_selector=experiment_config.use_selector,
		)
	model_instance.to(device_decision.device)

	return model_instance