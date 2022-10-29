import torch

import wandb
from . import device_decision

def train(model, dataloader, optimizer, sigma_engagement, experiment_config, meta_config):
	device = device_decision.device

	# training begins here
	size = len(dataloader.dataset)
	last_print_point = 0
	epoch_loss_total = 0
	epoch_loss_sharpening = 0

	model.train()
	for batch, (X, y) in enumerate(dataloader):
		current_point = batch * len(X)
		X, y = X.to(device), y.to(device)

		# forward pass
		pred = model(X)

		# loss computation
		loss_reconstruction = reconstruction_loss(pred, y)
		loss_sharpening = sharpening_loss(model)
		loss_sharpening_val = loss_sharpening.item()

		loss_total = loss_reconstruction + sigma_engagement * experiment_config.sigma * loss_sharpening
		loss_total_val = loss_total.item()
		epoch_loss_total += loss_total_val
		epoch_loss_sharpening += loss_sharpening_val

		if meta_config.wandbosity >= 2:
			metrics = {
				'loss_total': loss_total_val
			}
			wandb.log(metrics)
		
		# backpropagation
		optimizer.zero_grad()
		loss_total.backward()
		optimizer.step()

		# print progress
		if meta_config.verbosity >= 2 and current_point - last_print_point > size//10:
			last_print_point = current_point
			current = batch * len(X)
			print(f" - loss: total {loss_total_val: >7.3f}, sharpening {loss_sharpening_val: >7.3f} [{current:>5d}/{size:>5d}]", end="\t\t\r")

		del loss_total, pred, X, y

	batch = batch + 1
	
	epoch_loss_total /= batch
	epoch_loss_sharpening /= batch

	if meta_config.verbosity >= 1:
		print('\x1b[2K', end="\r") # line clear
		print(
			f" - mean train total loss: \t{epoch_loss_total: >7.3f}",
			end="\t\t\t\n"
		)
		print(
			f" - mean train sharpening loss: \t{epoch_loss_sharpening: >7.3f}, \tsigma'd {epoch_loss_sharpening * experiment_config.sigma: >7.3f} ",
			end="\t\t\t\n"
		)

	return epoch_loss_total

def sharpening_loss(model):
	loss = 0
	for unit in model.units:
		loss += unit.sharpening_loss()
	return loss / len(model.units)

def reconstruction_loss(predictions, y):
	predictions = torch.clamp(predictions, 0, 1)
	loss = torch.nn.BCELoss()(predictions, y)
	
	return loss

def test(model, dataloader, experiment_config, meta_config):
	device = device_decision.device

	# training begins here
	size = len(dataloader.dataset)
	last_print_point = 0
	accumulated_row_accuracy = 0
	accumulated_element_accuracy = 0
	elements_evaluated = 0
	rows_evaluated = 0

	with torch.no_grad():
		model.eval()
		for batch, (X, y) in enumerate(dataloader):
			current_point = batch * len(X)
			X, y = X.to(device), y.to(device)

			# forward pass
			pred = model(X)
			comprehended_predictions = torch.round(pred)
			agreements = comprehended_predictions == y
			correct_rows_count = agreements.all(dim=1).sum().item()
			accumulated_row_accuracy += correct_rows_count
			correct_element_count = torch.count_nonzero(agreements)
			accumulated_element_accuracy += correct_element_count.item()
			elements_evaluated += y.numel()
			rows_evaluated += y.shape[0]

			del pred, X, y
	
	element_accuracy_final = accumulated_element_accuracy / elements_evaluated
	row_accuracy_final = accumulated_row_accuracy / rows_evaluated

	if meta_config.verbosity >= 1:
		print('\x1b[2K', end="\r") # line clear
		print(
			f" - mean accuracy: element \t{element_accuracy_final: >7.3f}, row \t{row_accuracy_final: >7.3f}",
			end="\t\t\t\n"
		)

	return {
		'accuracy_element': element_accuracy_final,
		'accuracy_row': row_accuracy_final,
		'raw_element_count': elements_evaluated,
		'raw_row_count': rows_evaluated,
		'raw_correct_element_count': accumulated_element_accuracy,
		'raw_correct_row_count': accumulated_row_accuracy
	}