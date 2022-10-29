import torch

torch.set_default_dtype(torch.float64)

class IODataset(torch.utils.data.Dataset):
	def __init__(self, x, y, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.x = x.clone().detach().requires_grad_(True)
		self.y = y.clone().detach().requires_grad_(True)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = [ self.x[idx], self.y[idx] ]
		return sample

	def __len__(self):
		return self.x.size(0)
