import itertools
import math
import torch
from torch import nn
from torch import Tensor

class MLP(nn.Module):
	def __init__(self, profile: list[int], dropout: float = 0):
		super().__init__()
		
		self.layers = nn.ModuleList(
			[
				nn.Sequential(
					nn.Linear(profile[i], profile[i+1]),
					nn.ReLU(),
					nn.Dropout(dropout)
				) for i in range(len(profile)-2)
			] + [ nn.Linear(profile[-2], profile[-1]) ]
		)

	def forward(self, x: Tensor) -> Tensor:
		for layer in self.layers:
			x = layer(x)

		return torch.sigmoid(x)

class AIG(nn.Module):
	def __init__(self, input_candidates: int):
		super().__init__()
		
		self.left_choice_parameters = nn.Parameter(torch.rand(input_candidates))
		self.right_choice_parameters = nn.Parameter(torch.rand(input_candidates))

	def forward(self, x):
		left_choices = torch.softmax(self.left_choice_parameters, dim=0).unsqueeze(0)
		right_choices = torch.softmax(self.right_choice_parameters, dim=0).unsqueeze(0)

		left_signal = torch.sum(x * left_choices, dim=1)
		right_signal = torch.sum(x * right_choices, dim=1)

		signal = left_signal * right_signal
		return torch.clamp((1 - signal).unsqueeze(1), min=0, max=1)

	def sharpening_loss(self):
		left_choice_entropy = torch.special.entr(torch.softmax(self.left_choice_parameters, dim=0)).sum()
		right_choice_entropy = torch.special.entr(torch.softmax(self.right_choice_parameters, dim=0)).sum()

		return left_choice_entropy + right_choice_entropy
	
from typing import Callable, TypeVar, Generic, List

T = TypeVar('T')

class IncrementalUnitNet(nn.Module, Generic[T]):
	def __init__(self, unit_factory: Callable[[int], T], input_count: int, output_count: int, unit_count: int, layer_width: int = 1, use_selector: bool = True):
		super().__init__()

		self.input_count = input_count
		self.output_count = output_count
		self.unit_count = unit_count
		self.layer_width = layer_width
		self.use_selector = use_selector

		self.units = nn.ModuleList(itertools.chain.from_iterable(
			[ unit_factory(input_count + layer*layer_width) for j in range(layer_width) ]
				for layer in range(math.ceil(unit_count / layer_width))
		))
		self.final_selector = nn.Parameter(torch.randn(input_count+unit_count, output_count))

	def forward(self, x):
		y = x
		for i, unit in enumerate(self.units):
			y = torch.cat((y, unit(x)), dim=1)

			if (i+1) % self.layer_width == 0:
				x = y

		if self.use_selector:
			return torch.matmul(x, torch.softmax(self.final_selector, dim=0))
		else:
			return x[:, -self.output_count:]

	def reset_final_selector(self):
		self.final_selector = nn.Parameter(torch.randn(self.input_count+self.unit_count, self.output_count))

class LUT(nn.Module):
	def __init__(self, input_candidates: int, input_count: int = 4):
		super().__init__()

		self.input_candidates = input_candidates
		self.input_count = input_count

		self.choice_parameters = nn.Parameter(torch.rand(input_candidates, input_count))
		self.lut = nn.Parameter(torch.randn(*[ 2 for i in range(input_count) ]))

	def make_input_choices(self, x):
		choices_per_lut_input_line = torch.softmax(self.choice_parameters, dim=0)
		input_line_signals = torch.matmul(x, choices_per_lut_input_line)

		return input_line_signals

	def forward(self, x):
		input_line_signals = self.make_input_choices(x)

		return self.soft_lookup(input_line_signals, []).unsqueeze(1)

	def soft_lookup(self, x, fixed_dims: list):
		current_dim = len(fixed_dims)
		if current_dim == self.input_count:
			constrained_lut = torch.sigmoid(self.lut)
			return constrained_lut[tuple(fixed_dims)]

		ret = x[:, current_dim] * self.soft_lookup(x, fixed_dims + [0]) + (1 - x[:, current_dim]) * self.soft_lookup(x, fixed_dims + [1])
		return ret
	
	def sharpening_loss(self):
		choice_entropies = torch.special.entr(torch.softmax(self.choice_parameters, dim=0)).sum()

		return choice_entropies

class LAB(nn.Module):
	def __init__(self, input_candidates: int, input_count: int = 4):
		super().__init__()

		self.input_candidates = input_candidates
		self.input_count = input_count

		self.lut = LUT(input_candidates, input_count)
		self.lut_vs_add_choice_parameters = nn.Parameter(torch.rand(3))

	def forward(self, x):
		input_signals = self.lut.make_input_choices(x)

		addition_result = input_signals[:, 0] + input_signals[:, 1] + input_signals[:, 2]
		addition_out_1 = torch.clamp((addition_result - 2).unsqueeze(1), min=0, max=1)
		addition_out_2 = torch.clamp((addition_result >= 2).type(torch.float64).unsqueeze(1), min=0, max=1)

		lut_out = self.lut(x)

		lut_vs_add_choices = torch.softmax(self.lut_vs_add_choice_parameters, dim=0)
		return lut_vs_add_choices[0] * lut_out + lut_vs_add_choices[1] * addition_out_1 + lut_vs_add_choices[2] * addition_out_2
	
	def sharpening_loss(self):
		return self.lut.sharpening_loss()