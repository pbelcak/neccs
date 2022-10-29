import torch
import math

def make_binary_array(number: int, length: int) -> list:
	return [ (number>>k)&1 for k in range(0, length) ]

def make_data_bitwise_not(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width)
			for x in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(max_number - x - 1, width)
			for x in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_and(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x & y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_or(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x | y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_xor(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x ^ y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_shr(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width)
			for x in range(max_number)
			for shift in range(width)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x >> shift, width)
			for x in range(max_number)
			for shift in range(width)
	], dtype=torch.float64)

	return x, y

def make_data_shl(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width)
			for x in range(max_number)
			for shift in range(width)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x << shift, width)
			for x in range(max_number)
			for shift in range(width)
	], dtype=torch.float64)

	return x, y

def make_data_negation(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width)
			for x in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(max_number - x, width)
			for x in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_addition(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x + y, width+1)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_subtraction(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(max_number + x - y, width+1)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_multiplication(width: int):
	max_number = 2**width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, width) + make_binary_array(y, width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x * y, 2*width)
			for x in range(max_number)
			for y in range(max_number)
	], dtype=torch.float64)

	return x, y

def make_data_division(numerator_width: int, denominator_width: int):
	max_numerator = 2**numerator_width
	max_denominator = 2**denominator_width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, numerator_width) + make_binary_array(y, denominator_width)
			for x in range(max_numerator)
			for y in range(max_denominator)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x // y if y > 0 else 0, numerator_width)
			for x in range(max_numerator)
			for y in range(max_denominator)
	], dtype=torch.float64)

	return x, y

def make_data_modulo(numerator_width: int, denominator_width: int):
	max_numerator = 2**numerator_width
	max_denominator = 2**denominator_width

	x = torch.tensor([
		[0, 1, ] + make_binary_array(x, numerator_width) + make_binary_array(y, denominator_width)
			for x in range(max_numerator)
			for y in range(max_denominator)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(x % y if y > 0 else 0, numerator_width)
			for x in range(max_numerator)
			for y in range(max_denominator)
	], dtype=torch.float64)

	return x, y

def make_data_multiplexer(num_lines: int):
	max_number = 2**num_lines
	select_width = int(math.log2(num_lines))
	max_select = 2**select_width

	x = torch.tensor([
		make_binary_array(signals, num_lines) + make_binary_array(select, select_width)
			for signals in range(max_number)
			for select in range(max_select)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(make_binary_array(signals, num_lines)[select], 1)
			for signals in range(max_number)
			for select in range(max_select)
	], dtype=torch.float64)

	return x, y

def make_data_demultiplexer(num_lines: int):
	max_number = 2**num_lines
	select_width = int(math.log2(num_lines))
	max_select = 2**select_width

	x = torch.tensor([
		make_binary_array(signal, 1) + make_binary_array(select, select_width)
			for signal in range(2)
			for select in range(max_select)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(signal << select, num_lines)
			for signal in range(2)
			for select in range(max_select)
	], dtype=torch.float64)

	return x, y

def make_data_decoder(select_width: int):
	max_select = 2**select_width

	x = torch.tensor([
		make_binary_array(select, select_width)
			for select in range(max_select)
	], dtype=torch.float64)
	y = torch.tensor([
		make_binary_array(2**select, max_select)
			for select in range(max_select)
	], dtype=torch.float64)

	return x, y

def make_data_priority_encoder(num_lines: int):
	max_number = 2**num_lines
	encoded_width = int(math.ceil(math.log2(num_lines)))

	x = torch.tensor([
		make_binary_array(signals, num_lines)
			for signals in range(max_number)
	], dtype=torch.float64)
	y = torch.tensor([
		([1] + make_binary_array(make_binary_array(signals, num_lines).index(1), encoded_width) if 1 in make_binary_array(signals, num_lines) else [0] + make_binary_array(0, encoded_width))
			for signals in range(max_number)
	], dtype=torch.float64)

	return x, y