import argparse
import time

def setup_parser():
	# meta config zone
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-j',
		'--job-id',
		type=int,
		default=int(time.time()),
		help='The job id (and the name of the wandb group)'
	)
	parser.add_argument(
		'-o',
		'--output-path',
		type=str,
		default="training_bests",
		help='The directory which will contain job sub-directories containing the checkpoints of best models'
	)

	parser.add_argument(
		'-i',
		'--input-path',
		type=str,
		default="./data",
		help='The path to the directory containing the data to use (default: ./data)'
	)

	parser.add_argument(
		'--seed',
		type=int,
		default=1235,
		help='The seed for torch, numpy, and python randomness (default: 1234)'
	)
	parser.add_argument(
		'--model',
		type=str,
		default='incremental_aig_net',
		choices=['mlp', 'aig', 'incremental_aig_net', 'incremental_lut_net', 'incremental_lab_net' ],
		help='Choose which model to use (default: mlp)'
	)
	parser.add_argument(
		'--task',
		type=str,
		help='Choose which task to train the circuit for (no default)'
	)

	parser.add_argument(
		'-m',
		'--mode',
		type=str,
		default='single',
		choices=['single', 'sweep'],
		help='Choose whether to do a single evaluation run or whether to start a sweep agent (default: single)'
	)
	parser.add_argument(
		'--sweep-id',
		type=str,
		default=0,
		help='The id of the sweep to connect to (usually a string rather than a number)'
	)
	parser.add_argument(
		'--sweep-runs',
		type=int,
		default=1,
		help='The number of sweep runs to do in this job (default: 1)'
	)
	parser.add_argument(
		'--verbosity',
		type=int,
		default=2,
		help='The terminal output verbosity level (0 is min, 2 is max, default: 2)'
	)
	parser.add_argument(
		'--wandbosity',
		type=int,
		default=2,
		help='The level of verbosity for wandb (0 is min, 2 is max, default: 2)'
	)

	return parser