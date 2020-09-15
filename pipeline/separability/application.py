"""The command line interface of the application."""

__version__ = '1.0'

import tensorflow as tf
import pandas as pd
import yaml
import argparse

from . import innvestigate


from .logging import ConsoleLogger
from .datasets import load_cifar10
from .compute_separabilities import evaluate_model_separability


class Application:
	"""Represents the command-line interface application."""

	def __init__(self):
		"""Initializes a new Application instance."""

		# Initializes the logger
		self.logger = ConsoleLogger()

	def run(self):
		"""Runs the application."""

		param_file = self.parse_command_line_args()

		self.logger.set_verbosity("info")

		self.logger.log_info("read param file")
		with open(param_file, 'r') as stream:
			try:
				params = yaml.safe_load(stream)
			except yaml.YAMLError as exc:
				print(exc)

		dataset_path = params['dataset_path']
		dataset = params['dataset']
		model_path = params['model_path']
		model_name = params['model_name']
		xai_method = params['xai_method']
		layer = params['layer']

		self.logger.log_info(dataset_path)

		train_data, test_data = load_cifar10()

		self.logger.log_info("dataset loaded")

		model = tf.keras.models.load_model(model_path)

		# check if model works
		score = model.evaluate(test_data)
		self.logger.log_info("model accuracy is " + str(score[1]))

		# get_relevances_for_layer(train_data, model, 0, )
		results = evaluate_model_separability(model, train_data, test_data, xai_method, layer)

		df = pd.DataFrame([results], columns=[layer])
		df.to_csv("/mnt/output/results.csv", index=False)

	def parse_command_line_args(self):

		argument_parser = argparse.ArgumentParser(
			prog='separability',
			add_help=False
		)

		argument_parser.add_argument(
			'param_file',
			type=str,
			help='the path to the yaml file including the given parameters'
		)

		arguments = argument_parser.parse_args()

		return arguments.param_file
