"""The command line interface of the application."""

__version__ = '1.0'

import tensorflow as tf
import pandas as pd
import yaml
import argparse


from .logging import ConsoleLogger
from .datasets import load_cifar10
from .train_vgg16_cifar import train_model


class Application:
	"""Represents the command-line interface application."""

	def __init__(self):
		"""Initializes a new Application instance."""

		# Initializes the logger
		self.logger = ConsoleLogger()

	def run(self):
		"""Runs the application."""

		self.logger.set_verbosity("info")

		train_data, test_data = load_cifar10()

		self.logger.log_info("dataset loaded")

		model = train_model(train_data, test_data)

		# check if model works
		score = model.evaluate(test_data)
		self.logger.log_info("model accuracy is " + str(score[1]))

		model.save("/mnt/output")

