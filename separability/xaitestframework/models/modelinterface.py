from abc import ABC, abstractmethod


class ModelInterface(ABC):

    def __init__(self):
        """ Initialize the model. """
        super().__init__()

    @abstractmethod
    def evaluate(self, data, labels):
        """ Evaluate the performance of the model on the given data. """
        return NotImplementedError

    @abstractmethod
    def predict(self, data):
        """ Compute model predictions for the given data. """
        return NotImplementedError

    @abstractmethod
    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter):
        """ Computes relevance maps for the given data and labels. """
        return NotImplementedError
