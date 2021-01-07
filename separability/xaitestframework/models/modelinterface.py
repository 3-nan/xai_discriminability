from abc import ABC, abstractmethod


class ModelInterface(ABC):

    def __init__(self):
        """ Initialize the model. """
        super().__init__()

    @abstractmethod
    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter):
        """ Computes relevance maps for the given data and labels. """
        return NotImplementedError
