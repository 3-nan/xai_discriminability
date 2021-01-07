import tensorflow as tf

from .modelinterface import ModelInterface
from ..helpers.analyzer_helper import parse_xai_method
from ..experiments import innvestigate


class TensorflowModel(ModelInterface):

    def __init__(self, model_path):
        """ Initializes the TensorflowModel object. """
        self.model = tf.keras.models.load_model(model_path)
        super().__init__()
        print("Model successfully initialized.")

    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter=None):
        """ Compute relevance maps for the provided data batch. """

        # initialize analyzer
        model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(self.model)
        analyzer = parse_xai_method(xai_method, additional_parameter=additional_parameter)
        ana = analyzer(model_wo_softmax)

        # compute relevance
        r_batch_dict = ana.analyze(batch,
                                   neuron_selection=neuron_selection,
                                   explained_layer_names=layer_names)

        return r_batch_dict
