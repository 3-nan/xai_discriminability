import tensorflow as tf
import tensorflow_addons as tfa

from .modelinterface import ModelInterface
from ..helpers.analyzer_helper import parse_xai_method
from ..experiments import innvestigate


def custom_init(model, model_path):
    """ Implement custom model loading. """
    model.model = tf.keras.models.load_model(model_path, compile=False)
    model.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.6, decay=1e-10),
                        loss=tfa.losses.SigmoidFocalCrossEntropy(),
                        metrics=[tf.keras.metrics.AUC(curve="PR", multi_label=True)])

    return model


class TensorflowModel(ModelInterface):

    def __init__(self, model_path, modelname):
        """ Initializes the TensorflowModel object. """
        try:
            self.model = tf.keras.models.load_model(model_path)
        except ValueError:
            self = custom_init(self, model_path)
        super().__init__(model_path, modelname)
        print("Model successfully initialized.")

    def evaluate(self, data, labels):
        """ Evaluates the model on the given data. """
        return self.model.evaluate(data, labels)

    def predict(self, data, batch_size=None):
        """ Compute predictions for the given data. """
        return self.model.predict(data, batch_size=batch_size)

    def get_layer_names(self, with_weights_only=False):
        """ Returns the layer names of the model. """

        if with_weights_only:
            layer_names = [layer.name for layer in self.model.layers if hasattr(layer, 'kernel_initializer')]

        else:
            layer_names = [layer.name for layer in self.model.layers]

        return layer_names

    def randomize_layer_weights(self, layer_name):
        """ Randomizes the weights of the model in the choosen layer. """

        layer = self.model.get_layer(name=layer_name)
        weights = layer.get_weights()
        weight_initializer = layer.kernel_initializer
        self.model.get_layer(name=layer.name).set_weights([weight_initializer(shape=weights[0].shape), weights[1]])

        return self

    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter=None):
        """ Compute relevance maps for the provided data batch. """

        # initialize analyzer
        if isinstance(self.model.layers[-1], tf.keras.layers.Activation):
            model_wo_softmax = tf.keras.models.Model(inputs=self.model.inputs,
                                                     outputs=self.model.layers[-2].output,
                                                     name=self.model.name)
        else:
            model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(self.model)
        analyzer = parse_xai_method(xai_method, additional_parameter=additional_parameter)
        ana = analyzer(model_wo_softmax)

        # compute relevance
        r_batch_dict = ana.analyze(batch,
                                   neuron_selection=neuron_selection,
                                   explained_layer_names=layer_names)

        return r_batch_dict
