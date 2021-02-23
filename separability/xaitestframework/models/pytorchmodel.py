import torch
import torch.nn as nn
import torchvision.models as models
import captum

from .modelinterface import ModelInterface
# from ..helpers.analyzer_helper import parse_xai_method
# from ..experiments import innvestigate


def parse_captum_method(xai_method, additional_parameter=None):
    """ Parse method name to captum method. """
    if xai_method == "IntegratedGradients":
        method = captum.attr.IntegratedGradients
    elif xai_method == "GradientAttribution":
        method = captum.attr.GradientAttribution
    elif xai_method == "Deconvolution":
        method = captum.attr.Deconvolution
    elif xai_method == "Lime":
        method = captum.attr.Lime
    else:
        raise NotImplementedError("Method {} is not implemented/may needs to be added manually.".format(xai_method))

    return method


def get_pytorch_model(modelname):

    modeldict = {
        "vgg16": models.vgg16,
        "densenet161": models.densenet161,
        "inception": models.inception_v3
    }

    return modeldict[modelname]()


class PytorchModel(ModelInterface):

    def __init__(self, model_path, modelname):
        """ Initializes the TensorflowModel object. """
        self.model = get_pytorch_model(modelname)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # build layer name dictionary
        conv_counter = 1
        linear_counter = 1
        self.layer_names = {}
        for layer in self.model.modules():
            if type(layer) not in [nn.Module, nn.Sequential]:
                if type(layer) == nn.Conv2d:
                    self.layer_names["conv" + str(conv_counter)] = layer
                    conv_counter += 1
                elif type(layer) == nn.Linear:
                    self.layer_names["linear" + str(linear_counter)] = layer
                    linear_counter += 1

        super().__init__(model_path, modelname, "pytorch")
        print("Model successfully initialized.")

    def evaluate(self, data, labels):
        """ Evaluates the model on the given data. """
        outputs = self.model(data)                      # ToDo: evaluation methods might change and might be only applicable to whole datasets
        return self.model.evaluate(data, labels)

    def predict(self, data, batch_size=None):
        """ Compute predictions for the given data. """
        outputs = self.model(data)                      # ToDo: implement usage of batch size, implement prediction
        return outputs

    def get_layer_names(self, with_weights_only=False):
        """ Returns the layer names of the model. """

        if with_weights_only:
            # layer_names = [layer.name for layer in self.model.layers if hasattr(layer, 'kernel_initializer')]   # ToDo
            layers = self.layer_names.keys()
            # for module in self.model.modules():
            #     if type(module) not in [nn.Module, nn.Sequential]:
            #         if hasattr(module, "weight"):
            #             layers.append(module)
        else:
            # layers = [layer for layer in self.model.modules() if type(layer) not in [nn.Module, nn.Sequential]]
            # alternatively self.model.named_modules (contains, (string, Module) tuples
            return NotImplementedError("get layer names without weights only not implemented")

        # build layer - layer name mapping

        return layers

    def randomize_layer_weights(self, layer_name):
        """ Randomizes the weights of the model in the choosen layer. """

        # layer_name.weight.data.fill_(0.01)
        layer = self.layer_names[layer_name]
        layer.reset_parameters()
        # alternatively self.model.state_dict()[layer_name]['weight']
        # layer = self.model.get_layer(name=layer_name)
        # weights = layer.get_weights()
        # weight_initializer = layer.kernel_initializer
        # self.model.get_layer(name=layer.name).set_weights([weight_initializer(shape=weights[0].shape), weights[1]])

        return self

    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter=None):
        """ Compute relevance maps for the provided data batch. """

        # initialize analyzer
        # if isinstance(self.model.layers[-1], tf.keras.layers.Activation):
        #     model_wo_softmax = tf.keras.models.Model(inputs=self.model.inputs,
        #                                              outputs=self.model.layers[-2].output,
        #                                              name=self.model.name)
        # else:
        #     model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(self.model)

        analyzer = parse_captum_method(xai_method, additional_parameter=additional_parameter)
        ana = analyzer(self.model)

        # compute relevance
        r_batch_dict = ana.attribute(batch, target=neuron_selection)    # baseline
        # explained_layer_names=layer_names)

        return r_batch_dict
