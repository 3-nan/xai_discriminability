import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import zennit.attribution
from captum.attr import (Deconvolution, GradientAttribution, NoiseTunnel, IntegratedGradients, Lime, Saliency,
                         LayerGradientXActivation,
                         LayerGradCam,
                         LayerDeepLift,
                         LayerIntegratedGradients)
from zennit.torchvision import VGGCanonizer, ResNetCanonizer, SequentialMergeBatchNorm
from zennit.canonizers import NamedMergeBatchNorm
from zennit.rules import *
from zennit.composites import *

from ..explainers.composites import *
from ..explainers.canonizers import *
# from ..explainers.zennit.torchvision import VGGCanonizer
# from ..explainers.zennit.composites import COMPOSITES, LAYER_MAP_BASE, LayerMapComposite
# from ..explainers.zennit.rules import *
# from ..explainers.composites import *
from .modelinterface import ModelInterface


CAPTUM_LAYERWISE = ["IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift"]

CAPTUM_DICT = {
    "Saliency": Saliency,
    # "IntegratedGradients": IntegratedGradients,
    # "GradientAttribution": GradientAttribution,
    "Deconvolution": Deconvolution,
    "SmoothGrad": Saliency,
    # "Lime": Lime,
    "GradientXActivation": LayerGradientXActivation,
    "GradCam": LayerGradCam,
    "DeepLift": LayerDeepLift,
    # "IntegratedGradients": LayerIntegratedGradients,
}

ZENNIT_GRADIENT = {
    "Gradient": LayerMapComposite,
    "IntegratedGradients": LayerMapComposite,
    "GradientxInput": LayerMapComposite,
}
# ZENNIT_DICT = {
#     "epsilon": LayerMapComposite(layer_map=LAYER_MAP_BASE + [(nn.Module, Epsilon())])
# }


def parse_captum_method(xai_method, additional_parameter=None):
    """ Parse method name to captum method. """
    try:
        method = CAPTUM_DICT[xai_method]
    except KeyError:
        # print("analyzer name {} not correct".format(xai_method))
        # analyzer = None
        raise ValueError("Analyzer name {} not correct.".format(xai_method))
    return method


def get_zennit_composite(xai_method, model, shape=None):
    """ Get the composite based on the xai_method. """
    composite_kwargs = {}
    if xai_method == 'epsilon_gamma_box':

        # the highest and lowest pixel values for the ZBox rule
        composite_kwargs['low'] = -1 * torch.ones(*shape, device=model.device)
        composite_kwargs['high'] = torch.ones(*shape, device=model.device)

    # use torchvision specific canonizers, as supplied in the MODELS dict
    if model.name == "vgg16":
        composite_kwargs['canonizers'] = [VGGCanonizer()]  # [MODELS[model_name][1]()]
    elif model.name == "vgg16bn":
        composite_kwargs['canonizers'] = [VGGCanonizer()]
        # print("no canonizer")
    elif model.name == "resnet18":
        composite_kwargs['canonizers'] = [ResNetCanonizer()]
        # print("no canonizer")
    elif model.name == "densenet121":
        # composite_kwargs['canonizers'] = [NamedMergeBatchNorm([
        #     [('features.conv0',), 'features.norm0'],
        #     [["features.denseblock1.denselayer1.conv1"], "features.denseblock1.denselayer1.norm2"],
        #     # [["features.denseblock1.denselayer1.conv2"], "features.denseblock1.denselayer2.norm1"]
        #     [["features.denseblock1.denselayer2.conv1"], "features.denseblock1.denselayer2.norm2"],
        #     # [["features.denseblock1.denselayer1.conv2", "features.denseblock1.denselayer2.conv2"], "features.denseblock1.denselayer3.norm1"],
        #     [["features.denseblock1.denselayer3.conv1"], "features.denseblock1.denselayer3.norm2"],
        #     [["features.denseblock1.denselayer4.conv1"], "features.denseblock1.denselayer4.norm2"],
        #     [["features.denseblock1.denselayer5.conv1"], "features.denseblock1.denselayer5.norm2"],
        #     [["features.denseblock1.denselayer6.conv1"], 'features.denseblock1.denselayer6.norm2'],
        # ])]
        canonizers = [NamedMergeBatchNorm([(('features.conv0',), 'features.norm0')])]
        #
        # layer_conv = [6, 12, 24, 16]
        #
        # for block, block_length in enumerate(layer_conv):
        #     for layer in range(block_length):
        #         for item in [1, 2]:
        #             canonizers.append(
        #                 NamedMergeDense([(
        #                     ('features.denseblock{}.denselayer{}.norm{}'.format(block+1, layer+1, item)),
        #                     ('features.denseblock{}.denselayer{}.relu{}'.format(block+1, layer+1, item)),
        #                     ('features.denseblock{}.denselayer{}.conv{}'.format(block+1, layer+1, item))
        #                 )])
        #             )
        # canonizers.append(NamedMergeBatchNorm([(('classifier',), 'features.norm5')]))
        # composite_kwargs['canonizers'] = [CompositeCanonizer(canonizers)]
        # print("no canonizer")
    else:
        raise ValueError("Model name not known")

    # create a composite specified by a name; the COMPOSITES dict includes all preset composites
    # provided by zennit.
    if xai_method in ZENNIT_GRADIENT.keys():
        composite = ZENNIT_GRADIENT[xai_method]
        composite = composite({}, **composite_kwargs)
    else:
        composite = COMPOSITES[xai_method]
        composite = composite(**composite_kwargs)

    return composite


def hook(module, input):
    module.saved_input = input[0]
    module.saved_input.retain_grad()
    module.saved_input.requires_grad_()


def get_pytorch_model(modelname, num_classes):

    modeldict = {
        "vgg16": models.vgg16,
        "vgg16bn": models.vgg16_bn,
        "resnet18": models.resnet18,
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "inception": models.inception_v3
    }

    model = modeldict[modelname]()

    # adapt last layer
    if modelname == "vgg16" or modelname == "vgg16bn":
        model.classifier[-1] = nn.Linear(4096, num_classes)      # TODO change num_classes
    elif modelname == "resnet18":
        model.fc = nn.Linear(512, num_classes)
    elif modelname == "densenet121":
        print("continue at get pytorch model")
    else:
        raise ValueError("please specify get pytorch model")

    return model


class PytorchModel(ModelInterface):

    def __init__(self, model_path, modelname):
        """ Initializes the TensorflowModel object. """
        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # HARD CODED
        self.num_classes = 1000  # 1000     # 20  # 1000

        # init
        self.model = get_pytorch_model(modelname, self.num_classes)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # build layer name dictionary
        conv_counter = 1
        linear_counter = 1
        self.layer_dict = {}
        for layer in self.model.modules():
            if type(layer) not in [nn.Module, nn.Sequential]:
                if type(layer) == nn.Conv2d:
                    self.layer_dict["conv" + str(conv_counter)] = layer
                    conv_counter += 1
                elif type(layer) == nn.Linear:
                    self.layer_dict["linear" + str(linear_counter)] = layer
                    linear_counter += 1

        super().__init__(model_path, modelname, "pytorch")
        print("Model successfully initialized.")

    def evaluate(self, data, labels):
        """ Evaluates the model on the given data. """
        outputs = self.model(torch.as_tensor(data, device=self.device))                      # ToDo: evaluation methods might change and might be only applicable to whole datasets
        return self.model.evaluate(data, labels)

    def predict(self, data, batch_size=None):
        """ Compute predictions for the given data. """
        data_tensor = torch.as_tensor(data, device=self.device).permute(0, 3, 1, 2)
        outputs = self.model(data_tensor)          # ToDo: implement usage of batch size, implement prediction

        # activation functionality
        sigmoid = torch.nn.Sigmoid()

        outputs = sigmoid(outputs)

        return outputs.detach().cpu().numpy()

    def get_activations(self, data, layer_name):
        """ Compute activations for the specified layer. """
        data_tensor = torch.as_tensor(data, device=self.device).permute(0, 3, 1, 2)

        global activations

        def hook_fn(module, input, output):
            global activations
            activations = output

        hook = self.layer_dict[layer_name].register_forward_hook(hook_fn)

        outputs = self.model(data_tensor)

        hook.remove()
        if layer_name in ["linear1", "linear2", "linear3"]:
            return activations.detach().cpu().numpy()
        else:
            return activations.detach().permute(0, 2, 3, 1).cpu().numpy()

    def get_layer_names(self, with_weights_only=False):
        """ Returns the layer names of the model. """

        if with_weights_only:
            # layer_names = [layer.name for layer in self.model.layers if hasattr(layer, 'kernel_initializer')]   # ToDo
            layers = list(self.layer_dict.keys())
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
        layer = self.layer_dict[layer_name]
        layer.reset_parameters()
        # alternatively self.model.state_dict()[layer_name]['weight']
        # layer = self.model.get_layer(name=layer_name)
        # weights = layer.get_weights()
        # weight_initializer = layer.kernel_initializer
        # self.model.get_layer(name=layer.name).set_weights([weight_initializer(shape=weights[0].shape), weights[1]])

        return self

    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter=None):
        """ Compute relevance maps for the provided data batch. """

        # convert layer_names to layers
        layers = [self.layer_dict[layer_name] for layer_name in layer_names]

        if xai_method in CAPTUM_DICT.keys():
            # Captum attribution computation
            analyzer = parse_captum_method(xai_method, additional_parameter=additional_parameter)

            if xai_method not in CAPTUM_LAYERWISE:
                ana = analyzer(self.model)

                if xai_method == "SmoothGrad":
                    ana = NoiseTunnel(ana)

                # compute relevance
                batch_tensor = torch.as_tensor(batch, device=self.device).permute(0, 3, 1, 2)

                if xai_method == "SmoothGrad":
                    r_batch = ana.attribute(batch_tensor, target=neuron_selection, nt_samples=20)
                else:
                    r_batch = ana.attribute(batch_tensor, target=neuron_selection)    # baseline
                # explained_layer_names=layer_names)
                r_batch_dict = {layer_names[0]: r_batch.detach().permute(0, 2, 3, 1).cpu().numpy()}

            else:
                r_batch_dict = {}

                for layer_name, layer in zip(layer_names, layers):

                    ana = analyzer(self.model, layer)

                    # compute relevance
                    batch_tensor = torch.as_tensor(batch, device=self.device).permute(0, 3, 1, 2)

                    r_batch = ana.attribute(batch_tensor, target=neuron_selection, attribute_to_layer_input=True)
                    # for i, layer in enumerate(layer_names):
                    if len(r_batch.size()) == 4:
                        r_batch_dict[layer_name] = r_batch.detach().permute(0, 2, 3, 1).cpu().numpy()
                    else:
                        r_batch_dict[layer_name] = r_batch.detach().cpu().numpy()

        elif xai_method in ZENNIT_GRADIENT.keys():
            r_batch_dict = {}

            # Zennit attribution computation
            batch_tensor = torch.as_tensor(batch, device=self.device).permute(0, 3, 1, 2)
            eye = torch.eye(self.num_classes, device=self.device)
            targets = np.ones(len(batch)) * neuron_selection

            composite = get_zennit_composite(xai_method, self, shape=list(batch_tensor.size()))

            if xai_method == "Gradient" or xai_method == "GradientxInput":
                attributor = zennit.attribution.Gradient(model=self.model, composite=composite)
            elif xai_method == "IntegratedGradients":
                attributor = zennit.attribution.IntegratedGradients(model=self.model, composite=composite,
                                                                    **{'n_iter': 10})
            else:
                print("No Zennit strategy for handling {} is implemented".format(xai_method))

            with attributor:
                output, relevance = attributor(batch_tensor, eye[targets])

                if xai_method == "GradientxInput":
                    r = relevance.detach().permute(0, 2, 3, 1).cpu().numpy()
                    r_batch_dict[layer_names[0]] = r * batch
                else:
                    r_batch_dict[layer_names[0]] = relevance.detach().permute(0, 2, 3, 1).cpu().numpy()
        else:
            # get handles for chosen layers
            handles = [layer.register_forward_pre_hook(hook) for layer in layers]

            r_batch_dict = {}

            # Zennit attribution computation
            batch_tensor = torch.as_tensor(batch, device=self.device).permute(0, 3, 1, 2)

            eye = torch.eye(self.num_classes, device=self.device)
            targets = np.ones(len(batch)) * neuron_selection

            composite = get_zennit_composite(xai_method, self, shape=list(batch_tensor.size()))

            with composite.context(self.model) as modified:

                batch_tensor.requires_grad_()

                out = modified(batch_tensor)

                output_relevance = eye[targets]/out.abs()       # /out.abs() removes model confidence from attributions
                torch.autograd.backward((out,), (output_relevance,))

                for handle in handles:
                    handle.remove()

            for i, name in enumerate(layer_names):
                r_batch = layers[i].saved_input.grad

                shape = list(r_batch.size())

                if len(shape) == 4:
                    r_batch_dict[name] = r_batch.detach().permute(0, 2, 3, 1).cpu().numpy()
                else:
                    r_batch_dict[name] = r_batch.detach().cpu().numpy()

            # r_batch = batch_tensor.grad

            # r_batch_dict = {layer_names[0]: r_batch.detach().permute(0, 2, 3, 1).cpu().numpy()}

        return r_batch_dict
