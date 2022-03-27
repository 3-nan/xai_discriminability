import copy
import torch
import torch.nn as nn
from zennit.canonizers import *


def setbyname(obj, name, value):
    ##
    def iteratset(obj, components, value):
        # print('components',components)
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            # print('found!!', components[0])
            # exit()
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    ##

    components = name.split('.')
    success = iteratset(obj, components, value)
    return success


class Tensorbiased_ConvLayer(nn.Conv2d):        # nn.Module

    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, newweight, baseconv, inputfornewbias):
        super(Tensorbiased_ConvLayer, self).__init__(baseconv.in_channels, baseconv.out_channels, baseconv.kernel_size)

        self.baseconv = baseconv
        self.inputfornewbias = inputfornewbias
        self.conv = self._clone_module(baseconv)
        self.conv.bias = None

    def gettensorbias(self, x):

        with torch.no_grad():
            tensorbias = self.baseconv(self.inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, x.shape[2], x.shape[3])).unsqueeze(0))
        return tensorbias

    def forward(self, x):

        if len(x.shape) != 4:
            raise NotImplementedError("bad tensor length")
        if self.inputfornewbias is None:
            return self.conv.forward(x)
        else:
            b = self.gettensorbias(x)
            return self.conv.forward(x) + b


class ThresholdReLU(nn.ReLU):

    def __init__(self, thresh, w_bn_sign, forconv):
        super(ThresholdReLU, self).__init__()

        # thresh will be -b_bn*vareps / w_bn + mu_bn
        if forconv:
            self.thresh = thresh.reshape((-1, 1, 1))
            self.w_bn_sign = w_bn_sign.reshape((-1, 1, 1))
        else:
            self.thresh = thresh
            self.w_bn_sign = w_bn_sign

    def forward(self, x):

        return (x - self.thresh) * ((x > self.thresh) * (self.w_bn_sign > 0) + (x < self.thresh) * (self.w_bn_sign < 0)) + self.thresh


class MergeBatchReluConv(Canonizer):
    ''' Class for merging BatchNorm -> ReLU -> Conv Layer structures from DenseNets the Alex Binder way. '''

    def __init__(self):
        super().__init__()

        self.bn = None
        self.relu = None
        self.conv = None

        self.batch_norm_params = None
        self.linear_params = None

    def register(self, bn, relu, conv):

        self.bn = bn
        self.relu = relu
        self.conv = conv

        self.batch_norm_params = {
            key: getattr(self.bn, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }
        self.linear_params = (self.conv.weight.data, getattr(self.conv.bias, 'data', None))
        # self.linear_params = (self.conv.weight.data, self.conv.bias.data)

        self.merge_batch_norm(self.conv, self.relu, self.bn)

    def remove(self):

        self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size)
        self.conv.weight.data = self.linear_params[0]
        if self.linear_params[1]:
            self.conv.bias.data = self.linear_params[1]

        self.relu = nn.ReLU()

        self.bn = nn.BatchNorm2d(self.bn.num_features)
        self.bn.weight.data = self.batch_norm_params['weight']
        self.bn.bias.data = self.batch_norm_params['bias']
        self.bn.running_mean = self.batch_norm_params['running_mean']
        self.bn.running_var = self.batch_norm_params['running_var']


    @staticmethod
    def merge_conv(conv, bn):

        var_bn = (bn.running_var.clone().detach() + bn.eps)**.5
        w_bn = bn.weight.clone().detach()
        bias_bn = bn.bias.clone().detach()
        mu_bn = bn.running_mean.clone().detach()

        newweight = conv.weight.clone().detach() * (w_bn / var_bn).reshape(1, conv.weight.shape[1], 1, 1)

        inputfornewbias = - (w_bn / var_bn * mu_bn) + bias_bn   # size [nchannels]

        if conv.padding == 0:

            ksize = (conv.weight.shape[2], conv.weight.shape[3])
            inputfornewbias2 = inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, ksize[0], ksize[1])).unsqueeze(0)   # shape (1, 64, ksize, ksize)

            with torch.no_grad():
                prebias = conv(inputfornewbias2)

            mi = ((prebias.shape[2] - 1) // 2, (prebias.shape[3] - 1) // 2)
            prebias = prebias.clone().detach()
            newconv_bias = prebias[0, :, mi[0], mi[1]]

            conv2 = copy.deepcopy(conv)
            conv2.weight = nn.Parameter(newweight)
            conv2.bias = nn.Parameter(newconv_bias)

            return conv2

        else:
            # bias is a tensor, if there is padding
            spatiallybiasedconv = Tensorbiased_ConvLayer(newweight, conv,inputfornewbias.clone().detach())

            return spatiallybiasedconv
            # raise NotImplementedError("We have a problem here.")

    # @staticmethod
    def merge_batch_norm(self, conv, relu, batch_norm):

        # bn-relu-conv chain
        assert (isinstance(batch_norm, nn.BatchNorm2d))
        assert (isinstance(conv, nn.Conv2d))

        # get the right threshrelu/clamplayer
        var_bn = (batch_norm.running_var + batch_norm.eps)**.5
        if torch.norm(batch_norm.weight) > 0.:
            thresh = - batch_norm.bias * var_bn / batch_norm.weight + batch_norm.running_mean
            thresh_relu = ThresholdReLU(thresh, torch.sign(batch_norm.weight), forconv=True)

        # success = setbyname(obj, name=relu, value=thresh_relu)
        relu = thresh_relu
        # if not success:
        #     raise ModuleNotFoundError(' could not find ', relu)


        # get the right convolution, likely with tensorbias
        merged_conv = self.merge_conv(conv, batch_norm)

        # success = setbyname(conv, name=conv, value=merged_conv)
        # if not success:
        #     raise ModuleNotFoundError(' could not find ', conv)
        conv = merged_conv

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)


class NamedMergeDense(MergeBatchReluConv):
    ''' Canonizer to merge the BN -> ReLU -> Conv combinations, specified by their respective names.
    '''

    def __init__(self, name_map):
        super(NamedMergeDense, self).__init__()
        self.name_map = name_map

    def apply(self, root_module):

        instances = []
        lookup = dict(root_module.named_modules())

        for batch_norm_name, relu_name, linear_name in self.name_map:
            instance = self.copy()
            instance.register(lookup[batch_norm_name], lookup[relu_name], lookup[linear_name])
            instances.append(instance)

        return instances

    def copy(self):
        return self.__class__(self.name_map)
