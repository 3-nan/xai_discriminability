import torch
from zennit.composites import register_composite, LayerMapComposite, SpecialFirstLayerMapComposite, LAYER_MAP_BASE
from zennit.rules import *
from zennit.types import Convolution, Linear


@register_composite("epsilon")
class EpsilonComposite(LayerMapComposite):

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Linear, Epsilon())
        ]

        super().__init__(layer_map, canonizers=canonizers)


@register_composite("epsilon_gamma")
class EpsilonGamma(LayerMapComposite):

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Gamma(gamma=0.25)),
            (torch.nn.Linear, Epsilon()),
        ]

        super().__init__(layer_map, canonizers=canonizers)


@register_composite("epsilon_gamma_test")
class EpsilonGammaTest(SpecialFirstLayerMapComposite):          # SpecialFirstLayerMapComposite

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Gamma(gamma=2)),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, ZPlus())
        ]

        # super().__init__(layer_map, canonizers=canonizers)
        super().__init__(layer_map, first_map, canonizers=canonizers)


@register_composite("alpha1_beta0")
class Alpha1Beta0(LayerMapComposite):

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Linear, AlphaBeta(alpha=1, beta=0))
        ]

        super().__init__(layer_map, canonizers=canonizers)


@register_composite("alpha2_beta1")
class Alpha2Beta1(LayerMapComposite):

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Linear, AlphaBeta(alpha=2, beta=1))
        ]

        super().__init__(layer_map, canonizers=canonizers)


@register_composite("alpha2_beta1_flat")
class Alpha2Beta1Flat(SpecialFirstLayerMapComposite):

    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Linear, AlphaBeta(alpha=2, beta=1))
        ]
        first_map = [
            (Linear, Flat())
        ]

        super().__init__(layer_map, first_map, canonizers=canonizers)
