from ..experiments import innvestigate


INNVESTIGATE_DICT = {
    "Gradient": innvestigate.analyzer.Gradient,
    "SmoothGrad": innvestigate.analyzer.SmoothGrad,
    "LRPZ": innvestigate.analyzer.LRPZ,
    "LRPEpsilon": innvestigate.analyzer.LRPEpsilon,
    "LRPWSquare": innvestigate.analyzer.LRPWSquare,
    "LRPGamma": innvestigate.analyzer.LRPGamma,
    "LRPAlpha1Beta0": innvestigate.analyzer.LRPAlpha1Beta0,
    "LRPAlpha2Beta1": innvestigate.analyzer.LRPAlpha2Beta1,
    "LRPSequentialPresetA": innvestigate.analyzer.LRPSequentialPresetA,
    "LRPSequentialPresetB": innvestigate.analyzer.LRPSequentialPresetB,
    "LRPSequentialCompositeBFlat": innvestigate.analyzer.LRPSequentialCompositeBFlat
}


def parse_xai_method(xai_method, additional_parameter=None):
    """ Parses the method name to the correct innvestigate Analyzer. """
    try:
        analyzer = INNVESTIGATE_DICT[xai_method]
    except KeyError:
        # print("analyzer name {} not correct".format(xai_method))
        # analyzer = None
        raise ValueError("Analyzer name {} not correct.".format(xai_method))
    return analyzer
