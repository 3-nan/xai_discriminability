# from ..models.tensorflowmodel import TensorflowModel
from ..models.pytorchmodel import PytorchModel


def init_model(model_path, modelname, framework="tensorflow"):
    """ Initializes the model in the correct framework. """
    if framework == "tensorflow":
        # model = TensorflowModel(model_path, modelname)
        raise ValueError("TensorflowModel not imported.")
    elif framework == "pytorch":
        model = PytorchModel(model_path, modelname)
    else:
        raise ValueError("No Modelinterface for {} found. Please check in helpers/model_helper.py".format(framework))

    return model
