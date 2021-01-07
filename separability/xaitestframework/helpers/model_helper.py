from ..models.tensorflowmodel import TensorflowModel


def init_model(model_path, framework="tensorflow"):
    """ Initializes the model in the correct framework. """
    if framework == "tensorflow":
        model = TensorflowModel(model_path)

    return model
