import torch
import os
import settings


def save_model_params(model, name):
    """ Saves the parameters of the specified model.
    :param model: Model to use.
    :param name: Extra name used to namespace the model.
    :return: Nothing
    """
    path = os.path.join(os.getcwd(), settings.CHECKPOINT_DIR, name)
    torch.save(model.state_dict(), path)


def load_model_params(model, name):
    """ Loads parameters for specified model. Analogous to save_model_params() """
    path = os.path.join(os.getcwd(), name)
    model.load_state_dict(torch.load(path))


def generate_model_from_settings():
    """ Uses the information in the settings.MODEL to generate a model """
    return settings.MODEL["model"](**settings.MODEL)
