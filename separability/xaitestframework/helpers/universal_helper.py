import os


def extract_filename(string):
    """ Remove path to file and file ending. """
    return os.path.splitext(string)[0].split("/")[-1]


def join_path(path, dirs):
    """ Join the given path with the presented dirs in list form / string form. """

    if path.endswith("/"):
        path = path[:-1]

    if isinstance(dirs, list):
        for dir in dirs:
            path += "/" + dir
    else:
        path += "/" + dirs

    return path


def compute_relevance_path(relevance_path, data_name, model_name, layer, rule):
    """ Compute path, where relevance maps are stored. """

    relevance_path = join_path(relevance_path, [data_name, model_name, layer, rule])

    return relevance_path
