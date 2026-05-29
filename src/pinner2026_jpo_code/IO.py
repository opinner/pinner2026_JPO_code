# -------------------------------------------
# Functions for loading data
# -------------------------------------------
import os
import pickle


def save_as_pickle(figure, name="figure"):
    # for example, a matplotlib figure can also be saved in a pickle format
    with open(f"{name}", "wb") as fs:
        pickle.dump(figure, fs)
    return


def load_pickle(name="figure"):
    with open(f"{name}", "rb") as fs:
        return pickle.load(fs)


def get_filepaths_from_directory(directory, inclusive="", exclusive=()):

    # endswith method requires string or tuple of strings
    if type(inclusive) is not str and type(inclusive) is not tuple:
        inclusive = tuple(inclusive)
    if not type(inclusive) is not str and type(inclusive) is not tuple:
        exclusive = tuple(exclusive)

    def all_files(folder):
        for path, _dirs, files in os.walk(folder):
            for f in files:
                yield os.path.join(path, f)

    list_of_files = list(all_files(directory))

    return [f for f in list_of_files if f.endswith(inclusive) and not f.endswith(exclusive)]
