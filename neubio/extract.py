import pandas


def load_file(path):
    with open(path, 'r') as fd:
        lines = fd.readlines()