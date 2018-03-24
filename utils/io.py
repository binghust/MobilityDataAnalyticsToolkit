# encoding=utf-8

import pickle


def save(var, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(var, fh)
        print('%s is saved.' % filename)


def load(filename):
    with open(filename, 'rb') as fh:
        var = pickle.load(fh)
        return var
