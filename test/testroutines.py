import numpy as np
from PIL import Image

class TiffReader(object):
    def imread(self, filename):
        return np.asarray(Image.open(filename))


###############################################################################
def printParametersToString(pars):
    txt = r''
    for key, value in pars.items():
        if key == 'algorithm':
            txt += "{0} = {1}".format(key, value.__name__)
        elif key == 'input':
            txt += "{0} = {1}".format(key, np.shape(value))
        elif key == 'refdata':
            txt += "{0} = {1}".format(key, np.shape(value))
        else:
            txt += "{0} = {1}".format(key, value)
        txt += '\n'
    return txt


def nrmse(im1, im2):
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(im1.size))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))


def rmse(im1, im2):
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(im1.size))
    return rmse


###############################################################################
