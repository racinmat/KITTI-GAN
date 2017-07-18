from smop.core import *
import ruamel.yaml as yaml
from devkit.python.utils import readVariable


@function
def loadCalibrationRigid(filename=None):
    # open file
    with open(filename, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

    # read calibration
    R = readVariable(data, 'R', 3, 3)
    T = readVariable(data, 'T', 3, 1)
    Tr = np.concatenate((np.concatenate((R, T), axis=1), np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return Tr

