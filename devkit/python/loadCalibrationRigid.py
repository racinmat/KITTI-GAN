from smop.core import *
import ruamel.yaml as yaml


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


@function
def readVariable(data=None, name=None, M=None, N=None):
    if name not in data:
        return []

    if M != 1 or N != 1:
        values = np.array(data[name].split(), dtype=float)
        values = values.reshape(M, N)
        return values
    else:
        return data[name]
