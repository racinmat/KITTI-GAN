import os
user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
print(user_paths)

from python.neural_network.Dataset import DataSet