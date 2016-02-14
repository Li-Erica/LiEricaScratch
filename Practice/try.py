import numpy as np
import sklearn as sk

from sk import datasets
iris= datasets.load_iris()
digit=datasets.load_digits()

print(digit.data)