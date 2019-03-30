"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
import math
import numpy as np
from scipy import stats

import random

def get_skew_and_kurt(data):

    data = np.array(data)
    # print(data.shape)  # test
    data = data.transpose()
    print(data.shape)  # test

    skew = []
    kurt = []
    for i in data:
        # print(len(i))
        skew.append(stats.skew(i))
        kurt.append(stats.kurtosis(i))

    skew_mean = sum(skew)/len(skew)  # ??
    kurt_mean = sum(kurt)/len(kurt)

    print('skew:', skew_mean)  # test
    print('Kurt:', kurt_mean)  # test

    return skew_mean, kurt_mean



