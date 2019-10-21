# -*- Coding: utf-8 -*-
from control.matlab import place as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import * 

print(798.74211953 +0.j < 0)


A = [[-10,5],[-10,0]]
C = [[0],[1]]
K = np.array(pl(A,C,[-6,-6.001]))
print(K)
