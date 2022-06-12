# Import Modules
import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from M3 import generate_m3

import sys
from trun_mvnt import rtmvn,rtmvt
import math

# Setup Cluster

data = generate_m3(theta, 0.1,respOpt,500)


# Sample Parameter Sets from Multivariate Normal
ptive_inf = float('inf')
ntive_inf = float('-inf')

mu_c, sigma_c = 10,20
mu_a, sigma_a = 2,10
mu_f, sigma_f = 1,10

D = np.diag(np.ones(3))
mu = np.zeros(3)
sig = np.diag([sigma_c**2, sigma_a**2, sigma_f**2, 0.0001])
lower = np.array([0,0,-10e100])
upper = np.array([10e100,10e100,10e100,10e100])

Datasets = 10
nSubjects = 1000
ntheta = 3


theta = np.array([np.zeros(Datasets),np.zeros(nSubjects, ntheta)])

iter = np.nditer()

