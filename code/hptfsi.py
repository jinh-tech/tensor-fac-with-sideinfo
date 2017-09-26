import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

#from path import path
from argparse import ArgumentParser
from utils import *


def main():
	p = ArgumentParser()
	p.add_argument('-d', '--data', type=path, required=True)
	p.add_argument('-o', '--out', type=path, required=True)
	p.add_argument('-m', '--mask', type=path, default=None)
	p.add_argument('-k', '--n_components', type=int, required=True)
	p.add_argument('-n', '--max_iter', type=int, default=200)
	p.add_argument('-t', '--tol', type=float, default=1e-4)
	p.add_argument('-s', '--smoothness', type=int, default=100)
	p.add_argument('-a', '--alpha', type=float, default=0.1)
	p.add_argument('-v', '--verbose', action="store_true", default=False)
	p.add_argument('--debug', action="store_true", default=False)
	args = p.parse_args()




if __name__ == '__main__':
	main()