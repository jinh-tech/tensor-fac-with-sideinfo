import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

# from path import path
from argparse import ArgumentParser
# from utils import *        ## TODO - add util functions

class HPTFSI():
	def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
					smoothness=100, verbose=True, alpha=0.1, debug=False)
		self.n_modes = n_modes
		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.smoothness = smoothness
		self.verbose = verbose
		self.debug = debug

		self.alpha = alpha                                      # shape hyperparameter
		self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

		self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
		self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

		self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
		self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

		self.kappa_shp = np.empty(self.n_modes,dtype=object)
		self.kappa_rte = np.empty(self.n_modes,dtype=object)        
		# Inference cache
		self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
		self.zeta = None
		self.nz_recon_I = None


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
	args.out.makedirs_p()

	assert args.data.exists() and args.out.exists()
	assert args.data.ext == '.npz'
	data_dict = np.load(args.data)
	data = data_dict['data']	#should be skt.sptensor
	side = data_dict['side']	#should be skt.sptensor
	del data_dict
	assert isinstance(data,skt.sptensor)
	assert isinstance(side,skt.sptensor)

	mask = None
	if args.mask is not None:
		assert args.mask.ext == '.npz':
		mask = np.load(args.mask)['data']
		mask = None

	start_time = time.time()
	hptfsi = HPTFSI(n_modes=data.ndim,
					n_components=args.n_components,
					max_iter=args.max_iter,
					tol=args.tol,
					smoothness=args.smoothness,
					verbose=args.verbose,
					alpha=args.alpha,
					debug=args.debug)
	hptfsi.fit(data,side)
	end_time = time.time()
	print "Training time = %d"%(end_time-start_time)
	serialize(hptfsi,args.out)

if __name__ == '__main__':
	main()