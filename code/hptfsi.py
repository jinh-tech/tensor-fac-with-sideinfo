import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

from argparse import ArgumentParser
from utils import *

class HPTFSI():
	
	def __init__(self, n_modes=3, n_components=100,  max_iter=200, tol=0.0001,smoothness=100, verbose=True, alpha=0.1, alpha_p=0.1, c=1., d=1., beta_p=1.):
				
		self.n_modes = n_modes+1
		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.smoothness = smoothness
		self.verbose = verbose

		self.alpha = alpha
		self.alpha_p = alpha_p # shape hyperparameter
		self.beta_p = beta_p
		self.c = c
		self.d = d

		self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
		self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates
		self.beta_gamma = np.empty(self.n_modes-1,dtype=object)
		self.beta_delta = np.empty(self.n_modes-1,dtype=object)
		
		self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
		self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations
		self.beta_E = np.empty(self.n_modes-1,dtype=object)

		self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)

	def _check_component(self, m):
		
		assert np.isfinite(self.E_DK_M[m]).all()
		assert np.isfinite(self.G_DK_M[m]).all()
		assert np.isfinite(self.gamma_DK_M[m]).all()
		assert np.isfinite(self.delta_DK_M[m]).all()

	def _init_component(self,m,dim,K,s):
		
		gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
		delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
		
		self.gamma_DK_M[m] = gamma_DK
		self.delta_DK_M[m] = delta_DK
		self.E_DK_M[m] = gamma_DK / delta_DK
		self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0,keepdims=True)
		self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))

	def _init_all_components(self,mode_dims):

		self.mode_dims = mode_dims
		for m, D in enumerate(mode_dims):
			self._init_component(m, D,self.n_components,self.smoothness)
		m = len(self.mode_dims)
		self._init_component(m,1,self.n_components,1.)
		
		for m in range(0,self.n_modes-1):
			self.beta_gamma[m] = np.ones((mode_dims[m],1),dtype=np.float)*(self.alpha*self.n_components + self.alpha_p)
			self.beta_delta[m] = rn.uniform(size=(mode_dims[m],1)) *self.beta_p
			self.beta_E[m] = self.beta_gamma[m]/self.beta_delta[m]

	def _update_cache(self, m):

		gamma_DK = self.gamma_DK_M[m]
		delta_DK = self.delta_DK_M[m]
		self.E_DK_M[m] = gamma_DK / delta_DK
		self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0,keepdims=True)
		self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

	def _reconstruct_nz_data(self, subs_I_M, G_DK_M):
		
		nz_recon_IK = np.ones((subs_I_M[0].size, self.n_components))	# I = subs_I_M[0].size, K = self.n_components
		for m in xrange(self.n_modes-1):
			nz_recon_IK *= G_DK_M[m][subs_I_M[m], :]
		return nz_recon_IK.sum(axis=1)

	def _reconstruct_nz_side(self, subs_I_M, G_DK_M):

		nz_recon_IK = (np.ones((subs_I_M[0].size, self.n_components)) * G_DK_M[0][subs_I_M[0], :]* G_DK_M[1][subs_I_M[1], :]) * G_DK_M[self.n_modes-1]	# I = subs_I_M[0].size, K = self.n_components
		return nz_recon_IK.sum(axis=1)

	def _update_gamma(self, m, data, side):
		
		tmp = data.vals / self._reconstruct_nz_data(data.subs,self.G_DK_M)
		uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M[0:self.n_modes-1])
		self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK
		if m==0 or m==1:
			tmp = side.vals / self._reconstruct_nz_side(side.subs,self.G_DK_M)
			uttkrp_DK = sp_uttkrp(tmp, side.subs, m, self.G_DK_M[0:2]) * self.G_DK_M[m] * self.G_DK_M[self.n_modes-1]
			self.gamma_DK_M[m][:,:] += uttkrp_DK

	def _update_delta(self,m):
		
		sumE_MK = self.sumE_MK
		sumE_MK[m, :] = 1.
		uttrkp_DK = sumE_MK.prod(axis=0,keepdims=True)
		self.delta_DK_M[m][:, :] = uttrkp_DK
		self.delta_DK_M[m][:, :] += self.beta_E[m]
		if m==0 or m==1:
			sumE_MK[self.n_modes-2,:] = 1.
			self.delta_DK[m][:,:] += sumE_MK.prod(axis=0,keepdims=True)*self.E_DK_M[self.n_modes-1]

	def _update_beta(self,m):

		self.beta_delta[m][:] = self.beta_p + self.E_DK_M[m].sum(axis=1,keepdims=True)

	def _update_lamda(self,side):

		tmp = (side.vals/self._reconstruct_nz_side(side.subs,self.G_DK_M))*(self.G_DK_M[0][side.subs[0],:]*self.G_DK_M[1][side.subs[1],:])
		self.gamma_DK_M[self.n_modes-1][:] = self.c + tmp.sum(axis=0,keepdims=True)*self.G_DK_M[self.n_modes-1]
		self.delta_DK_M[self.n_modes-1][:] = self.d + self.sumE_MK[0]*self.sumE_MK[1]

	def _mae_nz(self,data):

		return (data.vals - parafac(self.G_DK_M[0:self.n_modes-1])[data.subs]).sum()

	def _update(self, data, side, orig_data=None, mask_no=None):

		curr_elbo = -np.inf
		for itn in xrange(self.max_iter):
			s = time.time()
			for m in self.n_modes-1:
				self._update_gamma(m, data, side)
				self._update_delta(m)
				self._update_cache(m)
				self._update_beta(m)
				self._check_component(m)
			self._update_lamda(side)
			bound = self.mae_nz(data)
			delta = (curr_elbo - bound) if itn > 0 else np.nan
			e = time.time() - s
			if self.verbose:
				print 'ITERATION %d:    Time: %f   Objective: %.2f    Change: %.5f'% (itn, e, bound, delta)

			# if delta < self.tol:
			#     break

	def fit(self, data,side ,test_times=None,orig_data=None,mask_no=None,bool_test=False):
		
		self.bool_test = bool_test
		if self.bool_test == True:
			#TOOD
			print "LATER"
		
		self._init_all_components(data.shape)
		self._update(data,side)  orig_data,mask_no

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
	p.add_argument('-ap', '--alpha_p', type=float, default=0.1)
	p.add_argument('-c', '--c_val', type=float, default=1.)
	p.add_argument('-d', '--d_val', type=float, default=1.)
	p.add_argument('-bp', '--beta_p', type=float, default=1.)
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

	if args.mask is not None:
		assert args.mask.ext == '.npz':
		mask = np.load(args.mask)['data']

	start_time = time.time()
	hptfsi = HPTFSI(n_modes=data.ndim,
					n_components=args.n_components,
					max_iter=args.max_iter,
					tol=args.tol,
					smoothness=args.smoothness,
					verbose=args.verbose,
					alpha=args.alpha,
					alpha_p=args.alpha_p,
					c=args.c_val,
					d=args.d_val,
					beta_p=args.beta_p)

	hptfsi.fit(data,side)
	end_time = time.time()
	print "Training time = %d"%(end_time-start_time)
	serialize_hptfsi(hptfsi,args.out)

if __name__ == '__main__':
	main()