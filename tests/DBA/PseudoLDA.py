import numpy as np
import scipy as sp

class PLDA(object):
	def __init__(self, n_components=None):
		self.n_components = n_components
		self.T = None

	def fit(self, X, Y):
		# print('fit')
		C = np.unique(Y)

		d = X.shape[1]
		Sb = np.zeros((d, d))
		Si = np.zeros((d, d))

		n = len(Y)
		m = np.average(X,axis=0)
		for i in C:
			subset = np.where(Y == i)[0]
			ni = len(subset)
			mi = np.average(X[subset], axis=0)
			delta = mi - m
			# Sb = ni * np.cov(X[subset], rowvar=False)
			Sb += ni * np.outer(delta, delta)
			for j in subset:
				delta = X[j] - mi
				Si += np.outer(delta, delta)

		Sb /= float(n)
		Si /= float(n)
		A = np.dot(np.linalg.pinv(Si), Sb)
		eigvalues, eigvectors = sp.linalg.eig(A)
		# eigvalues, eigvectors = sp.linalg.eigh(Sb, Si)
		indices = [i[0] for i in sorted(enumerate(eigvalues), key=lambda x: -x[1])]

		if self.n_components is None:
			self.n_components = C-1

		self.transformMatrix = eigvectors[indices[:self.n_components]]

		# print(eigvalues[indices])
		# print(eigvectors[indices])
		return self

	def transform(self, X):
		return np.dot(X, self.transformMatrix.T)