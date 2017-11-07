from .ngl import nglGraph, vectorInt, vectorDouble

class Graph(nglGraph):
	def __init__(self, X, graph, maxN, beta, edges):
		if edges is None:
			edges = vectorInt()
		else:
			edges = vectorInt(edges)

		super(Graph, self).__init__(vectorDouble(X.flatten()), X.shape[0], X.shape[1], graph, maxN, beta, edges)

	def Neighbors(self, idx):
		return list(super(Graph, self).Neighbors(int(idx)))