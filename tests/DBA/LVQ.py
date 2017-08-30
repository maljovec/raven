import scipy

def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

class LVQ(object):
	def __init_(self, nCodeBooks, epochs=10, learningRate=0.3):
		self.n_codebooks = nCodeBooks
		self.lrate = learningRate
		self.epochs = epochs

	def get_best_matching_unit(self, test_row):
		distances = list()
		for codebook in self.codebooks:
			dist = scipy.spatial.distance.euclidean(codebook,test_row)
			distances.append((codebook, dist))
		distances.sort(key=lambda tup: tup[1])
		return distances[0][0]


	def train(self, X, Y):
		self.Xs = preprocessing.scale(X)
		self.Y = Y

		self.codebooks = [random_codebook(X) for i in range(self.n_codebooks)]
		for epoch in range(self.epochs):
			rate = self.lrate * (1.0-(epoch/float(self.epochs)))
			sum_error = 0.0
			for row in train:
				bmu = get_best_matching_unit(codebooks, row)
				for i in range(len(row)-1):
					error = row[i] - bmu[i]
					sum_error += error**2
					if bmu[-1] == row[-1]:
						bmu[i] += rate * error
					else:
						bmu[i] -= rate * error
			print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
		self.codebooks


	def predict(self, X):
		bmu = get_best_matching_unit(codebooks, test_row)
		return bmu[-1]
