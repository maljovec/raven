"""
    Decision Boundary Feature Extraction module
"""
# pylint: disable=C0103
# pylint: disable=R0902

import numpy as np
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn import cluster
import numdifftools as nd
import ngl
import time
from scipy.optimize import bisect, fsolve
import sys
from scipy.spatial import distance

from sklearn.neighbors import NearestNeighbors

def FindZero(X0, X1, func, eps):
    """
        Find the zero existing between two input values given a function
        func and a tolerance eps
    """
    # stepSize = 1e-3
    # alpha = 0.5
    y0 = func(X0)
    y1 = func(X1)
    # minStepSize = 1e-16
    # maxCount = 10000

    def f(alpha):
        """
            Parametric function to allow us to vary between X0 and X1
        """
        return func(alpha*X0 + (1-alpha)*X1)

    ## We cannot interpolate here because the SVM is misclassifying
    ## one of these values, so we will return a None and have the calling
    ## environment handle this appropriately.
    if np.sign(y0) == np.sign(y1):
        return None

    alpha = bisect(f, 0, 1, xtol=eps)
    # alpha = fsolve(f, 0.5)
    if alpha >= 0 and alpha <= 1:
        return alpha*X0 + (1-alpha)*X1
    else:
        return None

class DBFE(object):
    """
        Decision Boundary Feature Extraction Class
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.classes = sorted(np.unique(self.Y))
        ## Unfortunately, sklearn handles multiclass decision functions a bit
        ## differently than the binary classification case, so let's store this
        ## as a boolean, so we can easily identify in the algorithm where the
        ## forks occurs
        self.isMultiClass = len(self.classes) > 2
        if self.isMultiClass:
            self.decisionOffset = len(self.classes)-1
        else:
            self.decisionOffset = 0

        ## First, scale the input space
        self.Xs = X#preprocessing.scale(X)


        print('\tModel built:', end='')
        sys.stdout.flush()

        start = time.time()
        ## Next, create the decision function and get the boundary indices
        self.BuildModel()
        end = time.time()

        print(' {:5.2f} s'.format(end-start))
        print('\tBuild graph:', end='')
        sys.stdout.flush()

        ## Modification 1:
        ## Rather than compute only the nearest neighbors,
        ## build an ERG and interpolate each of those spanning edges.
        start = time.time()
        self.BuildGraph('beta skeleton', 50, 1.0)
        end = time.time()

        print(' {:5.2f} s'.format(end-start))
        print('\tExtract surface samples:', end='')
        sys.stdout.flush()

        start = time.time()
        self.ExtractSurfacePoints()
        end = time.time()

        print(' {:5.2f} s ({})'.format(end-start, self.S.shape[0]))
        print('\tFeatures extracted: ', end='')
        sys.stdout.flush()

        start = time.time()
        self.FeatureExtraction()
        end = time.time()

        print(' {:5.2f} s'.format(end-start))
        sys.stdout.flush()

    def BuildModel(self):
        """
            By default, we will use a random decision identifier and everything
            will be assumed to be on the boundary. This should be reimplemented
            by each subclass
        """
        self.decision = lambda x: (1 - 2*np.random.randint(0, 2))
        self.indices = list(range(len(self.Y)))

    def BuildGraph(self, graphType='beta skeleton', k=10, beta=1.0):
        """
            Build a graph on the data of this object and store it internally
        """
        nn = NearestNeighbors(k, n_jobs=-1).fit(self.Xs)
        pairs = []
        for i in range(len(self.Xs)):
            Xs = self.Xs[i].reshape(1, -1)
            neighborIndices = nn.kneighbors(Xs, return_distance=False).flatten()
            for j in neighborIndices:
                if i != j:
                    pairs.append((i, j))

        ## As seen here:
        ##  https://tinyurl.com/angxm5
        seen = set()
        pairs = [x for x in pairs if not (x in seen or x[::-1] in seen or seen.add(x))]

        edgesToPrune = []
        for edge in pairs:
            edgesToPrune.append(int(edge[0]))
            edgesToPrune.append(int(edge[1]))

        a = ngl.ngl.vectorInt([1, 2])
        ngl.ngl.vectorInt(edgesToPrune)
        self.graph = ngl.Graph(self.Xs, graphType, k, beta, edgesToPrune)

    def ExtractSurfacePoints(self, r=0.2):
        """
            This function will determine and return the surface points
        """
        count = int(min(r*len(self.Y), len(self.indices)))
        sortedIndices = []

        for i in self.indices:
            if self.isMultiClass:
                proximitiesToBoundaries = np.abs(self.decision(self.Xs[i]) - self.decisionOffset)
                ## Figure out which class this support vector is determining
                decisionIdx = np.argmin(proximitiesToBoundaries)
                sortedIndices.append((proximitiesToBoundaries[:, decisionIdx], i, decisionIdx))
            else:
                ## We don't need a decisionIdx here or a list, but setting them
                ## makes it look like the multiclass case
                proximitiesToBoundaries = [np.abs(self.decision(self.Xs[i]))]
                decisionIdx = 0
                sortedIndices.append((proximitiesToBoundaries, i, decisionIdx))

        ## Start with the indices closest to the boundaries, note we don't need
        ## the results of the decision function after this, only the sorted
        ## order of indices, and the decisionIdx (for the multiclass case)
        sortedIndices = [(i, d) for (v, i, d) in sorted(sortedIndices)]

        S = []
        decisionIndices = []
        counter = 0
        for i, decisionIdx in sortedIndices[:count]:
            start = time.time()
            neighborIndices = self.graph.Neighbors(i)
            # neighborIndices = self.graph.kneighbors(self.Xs[i], return_distance=False)
            for j in neighborIndices:
                ## Modification 2:
                ## Also, relax it to include edges where only one of the points
                ## is required to be on the decision boundary.
                # if j not in indices:
                #     continue
                if self.Y[i] != self.Y[j]:
                    ## Now interpolate along the edge specified by the points
                    ## X[i] and X[j] to find the zero crossing of the decision
                    ## function, for multiclass we have to appropriately index
                    ## the correct decision function
                    if self.isMultiClass:
                        offset = self.decisionOffset
                        decisionFunction = lambda x: self.decision(x)[:, decisionIdx] - offset
                    else:
                        decisionFunction = lambda x: self.decision(x)
                    surfacePoint = FindZero(self.Xs[i], self.Xs[j], decisionFunction, 1e-3)
                    if surfacePoint is not None:
                        S.append(surfacePoint)
                        ## For multiclass
                        decisionIndices.append(decisionIdx)
            counter += 1
            end = time.time()
        self.S = np.array(S)
        ## For multiclass
        self.decisionIndices = decisionIndices

    def ComputeNumericGradients(self):
        grads = np.zeros(self.S.shape)

        for i, (s, decisionIdx) in enumerate(zip(self.S, self.decisionIndices)):
            if self.isMultiClass:
                offset = len(self.classes)-1
                decisionFunction = lambda x: self.decision(x)[:, decisionIdx] - offset
            else:
                decisionFunction = lambda x: self.decision(x)
            N = nd.Gradient(decisionFunction)
            grads[i] = N(s)
        return grads

    def ComputeAnalyticGradients(self):
        grads = np.zeros(self.S.shape)

        ## degree p polynomial
        p = self.model.get_params()['degree']
        kernel = self.model.get_params()['kernel']

        ## k classes
        k = len(self.model.n_support_)

        ## Find the two closest classes for each support vector to determine
        ## which boundary it represents in a one-vs-one setting
        boundaries = np.zeros((len(self.model.support_vectors_), 2), dtype=int)
        for j, xj in enumerate(self.model.support_vectors_):

            ## Find out the two closest classes
            if k > 2:
                diffs = np.abs(self.model.decision_function(xj.reshape(1, -1)) - (k-1))
                a, b = np.argsort(diffs)[0, :2]
                if a > b:
                    a, b = b, a
            else:
                a, b = 0, 1
            boundaries[j] = [a, b]

        for i, s in enumerate(self.S):
            start = time.time()
            if k > 2:
                diffs = np.abs(self.model.decision_function(s.reshape(1, -1)) - (k-1))
                a, b = np.argsort(diffs)[0, :2]
                if a > b:
                    a, b = b, a
            else:
                a, b = 0, 1

            ovo = (a, b)
            decisionIdx = b-1

            if kernel == 'rbf':
                summand = np.zeros(len(s))
                for j, xj in enumerate(self.model.support_vectors_):
                    if ovo != tuple(boundaries[j]):
                        continue

                    coefficient = self.model.dual_coef_[decisionIdx, j] * (np.exp(-np.linalg.norm(s-xj)))
                    summand += coefficient * (s-xj)

                grads[i] = summand
            elif kernel == 'poly':
                summand = np.zeros(len(s))
                for j, xj in enumerate(self.model.support_vectors_):
                    if ovo != tuple(boundaries[j]):
                        continue

                    numerator = self.model.dual_coef_[decisionIdx, j] * (np.dot(s, xj))**p
                    denominator = np.dot(s, xj)
                    coefficient = numerator/denominator
                    summand += coefficient * xj

                grads[i] = summand
            else:
                print('Unsupported kernel type: {}'.format(kernel))

            end = time.time()

        return grads

    def FeatureExtraction(self):
        # start = time.time()
        # grads = self.ComputeNumericGradients()
        # np.savetxt('numericGrads.csv', grads, delimiter=',')
        # end = time.time()
        # print('\t\tNumeric Gradients: {} s'.format(end-start))

        # start = time.time()
        grads = self.ComputeAnalyticGradients()
        np.savetxt('analyticGrads.csv', grads, delimiter=',')
        # end = time.time()
        # print('\t\tAnalytic Gradients: {} s'.format(end-start))

        ## Normalize the normals
        gradMags = np.linalg.norm(grads, axis=1).reshape(-1, 1)

        ## Remove degenerate normals (where did these come from? Could be an
        ## important question)
        idxsToKeep = np.nonzero(gradMags)[0]

        Ni = grads[idxsToKeep]/gradMags[idxsToKeep]

        ## Modification 3:
        ## Partition the data at this point
        ## TODO: define criteria for separating the data

        # start = time.time()
        ## Construct the Decision Boundary Scatter Matrix (DBSM)
        DBSM = np.zeros((Ni.shape[1], Ni.shape[1]))
        for ni in Ni:
            DBSM += np.outer(ni, ni)
        DBSM /= float(len(Ni))

        # end = time.time()
        # print('\t\tDBSM: {} s'.format(end-start))

        # start = time.time()
        values, vectors = np.linalg.eig(DBSM)
        # end = time.time()
        # print('\t\tEigendecomposition: {} s'.format(end-start))

        features = []
        for value, vector in zip(values, vectors):
            if value != 0:
                features.append(vector)

        self.features = np.array(features)

    def transform(self, X, maxDimensionality=2):
        maxDimensionality = min(self.features.shape[0], maxDimensionality)
        return self.features[:maxDimensionality].dot(X)

    def writeTransform(self, baseFileName=''):
        pass

class DBFE_LVQ(DBFE):
    pass

class DBFE_KNN(DBFE):
    pass

class DBFE_SVM(DBFE):
    def __init__(self, X, Y, C=1, n_folds=0, tuned_parameters={}, kernel='poly', degree=2):
        self.C = 1
        self.degree = degree
        self.kernel = kernel
        self.cv_parameters = tuned_parameters
        self.n_folds = n_folds

        super(DBFE_SVM, self).__init__(X, Y)


    def BuildModel(self):
        ## Train a support vector machine
        svc = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=1,
                  shrinking=True, tol=1e-3, cache_size=8000, class_weight=None,
                  max_iter=1e6)
        if self.n_folds > 0 and len(self.cv_parameters.keys()) > 0:
            cv = self.n_folds
            self.model = GridSearchCV(svc, self.cv_parameters, cv=cv, refit=True).estimator
        else:
            self.model = svc
        ## self.model = NuSVC(nu=0.5, kernel='rbf', gamma='auto', shrinking=True,
        ## tol=0.001, cache_size=8000, class_weight=None, max_iter=-1)
        _ = self.model.fit(self.Xs, self.Y)
        self.decision = lambda x: self.model.decision_function(x.reshape(1, -1))

        ## Get the support vectors
        self.indices = self.model.support_

class DBFE_Clustered_SVM(DBFE_SVM):
    def FeatureExtraction(self):
        grads = self.ComputeAnalyticGradients()
        np.savetxt('analyticGrads.csv', grads, delimiter=',')

        ## Normalize the normals
        gradMags = np.linalg.norm(grads, axis=1).reshape(-1, 1)

        ## Remove degenerate normals (where did these come from? Could be an
        ## important question)
        idxsToKeep = np.nonzero(gradMags)[0]
        self.filteredIndices = idxsToKeep

        Ni = grads[idxsToKeep]/gradMags[idxsToKeep]

        ## Modification 3:
        ## Partition the data at this point
        ## TODO: define criteria for separating the data
        X = np.hstack((self.S[idxsToKeep], Ni))

        if True:
            # X = np.loadtxt('simple_bwr_ls.csv', delimiter=',')
            connectivity = np.loadtxt('connectivity.csv', delimiter=',')
            self.clusterIds = np.loadtxt('spectralClustering5.csv', delimiter=',')
            beta = 1
            similarity = np.exp(-beta * connectivity / connectivity.std())
            print(similarity.shape)
        else:
            connectivity = np.zeros((X.shape[0], X.shape[0]))

            print('Bulding connectivity graph: ', end='')
            start = time.time()
            ## Allows downweighting of the euclidean distance vs cosine distance
            alpha = 0.5
            d = self.S.shape[1]
            for i,xi in enumerate(X):
                for j,xj in enumerate(X):
                    if i == j:
                        continue

                    euclidean = np.linalg.norm(xi[:d]-xj[:d])
                    cosine = distance.cosine(xi[d:], xj[d:])
                    connectivity[j, i] = connectivity[i, j] = alpha*euclidean + (1-alpha)*cosine

            end = time.time()
            print('{} s'.format(end-start))
            # print('connecitivity shape: {}'.format(connectivity.shape))
            np.savetxt('simple_bwr_ls.csv', X, delimiter=',')
            np.savetxt('simple_connectivity.csv', connectivity, delimiter=',')

            print('Clustering: ', end='')
            start = time.time()
            clustering = cluster.SpectralClustering(n_clusters=5, affinity='precomputed')
            self.clusterIds = clustering.fit_predict(similarity)
            np.savetxt('spectralClustering5.csv', self.clusterIds, delimiter=',')
            end = time.time()
            print('{} s'.format(end-start))
        
        labels = np.unique(self.clusterIds)
        self.features = {}
        for label in labels:
            idxs = np.where(self.clusterIds == label)

            Nl = Ni[idxs]

            # start = time.time()
            ## Construct the Decision Boundary Scatter Matrix (DBSM)
            DBSM = np.zeros((Nl.shape[1], Nl.shape[1]))
            for ni in Nl:
                DBSM += np.outer(ni, ni)
            DBSM /= float(len(Nl))

            # end = time.time()
            # print('\t\tDBSM: {} s'.format(end-start))

            # start = time.time()
            values, vectors = np.linalg.eig(DBSM)
            # end = time.time()
            # print('\t\tEigendecomposition: {} s'.format(end-start))

            features = []
            for value, vector in zip(values, vectors):
                if value != 0:
                    features.append(vector)

            self.features[label] = np.array(features)

    def transform(self, X, maxDimensionality=2):

        embedding = []
        
        Xs = self.S[self.filteredIndices]
        for x in X:
            deltas = Xs - x
            dist_2 = np.einsum('ij,ij->i', deltas, deltas)
            minIdx = np.argmin(dist_2)
            predictedLabel = self.clusterIds[minIdx]

            maxDimensionality = min(self.features[predictedLabel].shape[0], maxDimensionality)
            embedding.append(self.features[predictedLabel][:maxDimensionality].dot(x))
        return np.array(embedding)