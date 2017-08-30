import numpy as np
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import GridSearchCV
import numdifftools as nd
import ngl
import time

class DBFE(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        ## First, scale the input space
        self.Xs = X#preprocessing.scale(X)

        start = time.time()
        ## Next, create the decision function and get the boundary indices
        self.BuildModel()
        end = time.time()

        print('\tModel built: {:5.2f} s'.format(end-start))

        ## Modification 1:
        ## Rather than compute only the nearest neighbors,
        ## build an ERG and interpolate each of those spanning edges.
        start = time.time()
        self.BuildGraph('beta skeleton', 50, 1.0)
        end = time.time()

        print('\tBuild graph: {:5.2f} s'.format(end-start))

        start = time.time()
        self.ExtractSurfacePoints()
        end = time.time()

        print('\tExtract surface samples: {:5.2f} s ({})'.format(end-start, self.S.shape[0]))

        start = time.time()
        self.FeatureExtraction()
        end = time.time()

        print('\tFeatures extracted: {:5.2f} s'.format(end-start))

    def FindZero(self, X0, X1, eps):
        stepSize = 1e-3
        alpha = 0.5
        y0 = self.decision(X0)
        y1 = self.decision(X1)

        ## We cannot interpolate here because the SVM is misclassifying
        ## one of these values, so we will return a None and have the calling
        ## environment handle this appropriately.
        if np.sign(y0) == np.sign(y1):
            return None

        S = alpha*X0 + (1-alpha)*X1
        h = self.decision(S)
        sgn = 0
        while abs(h) > eps:
            lastSign = sgn
            ## TODO: make this smarter
            if h > 0:
                if y1 > y0:
                ## move towards 0
                    alpha += stepSize
                    sgn = +1
                else:
                    alpha -= stepSize
                    sgn = -1
            else:
                if y1 > y0:
                ## move towards 1
                    alpha -= stepSize
                    sgn = -1
                else:
                    alpha += stepSize
                    sgn = +1
            if sgn == lastSign:
                stepSize *= 2
                stepSize = min(stepSize,1e-2)
            else:
                stepSize /= 2
                stepSize = max(stepSize,1e-10)

            S = alpha*X0 + (1.-alpha)*X1
            h = self.decision(S)
            if alpha < 0 or alpha > 1:
                print('\t\th: {}, alpha: {}, y0: {}, y1: {}'.format(h[0], alpha, stepSize, y0[0], y1[0]))
        return S

    def BuildModel(self):
        """
            By default, we will use a random decision identifier and everything
            will be assumed to be on the boundary. This should be reimplemented
            by each subclass
        """
        self.decision = lambda x: (1 - 2*np.random.randint(0,2))
        self.indices = list(range(len(self.Y)))

    def BuildGraph(self, graphType='beta skeleton', k=50, beta=1.0):
        self.graph = ngl.Graph(self.Xs, graphType, k, beta, None)

    def ExtractSurfacePoints(self, r=0.2):
        """
            This function will determine return the surface points
        """
        count  = int(min(r*len(self.Y), len(self.indices)))
        sortedIndices = []
        for i in self.indices:
            sortedIndices.append((self.decision(self.Xs[i]),i))

        sortedIndices = [ i for v,i in sorted(sortedIndices)]

        S = []
        for i in sortedIndices[:count]:
            neighborIndices = self.graph.Neighbors(i)
            for j in neighborIndices:
                ## Modification 2:
                ## Also, relax it to include edges where only one of the points
                ## is required to be on the decision boundary.
                # if j not in indices:
                #     continue
                if self.Y[i] != self.Y[j]:
                    ## TODO: interpolate
                    surfacePoint = self.FindZero(self.Xs[i], self.Xs[j], 1e-3)
                    if surfacePoint is not None:
                        S.append(surfacePoint)
        self.S = np.array(S)

    def FeatureExtraction(self):

        ## Compute the normals numerically
        N = nd.Gradient(self.decision)
        grads = np.zeros(self.S.shape)
        for i,s in enumerate(self.S):
            grads[i] = N(s)

        ## Normalize the normals
        gradMags = np.linalg.norm(grads,axis=1).reshape(-1,1)
        Ni = grads/gradMags

        ## Modification 3:
        ## Partition the data at this point
        ## TODO: define criteria for separating the data

        ## Construct the Decision Boundary Scatter Matrix (DBSM)
        DBSM = np.zeros((Ni.shape[1],Ni.shape[1]))
        for ni in Ni:
            DBSM += np.outer(ni,ni)

        DBSM /= float(len(Ni))
        values,vectors = np.linalg.eig(DBSM)

        features = []
        for value,vector in zip(values,vectors):
            if value != 0:
                features.append(vector)

        self.features = np.array(features)

    def transform(self, X, maxDimensionality=2):
        maxDimensionality = min(self.features.shape[0],maxDimensionality)
        return self.features[:maxDimensionality].dot(X)

class DBFE_LVQ(DBFE):
    pass

class DBFE_KNN(DBFE):
    pass

class DBFE_SVM(DBFE):
    def BuildModel(self, n_folds=12, tuned_parameters = {'C': np.linspace(2,60,10)}, kernel='poly', degree=2):
        ## Train a support vector machine
        svc = SVC(C=10000, kernel=kernel, degree=degree, gamma='auto', shrinking=True, tol=1e-3, cache_size=8000, class_weight=None, max_iter=1e6)
        self.model = GridSearchCV(svc, tuned_parameters, cv = n_folds, refit=True).estimator
        # self.model = NuSVC(nu=0.5, kernel='rbf', gamma='auto', shrinking=True, tol=0.001, cache_size=8000, class_weight=None, max_iter=-1)
        _ = self.model.fit(self.Xs, self.Y)
        self.decision = lambda x: self.model.decision_function(x.reshape(1, -1))

        ## Get the support vectors
        self.indices = self.model.support_