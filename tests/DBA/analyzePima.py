from DBFE import DBFE_SVM
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

filename = 'data/pima-indians-diabetes.csv'
dataset = pd.read_csv(filename).as_matrix()
n_folds = 12
tuned_parameters = {'C': np.linspace(2,60,10)}

svc = SVC(C=60, kernel='poly', degree=2, decision_function_shape='ovr', gamma='auto', shrinking=True, tol=1e-3, cache_size=8000, class_weight=None, max_iter=1e6)
svc_cv = GridSearchCV(svc, tuned_parameters, cv = n_folds, refit=True)

X = dataset[:,:-1]
Y = dataset[:,-1]

scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

k_fold = KFold(n_folds)

for d in range(1,9):
    for method in ['LDA', 'PCA', 'DBA']:
        scores = [None]*n_folds
        print(d,method)
        for k, (train, test) in enumerate(k_fold.split(X, Y)):

            start = time.time()
            if method == 'DBA':
                model = DBFE_SVM(X[train],Y[train])
                train_X = model.transform(X[train].T,d).T
                test_X = model.transform(X[test].T,d).T
            else:
                if method == 'PCA':
                    model = PCA(n_components=d).fit(X[train],Y[train])
                else:
                    model = LDA(n_components=d).fit(X[train],Y[train])
                train_X = model.transform(X[train])
                test_X = model.transform(X[test])

            end = time.time()
            print('\tTransform Data: {:5.2f} s'.format(end-start))
            start = time.time()

            ## Cross-validate for C
            test_model = svc_cv.fit(train_X, Y[train])

            end = time.time()
            print('\t      Fit Data: {:5.2f} s'.format(end-start))
            start = time.time()

            scores[k] = test_model.score(test_X, Y[test])

            end = time.time()
            print('\t    Score Data: {:5.2f} s'.format(end-start))

            print("\t[fold {0}] C: {1:5.2f} error: {2:5.2f}".format(k, test_model.estimator.get_params()['C'], 1-scores[k]))
            print("")
        print('D={} Error Rate: {:f}'.format(d, 1-np.average(scores)))
# start = time.time()
# dbfe = DBFE_SVM(X,Y)
# end = time.time()

# print('*'*80)
# print('Time: {} s'.format(end-start))
# print('Surface:')
# print(dbfe.S)
# print('*'*80)
# print('Features:')
# print(dbfe.features.shape)
# print('*'*80)