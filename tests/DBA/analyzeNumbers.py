from DBFE import DBFE_SVM
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PseudoLDA import PLDA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename = 'data/mfeat.csv'
dataset = pd.read_csv(filename).as_matrix()
print("File read")


svc = SVC(C=2, kernel='poly', degree=5, decision_function_shape='ovr', gamma='auto', shrinking=True, tol=1e-3, cache_size=8000, class_weight=None, max_iter=1e6)
svc_cv = svc

X = dataset[:,:-1]
Y = dataset[:,-1]

scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

test = []
for i in range(10):
    test += range(i*200,i*200+50)
train = np.array(list(set(range(2000))-set(test)))
test = np.array(test)

for d in range(5,75,5):
    for method in ['LDA','PCA','DBA']:

        print(d,method)


        start = time.time()
        if method == 'DBA':
            model = DBFE_SVM(X[train],Y[train], C=2, kernel='poly', degree=5)
            train_X = model.transform(X[train].T,d).T
            test_X = model.transform(X[test].T,d).T
        else:
            if method == 'PCA':
                model = PCA(n_components=d).fit(X[train],Y[train])
            else:
                model = PLDA(n_components=d).fit(X[train],Y[train])
            train_X = model.transform(X[train])
            test_X = model.transform(X[test])

        end = time.time()
        print('\tTransform Data: {:5.2f} s'.format(end-start))
        start = time.time()

        try:
            cols = ['x{}'.format(i) for i in range(train_X.shape[1])]
            df = pd.DataFrame(np.hstack((train_X, np.atleast_2d(Y[train]).T)), columns= cols + ['y'])
            sns.pairplot(df, hue='y', vars=cols)
        except ValueError:
            pass

        plt.savefig('numbers_{}_{}D.png'.format(method, d))
        plt.close()

        test_model = svc_cv.fit(train_X, Y[train])

        end = time.time()
        print('\t      Fit Data: {:5.2f} s'.format(end-start))
        start = time.time()

        score = test_model.score(test_X, Y[test])

        end = time.time()
        print('\t    Score Data: {:5.2f} s'.format(end-start))

        print('D={} Error Rate: {:f}'.format(d, 1-score))
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