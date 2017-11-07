import sys

import matplotlib.pyplot as plt
import numpy as np

fin = open(sys.argv[1])

maxDim = 1000
lda = np.array([None]*maxDim)
dba = np.array([None]*maxDim)
pca = np.array([None]*maxDim)

mode = None
for line in fin:
    if 'PCA' in line:
        mode = 'PCA'
        continue
    elif 'LDA' in line:
        mode = 'LDA'
        continue
    elif 'DBA' in line:
        mode = 'DBA'
        continue

    if line.startswith('D'):
        tokens = line[2:].split(' ', 1)
        dim = int(tokens[0])
        error = float(tokens[1].split(':')[1])
        print(mode, dim, error)
        if mode == 'PCA':
            pca[dim-1] = error
        elif mode == 'LDA':
            lda[dim-1] = error
        else:
            dba[dim-1] = error

plt.figure()

idxs = np.array(range(0,maxDim,1))

ldaIdxs = idxs[ lda != np.array(None)]
p1, = plt.plot(ldaIdxs+1, lda[ldaIdxs],c='#e41a1c', label='LDA', linestyle='solid')

pcaIdxs = idxs[ pca != np.array(None)]
p2, = plt.plot(pcaIdxs+1, pca[pcaIdxs],c='#377eb8', label='PCA', linestyle='dashed')

dbaIdxs = idxs[ dba != np.array(None)]
p3, = plt.plot(dbaIdxs+1, dba[dbaIdxs],c='#4daf4a', label='DBA', linestyle='dotted')

plt.legend(handles=[p1,p2,p3])

plt.show()
