import numpy as np

fac = np.loadtxt('mfeat-fac')
fou = np.loadtxt('mfeat-fou')
kar = np.loadtxt('mfeat-kar')
mor = np.loadtxt('mfeat-mor')
pix = np.loadtxt('mfeat-pix')
zer = np.loadtxt('mfeat-zer')
target = np.zeros((2000,1))
for i in range(10):
	target[200*i:200*i+200] = i

fullData = np.hstack((fac,fou,kar,mor,pix,zer,target))
hdr = ''
sep = ''
for i in range(fullData.shape[1]-1):
	hdr += sep + 'x{}'.format(i)
	sep = ','

hdr += sep + 'number'

np.savetxt('mfeat.csv', fullData, delimiter=',',header=hdr)
