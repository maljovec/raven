import matplotlib.pyplot as plt

fin = open('pima.log')

lda = [None]*8
dba = [None]*8
pca = [None]*8

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
		if mode == 'PCA':
			pca[dim-1] = error
		elif mode == 'LDA':
			lda[dim-1] = error
		else:
			dba[dim-1] = error

plt.figure()

p1, = plt.plot(range(1,9),lda,c='#e41a1c', label='LDA')
p2, = plt.plot(range(1,9),pca,c='#377eb8', label='PCA')
p3, = plt.plot(range(1,9),dba,c='#4daf4a', label='DBA')
plt.legend(handles=[p1,p2,p3])

plt.show()