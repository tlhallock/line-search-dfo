
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

filePattern = re.compile('([^_]*)_([0-9]*).p');

algos = {}

# point this at the output directory
for root, subFolders, files in os.walk('./runtimes/'):
	for file in files:
		m = filePattern.match(file)
		if not m:
			continue
		algo = m.group(1)
		nprob = m.group(2)

		with open(os.path.join(root, file), "rb") as input:
			if nprob not in algos:
				algos[nprob] = {}
			data = pickle.load(input)
			algos[nprob][algo] = data

probNdx = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 14, 15, 16, 17, -1, 19, 20, 21]
algoIdx = {
	'mine': 1,
	'pyOpt': 2
}

with open('../../octave/assign_to_matlab.m', 'w') as log:
	log.write('h = zeros(0,0,0);')

	for nprob in algos.keys():
		print(nprob)
		for algo in algos[nprob].keys():
			nfev = algos[nprob][algo]['nfev']
			fvals = algos[nprob][algo]['fvals']
			print('\t' + str(algo) + " " + str(nfev))
			for i in range(len(fvals)):
				log.write('h(' + str(i + 1) + ', ' + str(probNdx[int(nprob)]) + ', ' + str(algoIdx[algo]) + ') = ' + str(fvals[i][0]) + ';\n')
			xs = np.asarray(range(nfev))
			ys = np.zeros(nfev)
			for i in range(nfev):
				ys[i] = fvals[i][0]
			plt.plot(xs, ys, label=algo)

		plt.legend(loc='upper right')
		plt.savefig('plots/' + nprob + '_performance.png')
		plt.close()
