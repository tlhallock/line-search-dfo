
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

filePattern = re.compile('([^_]*)_([0-9]*).p');

algos = {}

# point this at the output directory
for root, subFolders, files in os.walk('/work/research/line-search-dfo/python/test/runtimes/'):
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


for nprob in algos.keys():
	# maxIterations = 0
	# for algo in algos[nprob].keys():
	# 	nfev = algos[nprob][algo]
	# 	if maxIterations < nfev:
	# 		maxIterations = 0
	# numAlgorithms = len(algos[nprob][algo])
	# xs = np.asarray(range(maxIterations))
	# ys = np.zeros(numAlgorithms, maxIterations)
	for algo in algos[nprob].keys():
		nfev = algos[nprob][algo]['nfev']
		fvals = algos[nprob][algo]['fvals']
		xs = np.asarray(range(nfev))
		ys = np.zeros(nfev)
		for i in range(nfev):
			ys[i] = fvals[i][0]
		plt.plot(xs, ys, label=algo)

	plt.legend(loc='upper right')
	plt.savefig('plots/' + nprob + '_performance.png')
	plt.close()
