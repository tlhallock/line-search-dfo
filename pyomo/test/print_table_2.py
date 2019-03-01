

import os
import json
import traceback


base = '/work/research/line-search-dfo/pyomo/test/images'
for directory_name in os.listdir(base):
	directory = os.path.join(base, directory_name)
	if not os.path.isdir(directory):
		continue
	log_file = os.path.join(directory, 'log.json')
	if not os.path.exists(log_file):
		continue
	try:
		with open(log_file, 'r') as infile:
			logs = json.load(infile)
	except:
		traceback.print_exc()
		continue

	last_log = logs['iterations'][len(logs['iterations'])-1]
	print(log_file)
	print(last_log['number-of-evaluations'])
	print(last_log['center'])


converged
images/hs_25_polyhedral/log.json:  "number-of-evaluations": 67,


images/hs_76_ellipse/log.json:  "number-of-evaluations": 530,,
images/hs_76_ellipse/log.json:  "new-function-value": -4.6838441301764195,


images/hs_76_polyhedral/log.json:  "number-of-evaluations": 426,,
    -4.681817540552573,




