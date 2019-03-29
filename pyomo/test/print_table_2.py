

import os
import json
import traceback
import re

# np.set_printoptions(linewidth=255)

print('algorithm', 'problem', 'message', 'n iterations', 'n evaluations', 'found min', 'true min', 'found minimizer', 'true minimizer')

base = '/work/research/line-search-dfo/pyomo/images'

missing = []
done = []
for directory_name in sorted(os.listdir(base)):
	directory = os.path.join(base, directory_name)
	if not os.path.isdir(directory):
		continue
	result_file = os.path.join(directory, 'result.json')
	if not os.path.exists(result_file):
		missing.append(directory)
		continue
	try:
		with open(result_file, 'r') as infile:
			result = json.load(infile)
	except:
		traceback.print_exc()
		continue

	done.append(" & ".join([
		str.center(str(re.sub('hs_[0-9]*_', '', directory_name).replace('_', ' ')), 25),
		str.center(str(result['problem']['problem_number']), 5),
		str.center(str(result['problem']['n']), 3),
		str.center(str(result['result']['message']), 10),
		str.center(str(result['result']['niter']), 5),
		str.center(str(result['result']['neval']), 5),
		str.center(str("%.3f" % result['result']['minimum']), 10),
		str.center(str("%.3f" % result['problem']['minimum']), 10),
		str.center(str("[" + ",".join(["%.2f" % xi for xi in result['result']['minimizer']]) + "]"), 5),
		str.center(str("[" + ",".join(["%.2f" % xi for xi in result['problem']['minimizer']]) + "]"), 5),
	]) + " \\\\")

done.sort(key=lambda x: (x[1], x[0]))

for text in done:
	print(text)

for m in missing:
	print('kde-open ' + str(m))

