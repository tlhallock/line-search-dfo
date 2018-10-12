import os
import re

lst = []

for d in os.listdir('images'):
    if not os.path.isdir(os.path.join('images', d)):
        continue
    num_evaluations = -1
    with open(os.path.join('images', d, 'log.txt'), 'r') as logfile:
        for line in logfile:
            m = re.search('total number of evaluations = ([0-9]*)', line)
            if m:
                num_evaluations = max(num_evaluations, int(m.group(1)))
    lst.append((d, num_evaluations))

lst = sorted(lst, key=lambda x: x[1])

with open('images/evaluations.txt', 'w') as output:
    for l in lst:
        output.write(str(l[0]).rjust(20, ' ') + ': ' + str(l[1]) + '\n')
        print(l)

