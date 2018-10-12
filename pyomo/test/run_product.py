




mat = [['model.q[2]', '-model.q[1]'],['-model.q[1]', 'model.q[0]']]
a = [['si[0]'], ['si[1]']]
b = [['si[0]', 'si[1]']]

def mult(a, b):
	m = len(a)
	inner = len(a[0])
	n = len(b[0])
	ret = [['' for i in range(n)] for j in range(m)]

	for i in range(m):
		for j in range(n):
			for k in range(inner):
				ret[i][j] += '(({}) * ({})) + '.format(
					a[i][k],
					b[k][j]
				)


	return ret

print(
	mult(b, mult(mat, a))
)


# '((si[0]) * (((model.q[2]) * (si[0])) + ((-model.q[1]) * (si[1])) + )) + ((si[1]) * (((-model.q[1]) * (si[0])) + ((model.q[0]) * (si[1])) + )) + '
# '((si[0]) * (((model.q[2]) * (si[0])) + ((-model.q[1]) * (si[1])))) + ((si[1]) * (((-model.q[1]) * (si[0])) + ((model.q[0]) * (si[1]))))'
# '(si[0] * ((model.q[2] * si[0]) + (-model.q[1] * si[1]))) + (si[1] * ((-model.q[1] * si[0]) + (model.q[0] * si[1])))'
# 'si[0] * ((model.q[2] * si[0]) + (-model.q[1] * si[1]))) + (si[1] * ((-model.q[1] * si[0]) + (model.q[0] * si[1]))'
# 'si[0] * (model.q[2] * si[0] + -model.q[1] * si[1])) + (si[1] * ((-model.q[1] * si[0]) + (model.q[0] * si[1]))'


#q = [[sym('q2'), -sym('q1')];[-sym('q1'), sym('q0')]]
#s = [[sym('s0')]; [sym('s1')]]
#s' * q * s



