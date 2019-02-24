
import numpy as np
import itertools

#
# class Poly:
# 	def __init__(self):
# 		powers =
#

def get_power(powers, component):
	return len([x for x in powers if x == component])


def differentiate(powers, with_respect_to):
	p = get_power(powers, with_respect_to)
	if p == 0:
		return "", (), True
	if p == 1:
		return "", tuple(p for p in powers if p != with_respect_to), False
	new_num = p - 1
	return p, tuple([
		p for p in powers if p != with_respect_to
	] + [
		with_respect_to for _ in range(new_num)
	]), False


def format_powers(coeff, powers):
	if len(powers) == 0:
		return "0.0"
	return coeff + "*" + "*".join(["x[" + str(i) + "]" for i in powers])


def get_gradient_component(powers, i, coeff):
	c, ps, zero = differentiate(powers, i)
	if zero:
		return "0.0"
	if len(ps) == 0:
		return str(c) + ("*" if c != "" else "") + coeff
	return format_powers(
		str(c) + ("*" if c != "" else "") + coeff,
		ps
	)


def get_hessian_component(powers, i, j, coeff):
	c1, ps1, zero = differentiate(powers, i)
	if zero:
		return "0.0"
	if len(ps1) == 0:
		return "0.0"
	c2, ps2, zero = differentiate(ps1, j)
	if zero:
		return "0.0"
	if len(ps2) == 0:
		return str(c1) + ("*" if c1 != "" else "") + str(c2) + ("*" if c2 != "" else "") + coeff

	return format_powers(
		str(c1) + ("*" if c1 != "" else "") + str(c2) + ("*" if c2 != "" else "") + coeff,
		ps2
	)



def print_basis_func(dimension, power):
	ret = []
	for powers in itertools.combinations_with_replacement([i for i in range(dimension)], power):
		coeff = "1.0/(" + "*".join([str(i) for i in range(1, power+1)]) + ")"
		expr = format_powers(coeff, powers)
		gradient = "np.array([" + ", ".join([
			get_gradient_component(powers, i, coeff) for i in range(dimension)
		]) + "])"
		hessian = "np.array([" + ", ".join([
			"[" + ", ".join([
				get_hessian_component(powers, j, i, coeff)
				for j in range(dimension)
			]) + "]"
			for i in range(dimension)
		]) + "])"
		ret.append('''{
\t"func": lambda x: ''' + expr + ''',
\t"grad": lambda x: ''' + gradient + ''',
\t"hess": lambda x: ''' + hessian + ''',
\t"degree": ''' + str(power) + '''
}''')
	return ", ".join(ret)


e = ",".join([
	print_basis_func(2, i).replace('1.0/(1)', "1.0").replace("1.0*", "")
	for i in range(4 + 1)
])
print(e)
