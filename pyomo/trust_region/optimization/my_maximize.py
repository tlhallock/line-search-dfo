
import numpy as np
import trust_region.util.presolver.expressions as exprs
import trust_region.util.presolver.presolve as ps


def presolve_ellipse(A, bbar, start):
	dim = A.shape[1]
	var, l_matrix = exprs.create_upper_triangular_matrix("x", dim)
	a_matrix = exprs.create_constant_array(A)
	q_matrix = exprs.simplify(l_matrix.transpose().multiply(l_matrix))

	params = ps.Params(
		variable=var,
		f=exprs.Negate(l_matrix.multiply_diagonals()),
		c=exprs.simplify(exprs.Vector([
			exprs.Sum([
				a_matrix.row(i).multiply(q_matrix.multiply(a_matrix.row(i).transpose())).as_expression(),
				exprs.Constant(-bbar[i]*bbar[i]/2)
			])
			for i in range(len(bbar))
		])),
		x0=start,
		r0=np.linalg.norm(start)
	)
	# print(params.pretty_print())
	result = ps.solve(params)
	result['volume'] = -result['value']
	optimal_l_inverse = l_matrix.evaluate(values={'x': result['minimizer']})
	optimal_q_inverse = optimal_l_inverse.T@optimal_l_inverse
	constraint_violations = [
		A[i].T@optimal_q_inverse@A[i] - bbar[i]*bbar[i]/2
		for i in range(A.shape[0])
	]
	print(constraint_violations)
	result['l-inverse'] = optimal_l_inverse
	print(result)
	return result


#presolve(np.ones((3, 3)), np.random.random(3))
