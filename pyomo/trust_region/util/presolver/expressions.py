import json
import numpy as np


def parse_json_expr_expression(json_expr):
	expr_type = json_expr['type']
	if expr_type == 'product':
		return Product([
			parse_json_expr_expression(e)
			for e in json_expr['factors']
		])
	elif expr_type == 'sum':
		return Sum([
			parse_json_expr_expression(e)
			for e in json_expr['terms']
		])
	elif expr_type == 'power':
		return Power(
			parse_json_expr_expression(json_expr['base']),
			parse_json_expr_expression(json_expr['power'])
		)
	elif expr_type == 'negate':
		return Negate(
			parse_json_expr_expression(json_expr['expression'])
		)
	elif expr_type == 'constant':
		return Constant(
			json_expr['value']
		)
	elif expr_type == 'variable':
		return Variable(
			json_expr['name']
		)
	elif expr_type == 'vector':
		return Variable([
			parse_json_expr_expression(c)
			for c in json_expr['components']
		])
	elif expr_type == 'array':
		return Variable([
			[
				parse_json_expr_expression(c)
				for c in row
			]
			for row in json_expr['components']
		])
	else:
		raise Exception('unknown expression type: ' + expr_type)


def _repeatedly_simplify(e):
	old = e
	simplified, e = e.simplify()
	if simplified:
		print('from', old.pretty_print())
		print('to', e.pretty_print())
		print("======================")
	simplified_again = simplified
	while simplified_again:
		old = e
		simplified_again, e = e.simplify()
		if simplified_again:
			print('from', old.pretty_print())
			print('to', e.pretty_print())
			print("======================")
	return simplified, e


def simplify(expression):
	print('simplifying', expression.pretty_print())
	return _repeatedly_simplify(expression)[1]


def create_variable_array(name, dimension):
	return Vector([
		IndexedVariable(name, i)
		for i in range(dimension)
	])


class Expression:
	def __init__(self):
		pass

	def get_type(self):
		raise Exception("Not implemented")

	def to_json_expr(self):
		raise Exception("Not implemented")

	def pretty_print(self):
		raise Exception("Not implemented")

	def evaluate(self, values):
		raise Exception("Not implemented")

	def differentiate(self, variable):
		raise Exception("Not implemented")

	def clone(self):
		raise Exception("Not implemented")

	def simplify(self):
		raise Exception("Not implemented")

	def expand(self):
		raise Exception("Not implemented")

	def is_one(self):
		return False

	def is_zero(self):
		return False

	def gradient(self, vector_variable):
		return Vector([
			self.differentiate(comp.pretty_print())
			for comp in vector_variable.components
		])


class Product(Expression):
	def __init__(self, factors=[]):
		self.factors = factors

	def times(self, factor):
		ret = self.clone()
		ret.factors.append(factor.clone())
		return ret

	def expand(self):
		other_terms = [Product([
			factor.clone()
			for factor in self.factors
			if factor.get_type() != 'sum' or len(factor.terms) == 0
		])]
		for sum_factor in self.factors:
			if sum_factor.get_type() != 'sum' or len(sum_factor.terms) == 0:
				continue
			next_other_terms = []
			for term in sum_factor.terms:
				for other_term in other_terms:
					next_other_terms.append(other_term.times(term))
			other_terms = next_other_terms
		return Sum(other_terms)

	def get_type(self):
		return 'product'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'factors': [factor.to_json_expr() for factor in self.factors]
		}

	def pretty_print(self):
		return "*".join(["(" + factor.pretty_print() + ")" for factor in self.factors])

	def evaluate(self, values):
		accum = 1.0
		for factor in self.factors:
			accum *= factor.evaluate(values)
		return accum

	def differentiate(self, variable):
		return Sum([
			Product([
				self.factors[j].clone() if i != j else self.factors[j].differentiate(variable)
				for j in range(len(self.factors))
			])
			for i in range(len(self.factors))
		])

	def simplify(self):
		simplified = False
		new_factors = []
		for child in self.factors:
			s, c = _repeatedly_simplify(child)
			if s:
				simplified = True
			new_factors.append(c)

		simplified_again = True
		while simplified_again:
			simplified_again = False
			if len(new_factors) == 0:
				return True, One()

			if len(new_factors) == 1:
				return True, new_factors[0]

			for e in new_factors:
				if e.is_zero():
					return True, Zero()

			products = [e for e in new_factors if e.get_type() == 'product']
			non_products = [e for e in new_factors if e.get_type() != 'product']
			if len(products) > 0:
				new_factors = (
						[np for np in non_products] +
						[e for g in products for e in g.factors]
				)
				simplified = True
				simplified_again = True
				continue

			# could take away inverses as well, but too hard

			old_len = len(new_factors)
			new_factors = [f for f in new_factors if not f.is_one()]
			if old_len != len(new_factors):
				simplified = True
				simplified_again = True
				continue

			constants = [f for f in new_factors if f.get_type() == 'constant']
			if len(constants) >= 2:
				new_factors = [
								  f for f in new_factors if f.get_type() != 'constant'
							  ] + [
								  Constant(Product(constants).evaluate(None))
							  ]
				simplified = True
				simplified_again = True
				continue

		return simplified, Product(new_factors)

	def clone(self):
		return Product([e.clone() for e in self.factors])


class Sum(Expression):
	def __init__(self, terms=[]):
		self.terms = terms

	def get_type(self):
		return 'sum'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'terms': [term.to_json_expr() for term in self.terms]
		}

	def pretty_print(self):
		return "+".join([term.pretty_print() for term in self.terms])

	def evaluate(self, values):
		accum = 0.0
		for term in self.terms:
			accum += term.evaluate(values)
		return accum

	def clone(self):
		return Sum([e.clone() for e in self.terms])

	def differentiate(self, variable):
		return Sum([
			term.differentiate(variable)
			for term in self.terms
		])

	def simplify(self):
		simplified = False
		new_terms = []
		for child in self.terms:
			s, c = _repeatedly_simplify(child)
			if s:
				simplified = True
			new_terms.append(c)

		simplified_again = True
		while simplified_again:
			simplified_again = False
			if len(new_terms) == 0:
				return True, Zero()

			if len(new_terms) == 1:
				return True, new_terms[0]

			sums = [e for e in new_terms if e.get_type() == 'sum']
			non_sums = [e for e in new_terms if e.get_type() != 'sum']
			if len(sums) > 0:
				new_terms = (
						[np for np in non_sums] +
						[e for g in sums for e in g.terms]
				)
				simplified = True
				simplified_again = True
				continue

			# could take away inverses as well, but too hard

			old_len = len(new_terms)
			new_terms = [f for f in new_terms if not f.is_zero()]
			if old_len != len(new_terms):
				simplified = True
				simplified_again = True
				continue

			constants = [f for f in new_terms if f.get_type() == 'constant']
			if len(constants) > 1:
				new_terms = [
								f for f in new_terms if f.get_type() != 'constant'
							] + [
								Constant(Sum(constants).evaluate(None))
							]
				simplified = True
				simplified_again = True
				continue

		return simplified, Sum(new_terms)


class Power(Expression):
	def __init__(self, base, power):
		self.base = base
		self.power = power

	def get_type(self):
		return 'power'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'base': self.base.to_json_expr(),
			'power': self.power.to_json_expr()
		}

	def pretty_print(self):
		return "(" + self.base.pretty_print() + ")^(" + self.power.pretty_print() + ")"

	def evaluate(self, values):
		return self.base.evaluate(values) ** self.power.evaluate(values)

	def clone(self):
		return Power(self.base.clone(), self.power.clone())

	def differentiate(self, variable):
		# assume that the power does not have the variable
		return Product([
			self.power.clone(),
			Power(
				self.base.clone(),
				Sum([self.power.clone(), Negate(One())])
			),
			self.base.differentiate(variable)
		])

	def simplify(self):
		simplified1, new_base = _repeatedly_simplify(self.base)
		simplified2, new_power = _repeatedly_simplify(self.power)
		if new_power.is_one():
			return True, new_base
		if new_power.is_zero():
			return True, One()
		if new_base.is_zero():
			return True, Zero()
		if new_base.is_one():
			return True, One()
		return simplified1 or simplified2, Power(new_base, new_power)


class PositivePart(Expression):
	def __init__(self, expr):
		self.expr = expr

	def get_type(self):
		return 'positive-part'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'expression': self.expr.to_json_expr()
		}

	def pretty_print(self):
		return "[" + self.expr.pretty_print() + "]_+"

	def evaluate(self, variables):
		return max(0, self.expr.evaluate(variables))

	def clone(self):
		return PositivePart(self.expr.clone())

	def differentiate(self, variable):
		return Product([HeavySide(self.expr.clone()), self.expr.differentiate(variable)])

	def simplify(self):
		simplified, e = _repeatedly_simplify(self.expr)
		if e.get_type() == 'constant':
			return True, Constant(max(0, e.value)).simplify()[1]
		return simplified, PositivePart(e)


class HeavySide(Expression):
	def __init__(self, expr):
		self.expr = expr

	def get_type(self):
		return 'heavyside'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'expression': self.expr.to_json_expr()
		}

	def pretty_print(self):
		return "H_0(" + self.expr.pretty_print() + ")"

	def evaluate(self, variables):
		return 1.0 if self.expr.evaluate(variables) >= 0 else 0.0

	def clone(self):
		return HeavySide(self.expr.clone())

	def differentiate(self, variable):
		raise Exception("Not Supported")

	def simplify(self):
		simplified, e = _repeatedly_simplify(self.expr)
		if e.get_type() == 'constant':
			return True, One() if e.value >= 0 else Zero()
		return simplified, HeavySide(e)


class Nothing(Expression):
	def get_type(self):
		return 'nothing'

	def to_json_expr(self):
		return {'type': self.get_type()}

	def pretty_print(self):
		return 'nothing'

	def evaluate(self, values):
		raise Exception('Cannot evaluate Nothing')

	def clone(self):
		return Nothing()

	def differentiate(self, variable):
		return self.clone()

	def simplify(self):
		return False, self.clone()


class Negate(Expression):
	def __init__(self, expr):
		self.expr = expr

	def get_type(self):
		return 'negate'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'expression': self.expr.to_json_expr()
		}

	def pretty_print(self):
		return "-(" + self.expr.pretty_print() + ")"

	def evaluate(self, values):
		return -self.expr.evaluate(values)

	def clone(self):
		return Negate(self.expr.clone())

	def differentiate(self, variable):
		return Negate(self.expr.differentiate(variable))

	def simplify(self):
		simplified, e = _repeatedly_simplify(self.expr)
		if e.get_type() == 'constant':
			return True, Constant(-e.value).simplify()[1]
		return simplified, Negate(e)


class Constant(Expression):
	def __init__(self, value):
		Expression.__init__(self)
		self.value = value

	def get_type(self):
		return 'constant'

	def is_one(self):
		return self.value == 1

	def is_zero(self):
		return self.value == 0

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'value': self.value
		}

	def pretty_print(self):
		return str(self.value)

	def evaluate(self, values):
		return self.value

	def clone(self):
		return Constant(self.value)

	def differentiate(self, variable):
		return Zero()

	def simplify(self):
		if self.is_one():
			return True, One()
		if self.is_zero():
			return True, Zero()
		return False, self.clone()


class Variable(Expression):
	def __init__(self, variable_name):
		self.name = variable_name

	def get_type(self):
		return 'variable'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'name': name
		}

	def pretty_print(self):
		return self.name

	def evaluate(self, values):
		return values[self.name]

	def clone(self):
		return Variable(self.name)

	def differentiate(self, variable):
		if self.name == variable:
			return One()
		else:
			return Zero()

	def simplify(self):
		return False, self.clone()


class Vector(Expression):
	def __init__(self, components):
		self.components = components

	def get(self, idx):
		return self.components[idx]

	def get_type(self):
		return 'vector'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'components': [
				component.to_json_expr()
				for component in self.components
			]
		}

	def pretty_print(self):
		return 'np.array([' + ','.join([
			component.pretty_print()
			for component in self.components
		]) + ']'

	def evaluate(self, values):
		return np.array([
			c.evaluate(values) for c in self.components
		])

	def jacobian(self, vector_variable):
		return Array([
			comp.gradient(vector_variable).components
			for comp in self.components
		])

	def clone(self):
		return Vector([e.clone() for e in self.components])

	def simplify(self):
		simplified = False
		ret = []
		for c in self.components:
			simplified_again, c = c.simplify()
			if simplified_again:
				simplified = True
			while simplified_again:
				simplified_again, c = c.simplify()
			ret.append(c)
		return simplified, Vector(ret)


class Array(Expression):
	def __init__(self, components):
		self.components = components

	def get_type(self):
		return 'array'

	def to_json_expr(self):
		return {
			'type': self.get_type(),
			'components': [
				[
					component.to_json_expr()
					for component in row
				]
				for row in self.components
			]
		}

	def pretty_print(self):
		return 'np.array([' + ','.join([
			'[' + ','.join([
				component.pretty_print()
				for component in row
			]) + ']'
			for row in self.components
		]) + ']'

	def evaluate(self, values):
		return np.array([
			[
				c.evaluate(values) for c in row
			]
			for row in self.components
		])

	def clone(self):
		return Array([[e.clone() for e in row] for row in self.components])

	def simplify(self):
		es = [[_repeatedly_simplify(e) for e in r] for r in self.components]
		return any([x[0] for r in es for x in r]), Array([[x[1] for x in r] for r in es])

	def determinant(self):
		pass


class IndexedVariable(Variable):
	def __init__(self, name, idx):
		# VariableExpression.__init__(self, name + "[" + str(idx) + "]")
		self.variable_name = name
		self.idx = idx

	def clone(self):
		return IndexedVariable(self.variable_name, self.idx)

	def evaluate(self, variables):
		v = variables[self.variable_name]
		return v[self.idx]

	def pretty_print(self):
		return self.variable_name + "[" + str(self.idx) + "]"

	def differentiate(self, variable):
		if self.pretty_print() == variable:
			return One()
		else:
			return Zero()

	def simplify(self):
		return False, self.clone()


class One(Constant):
	def __init__(self):
		Constant.__init__(self, 1)

	def clone(self):
		return One()

	def is_one(self):
		return True

	def simplify(self):
		return False, self.clone()


class Zero(Constant):
	def __init__(self):
		Constant.__init__(self, 0)

	def clone(self):
		return Zero()

	def is_zero(self):
		return True

	def simplify(self):
		return False, self.clone()


# expr = ProductExpression([
# 	One(),
# 	ProductExpression([
# 		ConstantExpression(3),
# 		ConstantExpression(5),
# 	]),
# 	SumExpression([
# 		VariableExpression('foo'),
# 		ConstantExpression(0)
# 	]),
# 	ProductExpression([
# 		VariableExpression('foo'),
# 		VariableExpression('foo'),
# 		VariableExpression('foo'),
# 		VariableExpression('foo'),
# 	])
# ])
#
# expr = ProductExpression([
# 	SumExpression([
# 		VariableExpression('a'),
# 		VariableExpression('b'),
# 	]),
# 	SumExpression([
# 		VariableExpression('c'),
# 		VariableExpression('d'),
# 	])
# ])
#
# print(expr.pretty_print())
# print(expr.expand().pretty_print())
# print(expr.pretty_print())
# print(expr.differentiate('foo').pretty_print())
# print(simplify(expr.differentiate('foo')).pretty_print())
