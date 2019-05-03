import numpy as np
import math
import itertools



class Monomial:
	def __init__(self, powers, coefficient=1.0):
		if powers.__class__.__name__ in ['tuple', 'list']:
			self.powers = MultiIndex(powers)
		else:
			self.powers = powers
		self.coefficient = coefficient

	def evaluate(self, x):
		product = 1.0
		for i, a in enumerate(self.alpha):
			power = 1.0
			for _ in range(a):
				power = power * x[i]
			product = product * power
		return product

	@property
	def degree(self):
		return self.powers.degree

	def differentiate_single_component(self, idx):
		if self.powers.alpha[idx] == 0:
			return Monomial(
				MultiIndex([0 for _ in self.powers.alpha]),
				0.0
			)
		if self.powers.alpha[idx] == 1:
			return Monomial(
				MultiIndex([a if i != idx else 0 for i, a in enumerate(self.powers.alpha)]),
				self.coefficient
			)
		return Monomial(
			MultiIndex([a if i != idx else a-1 for i, a in enumerate(self.powers.alpha)]),
			self.coefficient * self.powers.alpha[idx]
		)

	def pretty_print(self):
		return str(self.coefficient) + " * " + " * ".join([
			"x[" + str(i) + "]**" + str(a)
			for i, a in enumerate(self.powers.alpha)
		])

	def copy(self):
		return Monomial(self.powers.copy(), self.coefficient)

	def differentiate(self, d):
		current = self.copy()
		for i, a in enumerate(d.alpha):
			for _ in range(a):
				current = current.differentiate_single_component(i)
		return current