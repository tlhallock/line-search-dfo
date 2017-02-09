
from numpy.random import seed
from numpy import array as arr

seed(seed=1776)

from utilities.history_bound_function import HistoryBoundRandomFunction as rfun

fun = rfun(1,2)

x = arr([1, 1])
print(fun.evaluate(x).evaluate(x))

x = arr([1, 1.1])
print(fun.evaluate(x).evaluate(x))

x = arr([5, 5])
print(fun.evaluate(x).evaluate(x))


