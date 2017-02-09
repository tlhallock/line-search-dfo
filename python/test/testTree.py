
import sys

from utilities.boxable_query_set import Tree

t = Tree()

t.balance()
t.print(sys.stdout)

t.add(1, "one")
t.add(2, "two")
t.add(3, "three")
t.add(4, "four")
t.add(5, "five")
t.add(6, "six")
t.add(7, "seven")
t.add(8, "eight")
t.add(9, "nine")
t.add(10, "ten")
t.add(11, "eleven")

t.print(sys.stdout)

for s in t.range(2.4, 4.5):
	print(s)

t.balance()
t.print(sys.stdout)

for i in range(132):
	t.add(i + .5, "num")
	t.print(sys.stdout)
	sys.stdout.write("==================================\n")
