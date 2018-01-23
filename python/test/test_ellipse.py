# from numpy import array
# from numpy import empty
from numpy.linalg import cholesky
from numpy import linspace
from numpy import asarray
from numpy import array
from numpy import empty
from numpy import dot
from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt
import matplotlib.patches as patches



from utilities.ellipse import getMaximalEllipseContaining

#x = array((2, 1/1.5))
#A = asarray([
#	[1, 0.25],
#	[-1, 1.5],
#	[0, -1]
#])
#b = asarray([
#	8,
#	1,
#	8
#])


x = array((7, -3))
A = asarray([
	[-1.0, 0],
	[1, 5],
	[0, -1]
])
b = asarray([
	8.0,
	8,
	8
])


mat2cons = lambda idx: lambda x: dot(A[idx, :], x) - b[idx]

ellipse = getMaximalEllipseContaining(A, b, x)
#Q = ellipse['Q']
#ds = ellipse['ds']
#lambdas = ellipse['lambdas']


#for d in ds:
#	print('norm', dot(d,dot(Q, d)))

# mat = zeros((2, 2))
# # mat[:, 0] = array((0, 0)) - x
# mat[:, 0] = another - x
# mat[:, 1] = array((closestX, closestY)) - x
# A = inv(mat)
#

#L = 10
#fig = plt.figure()
#ax = plt.axes()
#ax.set_xlim([-L, L])
#ax.set_ylim([-L, L])
#t = linspace(0, 2 * L, 1000)
# ax.plot(t, a * t)
# ax.plot(t, -a * t)
#ax.plot(array(x[0]), array(x[1]), 'bo')
#ax.plot(array(include[0]), array(include[1]), 'ro')

#tx = linspace(-L, L, num=100)
#ty = linspace(-L, L, num=100)
#X, Y = meshgrid(tx, ty)
#Z = empty((len(ty), len(tx)))

#def plotContour(fun, color, levels=[-1, 0]):
#	for i in range(0, len(tx)):
#		for j in range(0, len(ty)):
#			Z[j, i] = fun(array([tx[i], ty[j]]))
#	CS = plt.contour(X, Y, Z, levels, colors=color)

#for i in arange(A.shape[0]):
#	plotContour(mat2cons(i), 'b')

#for d in ds:
#	ax.add_patch(patches.Arrow(
#		x=ellipse['center'][0], y=ellipse['center'][1],
#		dx=d[0], dy=d[1],
#		facecolor="green", edgecolor="green"
#	))

#plotContour(ellipse['fun'], 'k', levels=[-1, 0])

#plt.show()
#plt.savefig('images/ellipse.png')

# f = lambda x: 0.5 * dot(x, dot(Q, x))
# fg = lambda x: dot(Q, x)
# for i in range(len(ds)):
# 	print('gradient of ellipse', fg(ds[i]))
# 	print('gradient of constraint', lambdas[i] * A[i, :])

print('all done')


