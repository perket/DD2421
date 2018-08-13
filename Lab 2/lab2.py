__author__ = 'pierrerudin'

from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab, random, math

# Init kernel
KERNELFUNCTION = 'rbf'  # linear, polynomial, rbf, sigmoid
N = 20
SIGMA = 5  # Low = High bias, low variance, High = Low bias, high variance
K = 6
DELTA = .1
C = 1000000#1000000+-1.25#10000+-1.5 #5+-1.75 #1200+-1.5#1 000 000 000 000
SLACK = True


def kernel(x,y):
    if KERNELFUNCTION == 'linear':
        return np.dot(np.transpose(x), y) + 1
    elif KERNELFUNCTION == 'polynomial':
        return (np.dot(np.transpose(x), y) + 1) ** N
    elif KERNELFUNCTION == 'rbf':
        return np.exp(-np.dot(np.transpose(x-y),(x-y))/(2*SIGMA**2))
    elif KERNELFUNCTION == 'sigmoid':
        return np.tanh(K * np.dot(np.transpose(x), y) - DELTA)
    else:
        print("ERROR")


def buildP(x,t):
    i = j = len(x) - 1
    P = np.zeros([i+1, j+1],dtype=np.double)
    while i >= 0:
        j = len(x) - 1
        while j >= 0:
            P[i][j] = t[i]*t[j]*kernel(x[i],x[j])
            j -= 1
        i -= 1
    return P


def buildq(n):
    q = np.array([-1] * n, dtype=np.double)
    return q


def buildG(n):
    if SLACK:
        G1 = np.diag(np.ones(n) * -1)
        G2 = np.identity(n)
        G = np.vstack((G1, G2))
    else:
        G = -1 * np.identity(n, dtype=np.double)
    return G


def buildh(n):
    h = np.zeros(n, dtype=np.double)
    if SLACK:
        h2 = C * np.ones(n)
        h = np.hstack((h, h2))
    return h


def getNonZeroAlpha(alpha):
    nonZeros = []
    n = 0
    for a in alpha:
        if a > 10**-5 and (1 if SLACK else 0) * a <= C:  #(eq. 11)
            i = alpha.index(a)
            nonZeros.append([a,i])
            n += 1
    print("Support vector count: %i",n)
    return nonZeros


def indicator(xx, nonZeroAlphas, data):
    x = np.array([d[:2] for d in data])
    t = np.array([d[2] for d in data])
    i = len(nonZeroAlphas) - 1
    ind = 0
    while i >= 0:
        alpha, n = nonZeroAlphas[i]
        ti = t[n]
        xi = x[n]
        ind += alpha*ti*kernel(xx,xi)
        i -= 1
    return ind


def generateData():
    #Uncommentthelinebelowtogenerate
    #thesamedatasetoverandoveragain.
    np.random.seed(100)

    classA = [(random.normalvariate(-1.5,1),
               random.normalvariate(0.5,1),1.0)
              for i in range(10)] +\
             [(random.normalvariate(1.5,1),
               random.normalvariate(0.5,1),1.0)
              for i in range(10)]

    # classA = [(random.normalvariate(-1.5,3),
    #            random.normalvariate(0.5,3), 1.0)
    #           for i in range(5)]
    #
    # classB = [(random.normalvariate(0,2.5),
    #            random.normalvariate(-.5,2.5), -1.0)
    #           for i in range(5)]



    classB = [(random.normalvariate(0.0,.5),
               random.normalvariate(-0.5,.5), -1.0)
              for i in range(20)]

    data = classA + classB
    random.shuffle(data)
    return data


def plotData(data):
    classA = [d for d in data if d[2] > 0]
    classB = [d for d in data if d[2] < 0]

    pylab.hold (True)
    pylab.plot([p[0] for p in classA],
    [p[1] for p in classA],
    ' bo ')
    pylab.plot([p[0] for p in classB],
    [p[1] for p in classB],
    ' ro ')
    pylab.show()


def plotContour(nonZeroAlphas, data):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange (-4, 4, 0.05)
    grid = matrix ([[indicator([x, y], nonZeroAlphas, data)
                     for y in yrange]
                    for x in xrange])
    pylab.contour(xrange, yrange, grid,
                  (-1.0, 0.0, 1.0),
                  colors=('red', 'black', 'blue'),
                  linewidths = (1, 3, 1))
    plotData(data)

data = generateData()

x = np.array([d[:2] for d in data])
t = np.array([d[2] for d in data])

P = buildP(x,t)
q = buildq(len(x))
G = buildG(len(x))
print(G.shape)
h = buildh(len(x))

r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])
nonZeroAlphas = getNonZeroAlpha(alpha)
xx = np.array([4,4], dtype=np.double)
#indxx = indicator(xx, nonZeroAlphas, t, x)
plotContour(nonZeroAlphas, data)