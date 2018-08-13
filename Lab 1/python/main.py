__author__ = 'pierrerudin'
import monkdata as m
import dtree
import random
#import drawtree_qt5 as qt
import pylab

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def rnd(x,n):
    return float(int(x*10**n+.5))/(10**n)

p = dtree.entropy(m.monk3)

def assignment_3():
    i = 0
    k = 0
    while i <= 5:
        f = dtree.averageGain(m.monk1, m.attributes[i])
        k += f
        print(i + 1, rnd(f, 8))
        i += 1

    print(k)

def assignment_5():
    t = dtree.buildTree(m.monk1, m.attributes)
    print(dtree.check(t, m.monk1))
    print(dtree.check(t, m.monk1test))

def assignment_7(monktrain, monkval):
    t = dtree.buildTree(monktrain, m.attributes)
    p1 = performance = dtree.check(t, monkval)
    better_found = True
    while better_found:
        prunes = dtree.allPruned(t)
        better_found = False
        for prune in prunes:
            tmp_performance = dtree.check(prune, monkval)
            if tmp_performance > performance:
                t = prune
                performance = tmp_performance
                better_found = True
    return p1,dtree.check(t, monkval)

runs = 10
fractions = [i/10 for i in range(3,9)]
classification_performance = []
for fraction in fractions:
    i = runs
    performance = [0,0]
    while i > 0:
        monktrain, monkval = partition(m.monk2, fraction)
        performance_run = assignment_7(monktrain, monkval)
        performance = [performance[i] + performance_run[i] for i in range(0,2)]
        i -= 1
    performance = [1-performance[i]/runs for i in range(0,2)]
    classification_performance.append(performance)

before_pruning = [c[0] for c in classification_performance]
after_pruning = [c[1] for c in classification_performance]

pylab.plot(fractions,
           before_pruning,
           'o',
           label='Before pruning')
pylab.plot(fractions,
           after_pruning,
           '+',
           label='After pruning')
pylab.legend(loc='upper right')
pylab.xlabel('Classification error')
pylab.ylabel('Fraction')
pylab.title('MONK-2')
pylab.show()
print(classification_performance)

#rnd(1.00009,4)
#assignment_3()
#assignment_5()
#print(p)
