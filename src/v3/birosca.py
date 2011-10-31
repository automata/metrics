# -*- coding: utf-8 -*-
import numpy as n
import pylab as p

x = n.array([3.,9.])
y = n.array([6.,3.])
meio = n.array([(x[0] + x[1])/2, (y[0] + y[1])/2])

p.plot(x,y,'ro')
p.plot(meio[0], meio[1], 'bo')

p.xlim([0,10])
p.ylim([0,10])
p.show()

