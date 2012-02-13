# -*- coding: utf-8 -*-
import numpy as n
import pylab as p

vi = n.array([1,  1,   1])
vj = n.array([2,  1,   3])
w  = n.array([1,  1.5, 3])
d = n.abs( (vj[0]-vi[0])*w[0] + (vj[1]-vi[1])*w[1] + (vj[2]-vi[2])*w[2] + (-((vj[0]**2 - vi[0]**2)/2) - ((vj[1]**2 - vi[1]**2)/2) -((vj[2]**2 - vi[2]**2)/2)) ) / n.sqrt( (vj[0]-vi[0])**2 + (vj[1]-vi[1])**2 + (vj[2]-vi[2])**2)

p.plot(vi[0], vi[1], vi[2], 'bo')
p.plot(vj[0], vj[1], vj[2], 'ro')
p.plot([vi[0],vj[0]], [vi[1],vj[1]], [vi[2],vj[2]])
p.plot(w[0], w[1], w[2], 'go')
p.xlim((-6,6,6))
p.ylim((-6,6,6))
p.show()

