# -*- coding: utf-8 -*-
# a validacao consiste no seguinte...
# sabemos que a equacao que encontramos eh valida para 4D, serah mesmo?
# para tal, validamos para 2D.
# note abaixo como ficou a equacao para 2D

import numpy as n
import pylab as p

# temos dois pontos quaisquer vi e vj (para 4D, vi e vj sao vetores de 4D)
vi = n.array([1., 1]) # esse eh a thesis
vj = n.array([5., 5]) # esse a antithesis
# iremos testar os pontos (x,y) seguintes... (1, 1), (1,5), (2,1), ...
for x in [1., 2., 3., 4., 5.]:
    for y in [1., 5.]:
        w = n.array([x, y]) # esse eh a synthesis, ou seja, o ponto que iremos calcular a distancia dele ateh a reta (plano em 2D == reta)
        # e abaixo eh a equacao principal... atencao a ela!!!!
        p1 = ( (vj[0]-vi[0])*w[0] + (vj[1]-vi[1])*w[1] + (vi[0]**2 - vj[0]**2)/2 + (vi[1]**2 - vj[1]**2)/2 )
        p2 = n.sqrt( (vj[0]-vi[0])**2 + (vj[1]-vi[1])**2 )
        d = p1 / p2
        # plotamos todos os pontos para os quais calculamos a distancia como sendo xs vermelhos
        p.plot(w[0], w[1], 'rx')
        # plotamos junto ao ponto a distancia calculada (dele ateh a reta)
        p.text(w[0], w[1], str(n.around(d, decimals=1)))
        #print w, d

# plotamos o ponto vi
p.plot(vi[0], vi[1], 'bo')
# e o ponto vj
p.plot(vj[0], vj[1], 'bo')
# tracamos a reta perpendicular ao plano (reta)
p.plot([vi[0],vj[0]], [vi[1],vj[1]])
# tracamos o plano, eh desse plano (no caso, reta) que calculamos a distancia dele aos pontos w
p.plot([0,6], [6,0], 'r--')
p.xlim((0,6))
p.ylim((0,6))
p.show()

