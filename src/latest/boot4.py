# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
import random as r
from scipy.interpolate import interp1d

# distribuição gaussiana
# mu: determina o centro da distribuição, quando mu=0 e std=1, temos a distribuição normal
# std: determina o quão espaçado é a boca do sino
def gauss(x, mu=0, std=1):
    return p.exp(-((x - mu)**2)/(2*std)) / p.sqrt(2*p.pi*std)

# carregando dados
dados = n.loadtxt('notas_filosofos.txt')
# notas de platão
minimo = 1.0
maximo = 9.0
# cuidado com std, para evitar picos
std = 0.4
samples_todos = []
for j in range(100):
    d_todas = []
    xs_todas = []
    for i in range(len(dados)):
        # 1. notas para cada linha (cada filósofo), notas = {x1, x2, ..., x8}
        notas = dados[i]

        # 2. calculamos 8 pontos aleatórios xsN e a distância dK para cada filósofo
        xs = []
        for i in range(len(notas)):
            xs.append(p.uniform(minimo, maximo))
        xs_todas.append(xs)

        # 3. calculamos dK
        dk = n.sqrt(n.sum([(xs[i] - notas[i])**2 for i in range(len(notas))]))

        # 4. guardamos dk em d_todas
        d_todas.append(dk)

    po = n.sum([n.exp(-0.5*((d_todas[i]**2)/n.std(dados, 0))) for i in range(7)])

    print '\nsimulação: ', j+1, '\n\nxs:\n', xs_todas, '\n\np: ', po

    # agora o fv é nosso po
    a = p.uniform()

    if a < po:
        samples_todos.append(xs_todas)

#int_x = p.arange(minimo, maximo,0.1)
#print len(int_x), len(samples_todos)
#f = interp1d(int_x, samples_todos[:len(int_x)])
#p.plot(int_x, [f(v) for v in int_x], 'bx')
print xs_todas
p.hist(samples_todos, normed=1, alpha=0.45, facecolor='green')
p.show()


