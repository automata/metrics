# -*- coding: utf-8 -*-
import pylab as p

# distribuição normal
def f(x):
    mu = 0
    std = 1
    return p.exp(-((x - mu)**2)/(2*std)) / p.sqrt(2*p.pi*std)

novos = []
minimo = -10.0
maximo = 10.0

for i in range(1000):
    # 1. sorteia um valor uniforme entre min e max
    v = p.uniform(minimo, maximo)

    # 2. calcula p(v)
    fv = f(v)

    # 3. sorteia um valor uniforme entre 0 e 1
    a = p.uniform()

    # 4. se a < v => guarda v na lista
    if a < fv:
        novos.append(v)

#p.hist(novos, normed=1, alpha=0.75)
int_x = p.arange(minimo, maximo,0.1)
p.plot(int_x, [f(x) for x in int_x])
p.axis([minimo, maximo, 0, 0.6])
p.grid(True)
#p.show()
p.savefig('hist_gauss.png')
