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
minimo = 0.0
maximo = 10.0
# cuidado com std, para evitar picos
std = 0.4

notas_todas = []
qtd_medidas = 8 # 8 notas
qtd_agentes = 7 # 7 filósofos/compositores/...

for i in range(qtd_medidas):
    # notas para cada coluna (por métrica)
    notas = dados[:,i]
    # 1. plotar a distribuição gaussiana com mu centrado em cada posição
    int_x = p.arange(minimo, maximo, 0.1)
    # não preciso mais disso... agora só basta interpolar...
    #p.plot(int_x, [gauss(v, x, std) for v in int_x])

    # 2. interpolar todas as curvas de distribuição, objetivando a curva toda
    # somamos os vetores produzidos pela aplicação de guass em valores de int_x e depois dividimos por N=7
    # por fim, temos um vetor da média, e então interpolamos linearmente usando interp1d
    # essa é a função interpolada para todas as 8 medidas
    f = interp1d(int_x, reduce(lambda a,b: a + b, [n.array([gauss(v, each, std) for v in int_x]) for each in notas])/qtd_agentes)

    #p.plot(int_x, [f(v) for v in int_x], 'bx')

    # 3. essa curva define a distribuição que usaremos como p(x) em monte carlo
    novos = []
    for i in range(1000):
        # 1. sorteia um valor uniforme entre min e max
        v = p.uniform(minimo+1, maximo-1)

        # 2. calcula p(v)
        fv = f(v)

        # 3. sorteia um valor uniforme entre 0 e 1
        a = p.uniform()

        # 4. se a < v => guarda v na lista
        if a < fv:
            novos.append(v)
    notas_todas.append(novos)

#p.hist(dados[0], normed=1)
qtd_aleatorios = 10
aleatorios_todos = []
for i in range(qtd_aleatorios):
    aleatorios_todos.append([r.choice(notas_todas[i]) for i in range(qtd_medidas)])
    #p.hist([r.choice(notas_todas[i]) for i in range(qtd_medidas)], normed=1)
print aleatorios_todos

#p.show()
# novos tem os samples que irei escolher para cada coluna de nota
#p.hist(novos, normed=1, alpha=0.45, facecolor='green')
#p.grid(True)
#p.show()
#p.savefig('hist_boot.png')
