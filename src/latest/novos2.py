# -*- coding: utf-8 -*-
import numpy as n
import random as r
import pylab as p
import pca_module as pp
from scipy.interpolate import interp1d

# distribuição gaussiana
# mu: determina o centro da distribuição, quando mu=0 e std=1, temos a distribuição normal
# std: determina o quão espaçado é a boca do sino
def gauss(x, mu=0, std=1):
    return p.exp(-((x - mu)**2)/(2*std)) / p.sqrt(2*p.pi*std)

def pca(dados):
    # II. normalizamos a matriz de dados (X = X - mean) e dividimos pelo d.p.
    #     (X = (X - mean) / dp
    for i in xrange(dados.shape[1]):
        dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

    # III. calculamos a matriz de covariância de X
    matriz_cov = n.cov(dados, bias=1, rowvar=0)

    # IV. calculamos os autovetores e autovalores e ordenamos em ordem decresc.
    autovalores, autovetores = n.linalg.eig(matriz_cov)
    args = n.argsort(autovalores)[::-1]
    autovalores = autovalores[args]
    autovetores = autovetores[args]

    # V. calculamos o vetor de componentes principais (= autovetores)

    # VI. calculamos os dados finais Y
    dados_finais = n.dot(autovetores.T, dados.T)

    # VII. interpretação
    autovalores_prop = [av/n.sum(autovalores) for av in autovalores]

    return matriz_cov, autovetores, autovalores, autovalores_prop, dados_finais.T

def pearson(dados):
    num_col = dados.shape[1]
    dp = n.std(dados, 0)
    covm = n.cov(dados, bias=1, rowvar=0)
    pearson = n.zeros((num_col, num_col))
    for i in xrange(num_col):
        for j in xrange(num_col):
            pearson[i,j] = covm[i,j] / (dp[i] * dp[j])
    return pearson

def montecarlo(qtd_aleatorios):
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
    aleatorios_todos = []
    for i in range(qtd_aleatorios):
        aleatorios_todos.append([r.choice(notas_todas[i]) for i in range(qtd_medidas)])
        p.hist([r.choice(notas_todas[i]) for i in range(qtd_medidas)], normed=1)
    return aleatorios_todos

def principal():
    # I. carregamos a matriz de dados    

    # Filósofos:
    #dados = n.loadtxt('notas_filosofos.txt')

    # Compositores:
    # dados = n.loadtxt('notas_filosofos.txt')
    # covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
    # print '\n\nautovetores', autovetores.T
    # print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
    # print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)
    # print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
    # print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
    # print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
    # print '\nDados finais:\n', n.around(dados_finais, decimals=2)
    # c1 = dados_finais[:,0]
    # c2 = dados_finais[:,1]

    # p.clf()
    # p.plot(c1, c2)
    # p.xlim((-4,4))
    # p.ylim((-4,4))
    # p.savefig('pca.eps')

    # Aleatórios:
    #num_agentes = [10,15,20,25,30,35,40,45,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    num_agentes = [3]
    num_vars = 8
    a = []
    autovalores_todos = {}
    for i in num_agentes:
        autovalores_todos[i] = []
        for j in xrange(1):
            # aqui eu jogo os valores sampleados
            #print '\n\ndados toscos\n\n', dados
            #print '\n\ndados monte carlo\n\n', montecarlo(10)
            d = n.loadtxt('notas_filosofos.txt')
            #dados = n.array([[r.uniform(1,9) for x in range(num_vars)] for y in range(i)])
            dados = n.array(montecarlo(i))
            print dados
            covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
            print '\n*** Teste %s para %s agentes ***\n' % (j, i)
            print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
            print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)
            print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
            print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
            print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
            #a.append(autovalores_prop[0] + autovalores_prop[1])
            autovalores_todos[i].append(autovalores_prop)
    # analisa autovalores dos testes aleatórios...
    for i in num_agentes:
        print '\n\nPara os autovalores de % agentes:\n', i
        print '\nMédia da soma dos dois primeiros:', n.mean([av[0] + av[1] for av in autovalores_todos[i]])
        print '\nVariância dos autovalores:', [n.var(av) for av in autovalores_todos[i]]
    # p.plot(num_agentes, a)
    # p.savefig('foo.eps')
    p.show()
        
if __name__ == '__main__':
    principal()
