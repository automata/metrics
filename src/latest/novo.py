# -*- coding: utf-8 -*-
import numpy as n
import random as r
import pylab as p
import pca_module as pp

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

def principal():
    # I. carregamos a matriz de dados    

    # Filósofos:
    #dados = n.loadtxt('notas_filosofos.txt')

    # Compositores:
    #dados = n.loadtxt('notas_compositores.txt')
    # covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
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
    num_agentes = [7,10,15,20,25,30,35,40,45,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    num_vars = 8
    a = []
    autovalores_todos = {}
    for i in num_agentes:
        autovalores_todos[i] = []
        for j in xrange(100):
            dados = n.array([[r.uniform(1,9) for x in range(num_vars)] for y in range(i)])
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
        
if __name__ == '__main__':
    principal()
