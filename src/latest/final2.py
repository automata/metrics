# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
#import pca_module as pp
#from scipy.interpolate import interp1d

# parâmetros
minimo = 1.0
maximo = 9.0
qtd_aleatorios = 500
std = 4
# carregamos os dados das notas originais
dados_orig = n.loadtxt('notas_compositores.txt')
# calculamos os samples por boostrap
dados_boot = []

def gauss(x, mu=0, std=1):
    return p.exp(-((x - mu)**2)/(2*std)) / p.sqrt(2*p.pi*std)

# repetimos até termos a quantidade que queremos de aleatorios
while (len(dados_boot) < qtd_aleatorios/7):
    d_todas = []
    xs_todas = []

    for i in xrange(len(dados_orig)):
        # 1. notas para cada linha (cada filósofo), notas = {x1, x2, ..., x8}
        notas = dados_orig[i]

        # 2. calculamos 8 pontos aleatórios xsN e a distância dK para cada filósofo
        xs = [p.uniform(minimo, maximo) for i in range(len(notas))]
        xs_todas.append(xs)

        # 3. calculamos dK
        dk = n.sqrt(n.sum([(xs[i] - notas[i])**2 for i in range(len(notas))]))

        # 4. guardamos dk em d_todas
        d_todas.append(dk)

    po = n.sum([gauss(k, 0, std) for k in d_todas])
    qo = 7 / po
    #print 'po', po, 'qo', qo
    #print 'd_todas', d_todas

    # se a < p(x) então ficamos com o valor
    a = p.uniform()
    if a < qo:
        dados_boot.append(xs_todas)     

# concatena sublistas [[a], [b,c]] => [a, b, c]
dados_boot = sum(dados_boot, [])
#print 'dados_boot', dados_boot
# unimos as duas matrizes de notas
dados = n.array(list(dados_orig) + list(dados_boot))
# print 'QTD:', len(dados)
#dados = n.array(dados_orig)

print '*** Tabela geral de notas:\n', dados

# fazemos o PCA

# normalizamos a matriz de dados (X = X - mean) e dividimos pelo d.p.
#     (X = (X - mean) / dp
for i in xrange(dados.shape[1]):
    dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

# calculamos a matriz de covariância de X
matriz_cov = n.cov(dados, bias=1, rowvar=0)
print '*** Matriz de covariância:\n', n.around(matriz_cov, decimals=2)

# calculamos os autovetores e autovalores e ordenamos em ordem decresc.
autovalores, autovetores = n.linalg.eig(matriz_cov)
args = n.argsort(autovalores)[::-1]
autovalores = autovalores[args]
autovetores = autovetores[args]
# autovalores (var.) como porcentagem dos autovalores
autovalores_prop = [av/n.sum(autovalores) for av in autovalores]
print '*** Autovalores (var. %):\n', n.around(autovalores_prop, decimals=2)

# calculamos os componentes principais para todos os dados
dados_finais = n.dot(autovetores.T, dados.T)
#print '*** Dados finais:\n', dados_finais

# calculamos os componentes principais para a submatriz de 7 elementos, as notas originais
principais_orig = n.dot(autovetores.T, dados[:7].T)

# plotamos os projeções do pca (autovetores * tabela inteira ou subtabela)
dados_finaisT = dados_finais.T
c1 = dados_finaisT[:,0]
c2 = dados_finaisT[:,1]
print '*** Componentes principais:\n', c1, c2
p.plot(c1, c2, 'rx')

principais_origT = principais_orig.T
c1_orig = principais_origT[:,0]
c2_orig = principais_origT[:,1]
print '*** Componentes principais:\n', c1_orig, c2_orig
p.plot(c1_orig, c2_orig, 'bo')
[p.text(c1_orig[i], c2_orig[i], str(i+1)) for i in range(len(c1_orig))]

p.savefig('pca_final.eps')

#print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)

