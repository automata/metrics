# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
import pca_module as pp
#from scipy.interpolate import interp1d

# parâmetros
minimo = 1.0
maximo = 9.0
qtd_aleatorios = 50000
std = 1.4
# carregamos os dados das notas originais
dados_orig = n.loadtxt('notas_compositores.txt')
# calculamos os samples por boostrap
ncomp = 7
ncarac = 8

# repetimos até termos a quantidade que queremos de aleatorios
#while (len(dados_boot) < qtd_aleatorios/7):
pos=n.zeros(qtd_aleatorios)
pts=n.zeros((qtd_aleatorios,ncarac))

# para cada ponto artificial
for i in xrange(qtd_aleatorios):
    # calculamos um vetor de 8 notas gaussianas entre 1 e 9
    #xs = n.random.normal(loc=5., scale=2.5, size=8)
    xs = n.random.rand(ncarac)*8+1
    # colocamos esse vetor na tabela de pontos artificiais (8 valores para cada linha i)
    pts[i]=xs
    # centralizando na média das distâncias
    dk = (n.sum((xs-dados_orig)**2,1)**.5)

    po = n.sum(n.exp (-0.5*(dk/std)**2) )

    pos[i]=po
    #print 'agente', i, 'xs:', xs, 'dk:', dk, 'p:', po

pos=( (pos-pos.min())/(pos.max()-pos.min()) )*.2

coin=n.random.uniform(size=qtd_aleatorios)
#print 'todos p:', pos, 'todas moedas:', coin
results= pos > coin

pts_reais=pts[results]
#print 'samples bootstrap:', pts_reais

# concatena sublistas [[a], [b,c]] => [a, b, c]
# dados_boot = sum(dados_boot, [])
#print 'dados_boot', dados_boot
# unimos as duas matrizes de notas
dados=n.vstack((dados_orig,pts_reais))
# print 'QTD:', len(dados)
#dados = n.array(dados_orig)

#print '*** Tabela geral de notas:\n', dados

# fazemos o PCA

# normalizamos a matriz de dados (X = X - mean) e dividimos pelo d.p.
#     (X = (X - mean) / dp
for i in xrange(dados.shape[1]):
    dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

# calculamos a matriz de covariância de X
matriz_cov = n.cov(dados, bias=1, rowvar=0)
#print '*** Matriz de covariância:\n', n.around(matriz_cov, decimals=2)

# calculamos os autovetores e autovalores e ordenamos em ordem decresc.
autovalores, autovetores = n.linalg.eig(matriz_cov)
args = n.argsort(autovalores)[::-1]
autovalores = autovalores[args]
autovetores = autovetores[args]
# autovalores (var.) como porcentagem dos autovalores
autovalores_prop = [av/n.sum(autovalores) for av in autovalores]
#print '*** Autovalores (var. %):\n', n.around(autovalores_prop, decimals=2)

# calculamos os componentes principais para todos os dados
dados_finais = n.dot(autovetores.T, dados.T)
#print '*** Dados finais:\n', dados_finais

# calculamos os componentes principais para a submatriz de 7 elementos, as notas originais
principais_orig = n.dot(autovetores.T, dados[:7].T)

# T, P, E = pp.PCA_nipals(dados[:7])
# princ = T[:,:2]
# c1 = P[0]
# c2 = P[1]
# cc1 = c1 / sum(abs(c1)) * 100
# cc2 = c2 / sum(abs(c2)) * 100
# # print 'CONTRIBUICOES'
# # print 'C1', [abs(x) for x in cc1]
# # print 'C2', [abs(x) for x in cc2]
# p.plot(princ[:,0], princ[:,1], 'bo', label='Composers scores')

# T, P, E = pp.PCA_nipals(dados)
# princ = T[:,:2]
# c1 = P[0]
# c2 = P[1]
# cc1 = c1 / sum(abs(c1)) * 100
# cc2 = c2 / sum(abs(c2)) * 100
# # print 'CONTRIBUICOES'
# # print 'C1', [abs(x) for x in cc1]
# # print 'C2', [abs(x) for x in cc2]
# p.plot(princ[:,0], princ[:,1], 'rx', label='Sampled scores')

###
# plotamos os projeções do pca (autovetores * tabela inteira ou subtabela)
dados_finaisT = dados_finais.T
c1 = dados_finaisT[:,0]
c2 = dados_finaisT[:,1]
#print '*** Componentes principais:\n', c1, c2
p.clf()
p.plot(c1, c2, 'rx')

principais_origT = principais_orig.T
c1_orig = principais_origT[:,0]
c2_orig = principais_origT[:,1]
#print '*** Componentes principais:\n', c1_orig, c2_orig
p.plot(c1_orig, c2_orig, 'bo')
[p.text(c1_orig[i], c2_orig[i], str(i+1)) for i in range(len(c1_orig))]
p.show()
#p.savefig('pca_final.eps')

# #print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)

