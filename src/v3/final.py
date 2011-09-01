# -*- coding: utf-8 -*-
import numpy as n
import pylab as p

#arquivo_notas = 'notas_compositores.txt'
#agents = ['Monteverdi', 'Bach', 'Mozart', 'Beethoven', 'Brahms', 'Stravinsky', 'Stockhausen']
#caracs = ['S-P', 'S-L', 'H-C', 'V-I', 'N-D', 'M-V', 'R-P', 'T-M']

arquivo_notas = 'notas_diretores.txt'
agents = ['Griffith','Eisenstein','Hichcock','Welles','Felini','Kubrick','Spielberg']
caracs = ['I-F','F-R','F-H','L-D','H-A','R-A','T-M','A-M']

# parâmetros
minimo = 1.0
maximo = 9.0
qtd_aleatorios = 500000 # ***
std = 1.3 # ***
# carregamos os dados das notas originais
dados_orig = n.loadtxt(arquivo_notas)
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
    # calculamos a probabilidade considerando todas as distâncias
    po = n.sum(n.exp (-0.5*(dk/std)**2) )
    # guardamos a probabilidade junto com as demais
    pos[i]=po

# normalizamos as probabilidades
pos=( (pos-pos.min())/(pos.max()-pos.min()) )*.2
# jogamos moedas entre [0,1) (Monte Carlo)
coin=n.random.uniform(size=qtd_aleatorios)
# comparamos as moedas com as probabilidades
results= coin < pos
# filtramos, ficando somente com os pontos que respeitaram a inequação acima
pts_reais=pts[results]
# unimos as duas matrizes de notas
dados=n.vstack((dados_orig,pts_reais))
print '\n*** Dimensões da tabela de notas:', dados.shape
print '\n*** Tabela geral de notas:\n', dados

#
# PCA
#

# normalizamos a matriz de dados (X = X - mean) e dividimos pelo d.p.
#     (X = (X - mean) / dp
for i in xrange(dados.shape[1]):
    dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

# calculamos a matriz de covariância de X
matriz_cov = n.cov(dados, bias=1, rowvar=0)
print '\n*** Matriz de covariância:\n', n.around(matriz_cov, decimals=2)

# calculamos a correlação de pearson para todas as notas
stds=n.std(dados, 0)
pearson=n.zeros((ncarac,ncarac))
for i in xrange(ncarac):
   for j in xrange(ncarac):
     pearson[i,j]=matriz_cov[i,j]/(stds[i]*stds[j])

print '\n*** Pearson (igual matriz covar.):\n'
for linha in pearson:
    print [str(round(x, ndigits=2)) for x in linha]

# calculamos os autovetores e autovalores e ordenamos em ordem decresc.
autovalores, autovetores = n.linalg.eig(matriz_cov)
args = n.argsort(autovalores)[::-1]
autovalores = autovalores[args]
autovetores = autovetores[args]
# autovalores (var.) como porcentagem dos autovalores
autovalores_prop = [av/n.sum(autovalores) for av in autovalores]
print '\n*** Autovalores (var. %):\n', [x*100 for x in n.around(autovalores_prop, decimals=2)]
# calculamos os componentes principais para todos os dados
dados_finais = n.dot(autovetores.T, dados.T)
print '\n*** Dados finais:\n', dados_finais

# calculamos os componentes principais para a submatriz de 7 elementos, as notas originais
principais_orig = n.dot(autovetores.T, dados[:7].T)

# plotamos os projeções do pca (autovetores * tabela inteira ou subtabela) 
dados_finaisT = dados_finais.T
# só nos interessam os dois primeiros PCAs da tabela de scores (T)
princ = dados_finaisT[:,:2]
c1 = dados_finaisT[:,0]
c2 = dados_finaisT[:,1]
print '\n*** Componentes principais [T] (todos):\n', c1, c2
# agora para as notas originais
principais_origT = principais_orig.T
princ_orig = principais_origT[:,:2]
c1_orig = principais_origT[:,0]
c2_orig = principais_origT[:,1]
print '\n*** Componentes principais [T] (somente originais):\n', c1_orig, c2_orig

# plotamos os PCA (todos em vermelho e originais em azul)
p.clf()
ax = p.gca()
aat = n.zeros(2)
aaf = n.zeros(2)
for i in xrange(princ_orig.shape[0]):
    cc = n.zeros(3) + float(i) / princ_orig.shape[0]
    x = princ_orig[i, 0]
    y = princ_orig[i, 1]
    aaf = n.sum(princ_orig[:i+1], 0) / (i+1)
    p.plot(aaf[0], aaf[1], 'ro')
    if i != 0:
        p.plot((aat[0], aaf[0]), (aat[1], aaf[1]), '--')
    aat = n.copy(aaf)
    p.plot(x, y, 'o', color=cc)
    p.text(x, y, str(i+1) + ' ' + agents[i], fontsize=12)
    # if i == 0:
    #     p.text(x-.7, y-.4, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 1:
    #     p.text(x-.7, y+.2, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 2:
    #     p.text(x, y+.2, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 3:
    #     p.text(x-.5, y-.5, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 5:
    #     p.text(x+.2, y-.3, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 6:
    #     p.text(x-1.5, y+.15, str(i+1) + ' ' + compositores[i], fontsize=12)
    # elif i == 7:
    #     p.text(x-.5, y+.4, str(i+1) + ' ' + compositores[i], fontsize=12)

p.plot(princ[:,0], princ[:,1], 'rx')        
p.plot(princ_orig[:,0], princ_orig[:,1], color="gray")
# [p.text(c1_orig[i], c2_orig[i], str(i+1)) for i in range(len(c1_orig))]
p.xlim((-5,4))
p.ylim((-3,4))
p.savefig('g1.eps')

p.clf()
p.plot(princ[:,0], label='First component')
p.plot(princ[:,1], label='Second component')
p.legend(loc='lower right')
p.plot(princ[:,0],"bo")
p.plot(princ[:,1],"go")
p.savefig('g2.eps')

# calculamos a contribuição de cada score de T (PCA) para sua formação
# *cálculo dos loadings P
c1 = autovetores[:,0]
c2 = autovetores[:,1]
cc1 = c1 / sum(abs(c1)) * 100
cc2 = c2 / sum(abs(c2)) * 100
print '\n*** Contribuições (autovetores) [P] (idem para todos e originais)\n'
print 'C1', [abs(x) for x in cc1]
print 'C2', [abs(x) for x in cc2]

#
# Oposição e Inovação
#

# para todos
oposicao=[]
inovacao=[]
for i in xrange(1, ncomp):
   a=princ[i-1]
   b=n.sum(princ[:i+1],0)/(i+1) # meio
   c=princ[i]

   Di=2*(b-a)
   Mij=c-a

   opos=n.sum(Di*Mij)/n.sum(Di**2)
   oposicao.append(opos)

   # reta ab (r1)
   gama=(b[1]-a[1])/(b[0]-a[0])
   neta=a[1]-gama*a[0]

   # reta perpendicular que passa por c
   gama2=-1./gama
   neta2=c[1]-gama2*c[0]

   # ponto de intersecção d
   d1=(neta-neta2)/(gama2-gama)
   d2=gama2*d1+neta2

   # distancia entre d e c
   dist=n.sqrt((c[0]-d1)**2+(c[1]-d2)**2)
   inovacao.append(dist)

# FIXME: para originais somente

#
# Dialética
#

dialeticas=[]
for i in xrange(2, ncomp):
   # eq da reta ab (r1):
   a=princ[i-2]
   b=princ[i-1]
   gama=(b[1]-a[1])/(b[0]-a[0])
   neta=a[1]-gama*a[0]

   # reta perpendicular que passa pelo meio (r2)
   c=(a+b)/2 # meio
   gama2=-1/gama
   neta2=c[1]-gama2*c[0]

   # reta // à ab (r3), passando por v_3=f
   f=princ[i]
   gama3=gama
   neta3=f[1]-gama3*f[0]

   # d em r2xr3
   d1=(neta2-neta3)/(gama3-gama2)
   d2=gama3*d1+neta3

   # distancia entre f e d
   dist=n.sqrt((f[0]-d1)**2+(f[1]-d2)**2)
   dialeticas.append(dist/ n.sqrt( (a[0]-b[0])**2+(a[1]-b[1])**2 ))

print '\n*** Oposição:\n', oposicao
print '\n*** Inovação:\n', inovacao
print '\n*** Dialéticas:\n', dialeticas

#
# Perturbação
#
