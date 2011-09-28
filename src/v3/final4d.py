# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
#import pca_module as pca

arquivo_notas = 'notas_compositores.txt'
agents = ['Monteverdi', 'Bach', 'Mozart', 'Beethoven', 'Brahms', 'Stravinsky', 'Stockhausen']
caracs = ['S-P', 'S-L', 'H-C', 'V-I', 'N-D', 'M-V', 'R-P', 'T-M']

# arquivo_notas = 'notas_filosofos.txt'
# agents = ['Plato', 'Aristotle', 'Descartes', 'Espinoza', 'Kant', 'Nietzsche', 'Deleuze']
# caracs = ['R-E', 'E-E', 'M-D', 'T-A', 'H-R', 'D-P', 'D-F', 'N-M']

# arquivo_notas = 'notas_diretores.txt'
# agents = ['Griffith','Eisenstein','Hichcock','Welles','Felini','Kubrick','Spielberg']
# caracs = ['I-F','F-R','F-H','L-D','H-A','R-A','T-M','A-M']

# parâmetros
minimo = 1.0
maximo = 9.0
qtd_aleatorios = 2000000 # ***
std = 1.4 # ***
# carregamos os dados das notas originais
dados_orig = n.loadtxt(arquivo_notas)
# calculamos os samples por boostrap
ncomp = 7
ncarac = 8

# repetimos até termos a quantidade que queremos de aleatorios
#while (len(dados_boot) < qtd_aleatorios/7):
pos=n.zeros(qtd_aleatorios)
pts=n.zeros((qtd_aleatorios,ncarac))

def seleciona(x):
    if (x < 1):
        return 1.0
    if (x > 9):
        return 9.0
    return x

# para cada ponto artificial
for i in xrange(qtd_aleatorios):
    xs = n.random.uniform(1, 9.1, 8)
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
copia_dados = n.copy(dados)
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
princ = dados_finaisT[:,:4]
c1 = dados_finaisT[:,0]
c2 = dados_finaisT[:,1]
c3 = dados_finaisT[:,2]
c4 = dados_finaisT[:,3]
print '\n*** Componentes principais [T] (todos):\n', c1, c2, c3, c4
# agora para as notas originais
principais_origT = principais_orig.T
princ_orig = principais_origT[:,:4]
c1_orig = principais_origT[:,0]
c2_orig = principais_origT[:,1]
c1_orig = principais_origT[:,2]
c2_orig = principais_origT[:,3]
print '\n*** Componentes principais [T] (somente originais):\n', c1_orig, c2_orig

# calculamos a contribuição de cada score de T (PCA) para sua formação
# *cálculo dos loadings P
c1 = autovetores[:,0]
c2 = autovetores[:,1]
c3 = autovetores[:,2]
c4 = autovetores[:,3]
cc1 = c1 / sum(abs(c1)) * 100
cc2 = c2 / sum(abs(c2)) * 100
cc3 = c3 / sum(abs(c3)) * 100
cc4 = c4 / sum(abs(c4)) * 100
print '\n*** Contribuições (autovetores) [P] (idem para todos e originais)\n'
print 'C1', [abs(x) for x in cc1]
print 'C2', [abs(x) for x in cc2]
print 'C3', [abs(x) for x in cc3]
print 'C4', [abs(x) for x in cc4]

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

    ########## Cálculo de inovação ##################
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    inov=n.sqrt(  ( n.sum((a-c)**2)*n.sum((b-a)**2) - n.sum( (a-c)*(b-a) )**2 )/n.sum((b-a)**2)  )
    
    #l = n.sum((a-c)**2)**.5 # distância entre a e c

    #lx = l/(c[0]-a[0]); lx2=lx**2 
    #ly = l/(c[1]-a[1]); ly2=ly**2
    #lz = l/(c[2]-a[2]); lz2=lz**2
    #lw = l/(c[3]-a[3]); lw2=lw**2
    #li = l/(c-a) # inclinações relativas aos eixos

    #x = (c[0]+lx2*a[0])/(1+lx2)
    #y = (c[1]+ly2*a[1])/(1+ly2)
    #z = (c[2]+lz2*a[2])/(1+lz2)
    #w = (c[3]+lw2*a[3])/(1+lw2)
    #pt=(c+li2*a)/(1+li**2) # ponto da reta 

    #inov = n.sum((pt-c)**2)**.5

    inovacao.append(inov)

# FIXME: para originais somente

#
# Dialética
#

dialeticas=[]
for i in xrange(2, ncomp):
   # eq da reta ab (r1):
   a=princ[i-2]
   b=princ[i-1]
   
   # o hiperplano é no formato sum(a**2)+sum(b**2) = sum(2h*(a-b)), h[0-3] são as variáveis
   # precisamos falcular a distância dele à este ponto
   f=princ[i]
   
   # generalizando a fórmula em http://mathworld.wolfram.com/Point-PlaneDistance.html
   # para 4D:
   # d = n.sum(a**2)+n.sum(b**2)
   # a,b,c = 2*(a-b)
   # portanto:
   dist =  (n.sum(2*(a-b)*f + a**2 + b**2)) / n.sum(2*(a-b))**.5
   
   
   #gama=(b[1]-a[1])/(b[0]-a[0])
   #neta=a[1]-gama*a[0]

   ## reta perpendicular que passa pelo meio (r2)
   #c=(a+b)/2 # meio
   #gama2=-1/gama
   #neta2=c[1]-gama2*c[0]

   ## reta // à ab (r3), passando por v_3=f
   #f=princ[i]
   #gama3=gama
   #neta3=f[1]-gama3*f[0]

   ## d em r2xr3
   #d1=(neta2-neta3)/(gama3-gama2)
   #d2=gama3*d1+neta3

   ## distancia entre f e d
   #dist=n.sqrt((f[0]-d1)**2+(f[1]-d2)**2)
   dialeticas.append(dist)

print '\n*** Oposição:\n', oposicao
print '\n*** Inovação:\n', inovacao
print '\n*** Dialéticas:\n', dialeticas