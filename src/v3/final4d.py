# -*- coding: utf-8 -*-

# esse eh o arquivo final para calculo de todos os resultados do metrics

import numpy as n
import pylab as p
import pca_module as pca

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
# compositores... minimo = 1.0, maximo = 9.0, qtd_aleatorios = 8000000, std = 1.1
minimo = 1.0
maximo = 9.0
qtd_aleatorios = 8000000 # *** change this values to change the distribution
std = 1.1 # *** change this values to change the distribution
# carregamos os dados das notas originais
dados_orig = n.loadtxt(arquivo_notas)
# calculamos os samples por boostrap
ncomp = 7
ncarac = 8
print '*** arquivo_notas:', arquivo_notas, 'qtd_aleatorios:', qtd_aleatorios, 'std:', std
# repetimos ateh qtd_aleatorios
pos=n.zeros(qtd_aleatorios)
pts=n.zeros((qtd_aleatorios,ncarac))

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

print '\n*** Pearson (igual matriz covar.):\n ###TABLE III. Pearson correlation coefficients between the eight musical characteristics.'
for linha in pearson:
    print [str(round(x, ndigits=2)) for x in linha]

# calculamos os autovetores e autovalores e ordenamos em ordem decresc.
autovalores, autovetores = n.linalg.eig(matriz_cov)
args = n.argsort(autovalores)[::-1]
autovalores = autovalores[args]
autovetores = autovetores[args]
# autovalores (var.) como porcentagem dos autovalores
autovalores_prop = [av/n.sum(autovalores) for av in autovalores]
print '\n*** Autovetores (raw):\n', n.around(autovetores, decimals=2)
print '\n*** Autovalores (raw):\n', n.around(autovalores, decimals=2)
print '\n*** Autovalores prop (raw):\n', n.around(autovalores_prop, decimals=2)
print '\n*** Autovalores (var. %) ###TABLE IV### New variances after PCA, in percentages for scores on III.:\n', [x*100 for x in n.around(autovalores_prop, decimals=2)]
# calculamos os componentes principais para todos os dados
dados_finais = n.dot(autovetores.T, dados.T)
print '\n*** Dados finais:\n', dados_finais

# calculamos os componentes principais para a submatriz de 7 elementos, as notas originais
principais_orig = n.dot(autovetores.T, dados[:7].T)

# plotamos os projeções do pca (autovetores * tabela inteira ou subtabela) 
dados_finaisT = dados_finais.T
# só nos interessam os dois primeiros PCAs da tabela de scores (T)
princ = dados_finaisT[:,:8]    # agora nos interessam os 4! ... ou 8
c1 = dados_finaisT[:,0]
c2 = dados_finaisT[:,1]
c3 = dados_finaisT[:,2]
c4 = dados_finaisT[:,3]
print '\n*** Componentes principais [T] (todos):\n', c1, c2, c3, c4
# agora para as notas originais
principais_origT = principais_orig.T
princ_orig = principais_origT[:,:8]    # agora nos interessam os 4! ... ou 8
c1_orig = principais_origT[:,0]
c2_orig = principais_origT[:,1]
c3_orig = principais_origT[:,2]
c4_orig = principais_origT[:,3]
print '\n*** Componentes principais [T] (somente originais):\n', c1_orig, c2_orig, c3_orig, c4_orig

######## plot

# plotamos os PCA (todos em vermelho e originais em azul)
p.clf()
ax = p.gca()
aat = n.zeros(2)
aaf = n.zeros(2)
p.plot(princ[:,0], princ[:,1], '+', alpha=0.5, color="#999999", label="Bootstrap samples")        
for i in xrange(princ_orig.shape[0]):
    cc = n.zeros(3) + float(i) / princ_orig.shape[0]
    x = princ_orig[i, 0]
    y = princ_orig[i, 1]
    aaf = n.sum(princ_orig[:i+1], 0) / (i+1)
    p.plot(aaf[0], aaf[1], 'o', color="#666666")
    if i != 0:
        p.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#333333')
    aat = n.copy(aaf)
    p.plot(x, y, 'bo')
    p.text(x, y, str(i+1), fontsize=12)
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

p.plot(princ_orig[:,0], princ_orig[:,1], color="#000000", label="Original samples")
# [p.text(c1_orig[i], c2_orig[i], str(i+1)) for i in range(len(c1_orig))]
#p.xlim((-5,5))
#p.ylim((-5,5))
p.legend(loc='best')
p.savefig('g1.eps')

p.clf()
ax = p.gca()
aat = n.zeros(2)
aaf = n.zeros(2)
for i in xrange(princ_orig.shape[0]):
    cc = n.zeros(3) + float(i) / princ_orig.shape[0]
    x = princ_orig[i, 0]
    y = princ_orig[i, 1]
    aaf = n.sum(princ_orig[:i+1], 0) / (i+1)
    p.plot(aaf[0], aaf[1], 'o', color="#666666")
    if i != 0:
        p.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#777777')
    aat = n.copy(aaf)
    p.plot(x, y, 'bo')
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

p.plot(princ_orig[:,0], princ_orig[:,1], color="#000000")
# [p.text(c1_orig[i], c2_orig[i], str(i+1)) for i in range(len(c1_orig))]
p.xlim((-5,5))
p.ylim((-5,5))
p.savefig('g1_originais.eps')

# p.clf()
# p.plot(princ_orig[:,0], label='First component')
# p.plot(princ_orig[:,1], label='Second component')
# p.legend(loc='lower right')
# p.plot(princ_orig[:,0],"bo")
# p.plot(princ_orig[:,1],"go")
# p.savefig('g2.eps')


######## plot


# calculamos a contribuição de cada score de T (PCA) para sua formação
# *cálculo dos loadings P
c1 = autovetores[:,0]
c2 = autovetores[:,1]
c3 = autovetores[:,2]
c4 = autovetores[:,3]
c5 = autovetores[:,4]
c6 = autovetores[:,5]
c7 = autovetores[:,6]
c8 = autovetores[:,7]

cc1 = c1 / sum(abs(c1)) * 100
cc2 = c2 / sum(abs(c2)) * 100
cc3 = c3 / sum(abs(c3)) * 100
cc4 = c4 / sum(abs(c4)) * 100
cc5 = c5 / sum(abs(c5)) * 100
cc6 = c6 / sum(abs(c6)) * 100
cc7 = c7 / sum(abs(c7)) * 100
cc8 = c8 / sum(abs(c8)) * 100

print '\n*** Contribuições (autovetores) [P] (idem para todos e originais) ###TABLE VI.### Percentages of the contributions from each musical characteristic on the four new main axes.\n'
print 'C1', [abs(x) for x in cc1]
print 'C2', [abs(x) for x in cc2]
print 'C3', [abs(x) for x in cc3]
print 'C4', [abs(x) for x in cc4]
print 'C5', [abs(x) for x in cc5]
print 'C6', [abs(x) for x in cc6]
print 'C7', [abs(x) for x in cc7]
print 'C8', [abs(x) for x in cc8]

#
# Oposição e Inovação
#
print 'princ_orig', princ_orig
print 'princ', princ
# para todos
oposicao=[]
inovacao=[]
# princ_orig tem os 4 componentes principais
# princ tem apenas os 2 primeiros componentes principais
for i in xrange(1, ncomp):
    a=princ_orig[i-1]    # conforme no artigo... a eh vi
    b=n.sum(princ_orig[:i+1],0)/(i+1) # meio   ... b eh a (average state)
    c=princ_orig[i] # ... c eh um vj

    Di=2*(b-a) # ... Di = 2 * a - vi
    Mij=c-a # ... Mij = vj - vi

    opos=n.sum(Di*Mij)/n.sum(Di**2)  # ... Wij = < Mij , Di > / || Di || ^ 2
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
dialeticas2d=[]
dialeticasTodos=[]
for i in xrange(2, ncomp):
   # eq da reta ab (r1):
   #a=princ_orig[i-2]
   #b=princ_orig[i-1]
   
   # o hiperplano é no formato sum(a**2)+sum(b**2) = sum(2h*(a-b)), h[0-3] são as variáveis
   # precisamos falcular a distância dele à este ponto
   #f=princ_orig[i]

    #       (__)
    #       (oo)                       (__)         *     (__)
    #        \/                        (oo)         |     (oo)
    #    ____| \____            /-------\/    o=o=o=|------\/
    #    ---/   --**           / |       /          |      |
    # *____/    |___//        *  ||----||           ||----||
    #     //--------/            ~~    ~~           ~~    ~~
    #    //__                      Cow           Cow pooing
    #    Cow marching            standing

   # a: v1
   # b: v2
   # c: v3
   # H: hiperplano "bissetriz" à a e b: H = a1*x1 + a2*x2 + a3*x3 + a4*x4 = b
   # w: vetor normal à H: w = (a1, a2, a3, a4) localizado entre a e b
   # distância de c à H: dist = sum(w*c) / sqrt(sum(w**2))
   # ou seja... dist = (a1*c1 + a2*c2 + a3*c3 + a4*c4) / sqrt(a1**2 + a2**2 + a3**3)
   a=princ_orig[i-2] # thesis
   b=princ_orig[i-1] # antithesis
   c=princ_orig[i]   # synthesis
   # dialetica para 4d
   dist = n.abs( (b[0]-a[0])*c[0] + (b[1]-a[1])*c[1] + (b[2]-a[2])*c[2] + (b[3]-a[3])*c[3] +
                 (-((b[0]**2 - a[0]**2)/2)
                  -((b[1]**2 - a[1]**2)/2)
                  -((b[2]**2 - a[2]**2)/2)
                  -((b[3]**2 - a[3]**2)/2)) ) / n.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2 + (b[3]-a[3])**2 )

   # dialetica para 2d
   dist2d = n.abs( (b[0]-a[0])*c[0] + (b[1]-a[1])*c[1] +
                 (-((b[0]**2 - a[0]**2)/2)
                  -((b[1]**2 - a[1]**2)/2)) ) / n.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2)

   # dialetica para 8d
   distTodos = n.abs( (b[0]-a[0])*c[0] + (b[1]-a[1])*c[1] + (b[2]-a[2])*c[2] + (b[3]-a[3])*c[3] + (b[4]-a[4])*c[4] + (b[5]-a[5])*c[5] + (b[6]-a[6])*c[6] + (b[7]-a[7])*c[7] +
                 (-((b[0]**2 - a[0]**2)/2)
                  -((b[1]**2 - a[1]**2)/2)
                  -((b[2]**2 - a[2]**2)/2)
                  -((b[3]**2 - a[3]**2)/2)
                  -((b[4]**2 - a[4]**2)/2)
                  -((b[5]**2 - a[5]**2)/2)
                  -((b[6]**2 - a[6]**2)/2)
                  -((b[7]**2 - a[7]**2)/2))) / n.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2 + (b[3]-a[3])**2 + (b[4]-a[4])**2 + (b[5]-a[5])**2 + (b[6]-a[6])**2 + (b[7]-a[7])**2)


   
   #w = n.sum(princ_orig[:i-1],0)/(i-1) # meio
   #w=a+(a+b)/2
   #print 'a', a, 'b', b, 'c', c, 'w', w
   #print '---'
   #dist = n.sum(w*c) / n.sqrt(n.sum(w**2))
   #dist = n.dot(-w,c) / n.sqrt(n.sum(w**2))

   # isso o greenkobold fez: e mais um pouco alí de cima...

   # generalizando a fórmula em http://mathworld.wolfram.com/Point-PlaneDistance.html
   # para 4D:
   # d = n.sum(a**2)+n.sum(b**2)
   # a,b,c = 2*(a-b)
   # portanto:
   #dist =  n.nan_to_num((n.sum(2*(a-b)*f + a**2 + b**2)) / (n.sum(2*(a-b))**.5))
   
   
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
   #d2=gama3*d1+ne-0ta3

   ## distancia entre f e d
   #dist=n.sqrt((f[0]-d1)**2+(f[1]-d2)**2)
   dialeticas.append(dist)
   dialeticas2d.append(dist2d)
   dialeticasTodos.append(distTodos)

print '\n###TABLE VII. TABLE VIII###.\n'
print '\n*** Oposição:\n', oposicao
print '\n*** Inovação:\n', inovacao
print '\n*** Dialéticas:\n', dialeticas
print '\n*** Dialéticas 2d:\n', dialeticas2d
print '\n*** Dialéticas Todos (8d):\n', dialeticasTodos
#dialeticas = n.array(n.abs(dialeticas))
#print ( (dialeticas-dialeticas.min())/(dialeticas.max()-dialeticas.min()) )

#
# Perturbação
#

nperturb = 1000
# distancias[original, ruido, amostra]
distancias = n.zeros((ncomp, ncomp, nperturb))
autovals = n.zeros((nperturb, 8))  # agora para 8d
princ_orig = princ_orig[:,:8]
princ = princ[:,:8]

for foobar in xrange(nperturb):
    dist = n.random.randint(-2, 3, copia_dados.shape)
    copia_dados += dist

    for i in xrange(copia_dados.shape[1]):
        copia_dados[:,i] = (copia_dados[:,i] - copia_dados[:,i].mean())/copia_dados[:,i].std()

    # fazemos pca para dados considerando esses pontos aleatórios entre -2 e 2
    # FIXME: substituir depois pca_nipals
    T, P, E = pca.PCA_nipals(copia_dados)
    autovals[foobar] = E[:8]
    princ = T[:,:8]
    for i in xrange(ncomp):
        for j in xrange(ncomp):
            distancias[i, j, foobar] = n.sum((princ_orig[i] - princ[j])**2)**.5

stds = n.zeros((ncomp, ncomp))
means = n.zeros((ncomp, ncomp))
main_stds = []
main_means = []
print 'dados', copia_dados
for i in xrange(ncomp):
    for j in xrange(ncomp):
        stds[i,j] = distancias[i,j,:].std()
        means[i,j] = distancias[i,j,:].mean()
        if i == j:
          main_stds.append(stds[i,j])
          main_means.append(means[i,j])
n.savetxt("mean2_.txt",means,"%.2e")
n.savetxt("stds2_.txt",stds,"%.2e")

print '###TABLE V.### Average and standard deviation of the deviations for each composer and for the 8 eigenvalues.'

print 'main_means', main_means
print 'main_stds', main_stds

# Cálculo das médias e variâncias dos desvios dos primeiros 4 autovalores

deltas = autovals - autovalores_prop[:8]
medias = deltas.mean(0)
desvios = deltas.std(0)
print 'eigenvalues means', medias
print 'eigenvalues stds', desvios

