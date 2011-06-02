# -*- coding: utf-8 -*-
import numpy as n, pca_module as pca, pylab as p, scipy.stats as stats, random

_na = 7
_nc = 8

agentes = ['A%i' % i for i in range(1, _na+1)]
caracteristicas = ['Feature %i' % i for i in range(1, _nc+1)]

# leitura da matriz de notas
#nn = n.loadtxt('notas.txt')
#nn = n.loadtxt('notas_filosofos.txt')
baz_max = 100
s = 0
for baz in range(baz_max):
    nn = n.array([[random.uniform(1,9) for x in range(_nc)] for y in range(_na)])

    #for i in range(len(nn)):
    #print '%s & %s \\' % (agentes[i], ' & '.join([str(x) for x in nn[i]]))
    
    # cálculo da matriz de correlação
    # pré-processamento
    for i in xrange(nn.shape[1]):
        nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

#coeficientes de pearson
    covm=n.cov(nn.T,bias=True)
    stds=n.std(nn,0)
    pearson=n.zeros((_nc,_nc))
    for i in xrange(_nc):
        for j in xrange(_nc):
            pearson[i,j]=covm[i,j]/(stds[i]*stds[j])

    m = []
    colunas = [[x[i] for x in nn] for i in range(_nc)]
    correlacao = [stats.pearsonr(n.array(col), n.array(coluna))[0]
                  for col in colunas
                  for coluna in colunas]
    for i in range(_nc):
        m.append(correlacao[i*_nc : i*_nc + _nc])
# pearson == m

#print 'PEARSON'

#for linha in m:
#    print [str(round(x, ndigits=2)) for x in linha]


# cálculo PCA
    T, P, E = pca.PCA_nipals(nn)
    princ = T[:,:2]

# cálculo dos autovalores %
    print 'Autovalores:', E * 100, '\nSoma 2 primeiros:', E[0] + E[1]
    s = s + (E[0] + E[1])
print 'Média:', s/baz_max
"""
# contribuições
c1 = P[0]
c2 = P[1]
cc1 = c1 / sum(abs(c1)) * 100
cc2 = c2 / sum(abs(c2)) * 100
print 'CONTRIBUICOES'
print 'C1', [abs(x) for x in cc1]
print 'C2', [abs(x) for x in cc2]

# oposições/inovação
oposicao=[]
inovacao=[]
for i in xrange(1,len(agentes)):
   a=princ[i-1]
   b=n.sum(princ[:i+1],0)/(i+1) #meio
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

# dialética
dialeticas=[]
for i in xrange(2,len(agentes)):
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
   dialeticas.append(dist/ n.sqrt(  (a[0]-b[0])**2+(a[1]-b[1])**2    ))

print 'OPOSICAO', oposicao
print 'INOVACAO', inovacao
print 'DIALÉTICAS', dialeticas

# gráficos
ax = p.gca()
aat = n.zeros(2)
aaf = n.zeros(2)
for i in xrange(princ.shape[0]):
    cc = n.zeros(3) + float(i) / princ.shape[0]
    x = princ[i, 0]
    y = princ[i, 1]
    aaf = n.sum(princ[:i+1], 0) / (i+1)
    p.plot(aaf[0], aaf[1], 'ro')
    if i != 0:
        p.plot((aat[0], aaf[0]), (aat[1], aaf[1]), '--')
    aat = n.copy(aaf)
    p.plot(x, y, 'o', color=cc)

    p.text(x, y, str(i+1) + ' ' + agentes[i], fontsize=12)

    if i != princ.shape[0] - 1:
        hw = .15
        h = .25
        dx = -(x - princ[i+1, 0])
        dy = -(y - princ[i+1, 1])
        # if i in [0,2,3]:
        #     dx-=n.sign(dx)*h*h
        #     dy-=n.sign(dy)*h*h
        # else:
        #     dx-=n.sign(dx)*h*h
        #     dy-=n.sign(dy)*h*(dy/dx)
        #arr = p.arrow(x, y, dx - n.sign(dx)*h, dy - n.sign(dy)*h, width=.015, color='grey', head_width=hw)
        #ax.add_patch(arr)
    
p.plot(princ[:,0], princ[:,1], color="black")
p.xlim((-4,4))
p.ylim((-4,4))
p.savefig('g1.eps')
p.clf()

p.plot(princ[:,0], label='First component')
p.plot(princ[:,1], label='Second component')
p.legend(loc='lower right')
p.plot(princ[:,0],"bo")
p.plot(princ[:,1],"go")
p.savefig('g2.eps')
"""
"""
######## PERTURBACAO

E_or=n.copy(E)
NN=1000
numeroDeAgentes=nn.shape[0]
# distancias[original, ruido, amostra]
distancias=n.zeros((numeroDeAgentes,numeroDeAgentes,NN))
autovals=n.zeros((NN,4))
princ_or=T[:,:2]

for foobar in xrange(NN):
    #nn = n.loadtxt('notas.txt')

    ############
    # PERTURBACAO
    #STD=1
    #nn+=n.random.normal(0,STD,nn.shape)
    dist=n.random.randint(-2,3,nn.shape)
    nn+=dist
    ############


    for i in xrange(nn.shape[1]):
      nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

    T,P,E=pca.PCA_nipals(nn)
    autovals[foobar]=E[:4]
    princ=T[:,:2]
    for i in xrange(numeroDeAgentes):
        for j in xrange(numeroDeAgentes):
            distancias[i,j,foobar]= n.sum((princ_or[i]-princ[j])**2)**.5

stds=n.zeros((numeroDeAgentes,numeroDeAgentes))
means=n.zeros((numeroDeAgentes,numeroDeAgentes))
main_stds = []
main_means = []
for i in xrange(numeroDeAgentes):
    for j in xrange(numeroDeAgentes):
        stds[i,j]=distancias[i,j,:].std()
        means[i,j]=distancias[i,j,:].mean()
        if i == j:
          main_stds.append(stds[i,j])
          main_means.append(means[i,j])
n.savetxt("mean2_.txt",means,"%.2e")
n.savetxt("stds2_.txt",stds,"%.2e")
print 'main_means', main_means
print 'main_stds', main_stds

# Cálculo das médias e variâncias dos desvios dos primeiros 4 autovalores
deltas=autovals - E_or[:4]
medias=deltas.mean(0)
desvios=deltas.std(0)
print 'eigenvalues means', medias
print 'eigenvalues stds', desvios
"""

