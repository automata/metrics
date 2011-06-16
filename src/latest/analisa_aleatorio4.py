# -*- coding: utf-8 -*-
import numpy as n, pca_module as pca, pylab as p, scipy.stats as stats, random as r, scipy.stats.stats as sss

#
# Funções auxiliares
#

# distribuição normal, média, dp
def pinned_gauss(a, b, mu, sigma):
    return min(b, max(a, r.gauss(mu, sigma)))

def truncated_gauss(a, b, mu, sigma):
    while True:
        n = r.gauss(mu, sigma)
        if a <= n <= b:
            return n

#
# Análise
#

agentes_todos = [7,10,15,20,50,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000]
_somas = []
_zscores = []

#for repete in range(1):
for num_agentes in agentes_todos[:1]:
    #print '\nTESTE %s' % repete
    print '\nQTD AGENTES', num_agentes
    # número de agentes (medidas)
    _na = num_agentes
    # número de características (vars. aleatórias)
    _nc = 8

    agentes = ['Plato', 'Aristotle', 'Descartes', 'Espinoza', 'Kant', 'Nietzsche', 'Deleuze']
    caracteristicas = ['R-E', 'E-E', 'M-D', 'T-A', 'H-R', 'D-P', 'D-F', 'N-M']

    #agentes = ['A%i' % i for i in range(1, _na+1)]
    #caracteristicas = ['Feature %i' % i for i in range(1, _nc+1)]

    # matriz de notas
    #nn = n.loadtxt('notas_compositores.txt')
    nn = n.loadtxt('notas_filosofos.txt')
    #nn = n.loadtxt('notas_aleatorias.txt')
    #nn = n.array([[r.uniform(1,9) for x in range(_nc)] for y in range(_na)])
    #nn_lista = nn.tolist()
    #print 'QTD NOTAS'
    #print [sum([[int(round(z)) for z in x].count(y) for x in nn_lista]) for y in range(1,10)]

    print '\nNOTAS'
    for i in range(len(nn)):
        print '%s & %s \\' % (agentes[i], ' & '.join([str(x) for x in nn[i]]))

    print 'MEDIA', n.mean(nn)

    print '\nZ-SCORES DAS NOTAS'
    _zs1 = sss.zscore(nn)
    for i in range(len(_zs1)):
        print [round(x, ndigits=2) for x in _zs1[i]]
    print 'MEDIA', n.mean(n.abs(_zs1))

    # cálculo da matriz de correlação
    # pré-processamento
    #for i in xrange(nn.shape[1]):
    #    nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

    # pearson
    print '\nMATRIZ DE COVARIANCIA'
    covm = n.cov(nn.T, bias=1)
    for i in range(len(covm)):
        print [round(x, ndigits=2) for x in covm[i]]
    print 'MEDIA', n.mean(n.abs(covm))

    print '\nZ-SCORES DA MATRIZ DE COV.'
    _zs2 = sss.zscore(covm)
    for i in range(len(_zs2)):
        print [round(x, ndigits=2) for x in _zs2[i]]
    print 'MEDIA', n.mean(n.abs(_zs2))

    def _cov(x, y):
        s = 0
        mean_x = sum(x)/len(x)
        mean_y = sum(y)/len(y)
        
        for i in range(len(x)):
            s += (x[i] - mean_x)*(y[i] - mean_y)

        return s/len(x)

    def _var(x):
        s = 0
        mean_x = sum(x)/len(x)
        
        for i in range(len(x)):
            s += (x[i] - mean_x)**2

        return s/len(x)

    def _cov_matrix(X):
        C = n.zeros((len(X[0]), len(X[0])))
        XT = n.transpose(X)
        for i in range(len(X[0])):
            for j in range(len(X[0])):
                if i==j:
                    C[i,i] = _cov(XT[i], XT[i])
                else:
                    C[i,j] = _cov(XT[i], XT[j])
        return C

    # print 'MINHA MATRIZ COV'
    # minha_matriz_cov = _cov_matrix(nn)
    # print minha_matriz_cov
    
    def _pearson(X):
        P = n.zeros((len(X[0]), len(X[0])))
        XT = n.transpose(X)
        for i in range(len(X[0])):
            for j in range(len(X[0])):
                P[i,j] = round(minha_matriz_cov[i,j] / n.sqrt(_var(XT[i])*_var(XT[j])), ndigits=2)
        return P

    print '\nDPs'
    stds=n.std(nn,0)
    print [round(x, ndigits=2) for x in stds]

    # print 'MEUS DPs'
    # _dps = []
    # for i in range(_nc):
    #     _dps.append(n.sqrt(_var(nn.T[i])))
    # print _dps

    print '\nPEARSON'
    pearson=n.zeros((_nc,_nc))
    for i in xrange(_nc):
        for j in xrange(_nc):
            pearson[i,j]=round(covm[i,j]/(stds[i]*stds[j]), ndigits=2)
    print pearson
    print 'MEDIA', n.mean(n.abs(pearson))

    print '\nZ-SCORES DE PEARSON'
    _zs = sss.zscore(pearson)
    for i in range(len(_zs)):
        print [round(x, ndigits=2) for x in _zs[i]]
    _zscores.append(_zs)
    print 'MEDIA', n.mean(n.abs(_zs))

    # print 'MEU PEARSON'
    # meu_pearson = _pearson(nn)
    # print meu_pearson

    # cálculo PCA
    T, P, E = pca.PCA_nipals(nn)
    princ = T[:,:2]

    # cálculo dos autovalores %
    #print T
    print '\nAUTOVALORES', [round(x, ndigits=2) for x in E * 100], 'SOMA', sum(E)
    print '\nSOMA DOIS PRIMEIROS', round(E[0] + E[1], ndigits=2)
    _somas.append(round(E[0] + E[1], ndigits = 2))

    #print 'NOVO PCA'
    # def pca(data, nRedDim=0, normalise=1):
    #     # centre data
    #     m = n.mean(data, axis=0)
    #     data -= m

    #     # covariance matrix
    #     C = n.cov(n.transpose(data))

    #     # compute eigenvalues and sort into descending order
    #     evals, evecs = n.linalg.eig(C)
    #     indices = n.argsort(evals)
    #     indices = indices[::-1]
    #     evecs = evecs[:,indices]
    #     evals = evals[indices]

    #     if nRedDim > 0:
    #         evecs = evecs[:,:nRedDim]

    #     if normalise:
    #         for i in range(n.shape(evecs)[1]):
    #             evecs[:,i] / n.linalg.norm(evecs[:,i]) * n.sqrt(evals[i])

    #     # produce the new data matrix
    #     x = n.dot(n.transpose(evecs), n.transpose(data))

    #     # compute the original data again
    #     y = n.transpose(n.dot(evecs, x)) + m

    #     return x, y, evals, evecs

    # x, y, evals, evecs = pca(nn, normalise=1)

    # print 'new data matrix'
    # print x
    # print 'original data matrix'
    # print y
    #print 'evals', evals, sum(evals)
    #print 'evecs', evecs
    
#print agentes_todos, _somas
#p.plot(agentes_todos, _somas)
#p.savefig('aleats_2.eps')

###############################################################################
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

