# -*- coding: utf-8 -*-
import numpy as n, pca_module as pca, pylab as p, scipy.stats as stats

compositores = ['Bach', 'Mozart', 'Beethoven', 'Chopin', 'Brahms', 'Stravinsky', 'Stockhausen']
caracteristicas = ['S-P', 'S-L', 'H-C', 'V-I', 'N-D', 'M-V', 'R-P', 'T-M']

# leitura da matriz de notas
nn = n.loadtxt('notas2.txt')

for i in range(len(nn)):
    print '%s & %s \\' % (compositores[i], ' & '.join([str(x) for x in nn[i]]))

# cálculo da matriz de correlação
# pré-processamento
for i in xrange(nn.shape[1]):
    nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

#coeficientes de pearson
covm=n.cov(nn.T,bias=True)
stds=n.std(nn,0)
pearson=n.zeros((8,8))
for i in xrange(8):
   for j in xrange(8):
     pearson[i,j]=covm[i,j]/(stds[i]*stds[j])

m = []
colunas = [[x[i] for x in nn] for i in range(8)]
correlacao = [stats.pearsonr(n.array(col), n.array(coluna))[0]
              for col in colunas
              for coluna in colunas]
for i in range(8):
    m.append(correlacao[i*8 : i*8 + 8])
# pearson == m

print 'PEARSON'

for linha in m:
    print [str(round(x, ndigits=2)) for x in linha]


# cálculo PCA
T, P, E = pca.PCA_nipals(nn)
princ = T[:,:2]

# cálculo dos autovalores %
print 'AUTOVALORES', E * 100

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
for i in xrange(1,7):
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
for i in xrange(2,7):
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
    if i == 0:
        p.text(x, y-.4, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 1:
        p.text(x, y+.2, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 2:
        p.text(x, y-.4, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 3:
        p.text(x, y+.2, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 4:
        p.text(x, y+.3, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 5:
        p.text(x, y-.45, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 6:
        p.text(x, y-.4, str(i) + ' ' + compositores[i], fontsize=12)
    elif i == 7:
        p.text(x, y-.4, str(i) + ' ' + compositores[i], fontsize=12)

    if i != princ.shape[0] - 1:
        hw = .15
        h = .25
        dx = -(x - princ[i+1, 0])
        dy = -(y - princ[i+1, 1])
        if i == 1:
            dx+=dx*h*h
            dy-=dy*h*h
        else:
            dx-=n.sign(dx)*h
            dy-=n.sign(dy)*h*(dy/dx)
        arr = p.arrow(x, y, dx, dy, width=.015, color='grey', head_width=hw)
        #ax.add_patch(arr)
    
#p.plot(princ[:,0], princ[:,1])
p.xlim((-5,4))
p.ylim((-5,4))
p.savefig('g1.eps')
p.clf()

p.plot(princ[:,0], label='First component')
p.plot(princ[:,1], label='Second component')
p.legend(loc='lower right')
p.plot(princ[:,0],"bo")
p.plot(princ[:,1],"go")
p.savefig('g2.eps')

