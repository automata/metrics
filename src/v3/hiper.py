# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
import pca_module as pca

princ = n.array([1,2,3,4])

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
