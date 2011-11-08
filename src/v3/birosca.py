# -*- coding: utf-8 -*-
import numpy as n
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

# v1: (3,6,2), v2: (9,3,2)

v1 = n.array([3.,6.,1.])
v2 = n.array([8.,3.,1.])
meio = (v1+v2)/2

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
#d = n.dot(-ponto, normal)
meio = n.array([0.,0.,0.])
normal = n.array([-4.,1.,1.])
d = n.dot(-meio, normal)

# equacao parametrica (o que importa eh z. x e y sao variaveis livres)
# http://www.mat.ufmg.br/gaal/aulas_online/at4_03.html
x, y = n.meshgrid(n.arange(-10,10,1), n.arange(-10,10,1))
z = (-normal[0]*x - normal[1]*y - d)/normal[2]

# plota em 3D
fig = p.figure()
ax = p3.Axes3D(fig)
#ax.plot3D([v1[0]], [v1[1]], [v1[2]],'bo')
#ax.plot3D([v2[0]], [v2[1]], [v2[2]],'ro')
ax.plot3D([meio[0]], [meio[1]], [meio[2]],'go')
ax.plot3D([normal[0]], [normal[1]], [normal[2]],'y+')
ax.plot_wireframe(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()

#p.plot(x,y,'ro')
#p.plot(meio[0], meio[1], 'bo')
#p.xlim([0,10])
#p.ylim([0,10])
#p.show()

