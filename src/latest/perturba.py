# -*- coding: utf-8 -*-
import numpy as n, pca_module as pca, pylab as p

f= ["Platao", "Aristoteles", "Descartes", "Espinoza", "kant", "Nietzsche", "Deleuze"]
notas_renato = [
  [4, 5, 9, 5, 4, 5, 5, 4], 
  [8, 7, 6, 4, 6, 7, 2, 4], 
  [2, 3, 9, 6, 7, 4, 6, 6],
  [8, 2, 1, 5, 2, 3, 1, 1],
  [5, 3, 9, 7, 4, 5, 7, 5],
  [7, 9, 1, 9, 5, 7, 1, 1],
  [4, 7, 1, 8, 3, 4, 3, 7.]]
nr=n.array(notas_renato)

notas_luciano = [
  [2, 2, 9, 5, 5, 2, 5, 5],
  [8, 8, 8, 7, 9, 9, 3, 1],
  [1, 2, 9, 7, 7, 1, 9, 9],
  [8, 2, 1, 5, 2, 3, 1, 1],
  [9, 2, 8, 6, 5, 2, 8, 5],
  [8, 9, 1, 9, 5, 9, 1, 2],
  [7, 8, 1, 8, 2, 7, 7, 5.]]
nl=n.array(notas_luciano)

nn=(nr+nl)/2

print "Pré-processamento!!"
for i in xrange(nn.shape[1]):
  nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

T,P,E=pca.PCA_nipals(nn)

princ_or=T[:,:2]
E_or=n.copy(E)
NN=1000
numeroDeFilosofos=nn.shape[0]
# distancias[original, ruido, amostra]
distancias=n.zeros((numeroDeFilosofos,numeroDeFilosofos,NN))
autovals=n.zeros((NN,4))


for foobar in xrange(NN):
    nn=(nr+nl)/2

    ############
    # PERTURBACAO
    #STD=1
    #nn+=n.random.normal(0,STD,nn.shape)
    dist=n.random.randint(-2,3,nn.shape)
    nn+=dist
    ############

    print foobar
    for i in xrange(nn.shape[1]):
      nn[:,i]=(nn[:,i]-nn[:,i].mean())/nn[:,i].std()

    T,P,E=pca.PCA_nipals(nn)
    autovals[foobar]=E[:4]
    princ=T[:,:2]
    for i in xrange(numeroDeFilosofos):
        for j in xrange(numeroDeFilosofos):
            distancias[i,j,foobar]= n.sum((princ_or[i]-princ[j])**2)**.5

stds=n.zeros((numeroDeFilosofos,numeroDeFilosofos))
means=n.zeros((numeroDeFilosofos,numeroDeFilosofos))
main_stds = []
main_means = []
for i in xrange(numeroDeFilosofos):
    for j in xrange(numeroDeFilosofos):
        stds[i,j]=distancias[i,j,:].std()
        means[i,j]=distancias[i,j,:].mean()
        if i == j:
          main_stds.append(stds[i,j])
          main_means.append(means[i,j])
n.savetxt("mean2_.txt",means,"%.2e")
n.savetxt("stds2_.txt",stds,"%.2e")
print 'main_means', main_means
print 'main_stds', main_stds

foo = n.loadtxt('mean2_.txt')


#p.plot(princ_or[:,0],princ_or[:,1],"o", color=n.arange(len(T[:,0]))/float(len(T[:,0])) )
p.plot(princ_or[:,0],princ_or[:,1],"o")
p.savefig("primeiro_pca_PERT_STD.png")
  
  
deltas=autovals - E_or[:4]
medias=deltas.mean(0)
desvios=deltas.std(0)
print 'eigenvalues means', medias
print 'eigenvalues stds', desvios
