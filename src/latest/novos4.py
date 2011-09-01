# -*- coding: utf-8 -*-
import numpy as n
import random as r
import pylab as p
import pca_module as pp
from scipy.interpolate import interp1d

# distribuição gaussiana
# mu: determina o centro da distribuição, quando mu=0 e std=1, temos a distribuição normal
# std: determina o quão espaçado é a boca do sino
def gauss(x, mu=0, std=1):
    return p.exp(-((x - mu)**2)/(2*std)) / p.sqrt(2*p.pi*std)

def pca(dados):
    # II. normalizamos a matriz de dados (X = X - mean) e dividimos pelo d.p.
    #     (X = (X - mean) / dp
    for i in xrange(dados.shape[1]):
        dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

    # III. calculamos a matriz de covariância de X
    matriz_cov = n.cov(dados, bias=1, rowvar=0)

    # IV. calculamos os autovetores e autovalores e ordenamos em ordem decresc.
    autovalores, autovetores = n.linalg.eig(matriz_cov)
    args = n.argsort(autovalores)[::-1]
    autovalores = autovalores[args]
    autovetores = autovetores[args]

    # V. calculamos o vetor de componentes principais (= autovetores)

    # VI. calculamos os dados finais Y
    dados_finais = n.dot(autovetores.T, dados.T)

    # VII. interpretação
    autovalores_prop = [av/n.sum(autovalores) for av in autovalores]

    return matriz_cov, autovetores, autovalores, autovalores_prop, dados_finais.T

def pearson(dados):
    num_col = dados.shape[1]
    dp = n.std(dados, 0)
    covm = n.cov(dados, bias=1, rowvar=0)
    pearson = n.zeros((num_col, num_col))
    for i in xrange(num_col):
        for j in xrange(num_col):
            pearson[i,j] = covm[i,j] / (dp[i] * dp[j])
    return pearson
dados_teste = n.loadtxt('notas_compositores.txt')
def montecarlo2(qtd_aleatorios):
    # carregando dados
    #dados = n.loadtxt('notas_teste.txt')
    global dados_teste
    minimo = 1.0
    maximo = 9.0
    samples_todos = []

    # repetimos até termos a quantidade que queremos de aleatorios
    while (len(samples_todos) < qtd_aleatorios/7):
        d_todas = []
        xs_todas = []
        for i in xrange(len(dados_teste)):
            # 1. notas para cada linha (cada filósofo), notas = {x1, x2, ..., x8}
            notas = dados_teste[i]

            # 2. calculamos 8 pontos aleatórios xsN e a distância dK para cada filósofo
            xs = [p.uniform(minimo, maximo) for i in range(len(notas))]
            xs_todas.append(xs)

            # 3. calculamos dK
            dk = n.sqrt(n.sum([(xs[i] - notas[i])**2 for i in range(len(notas))]))

            # 4. guardamos dk em d_todas
            d_todas.append(dk)
        #d_todas = [n.sqrt(n.sum([(xs[i] - dados[i])**2 for i in range(len(notas))])) for i in range(len(dados[0]))]
        po = n.sum([n.exp(-0.5*((d_todas[i]**2)/0.4)) for i in range(7)])
        #po = n.sum([n.exp(-0.5*((d_todas[i]**2)/n.mean(n.std(dados, 0)))) for i in range(7)])
        #_std = 4
        #_mu = 0
        #po = n.sum([n.exp(-((d_todas[i]-_mu)**2) / (2*_std)) / n.sqrt(2*n.pi*_std) for i in range(7)])

        #print 'sigma', n.mean(n.std(dados, 0))

        #print '\nsimulação: ', j+1, '\n\nxs:\n', xs_todas, '\n\np: ', po

        # agora o fv é nosso po
        a = p.uniform()

        #print a, po
        if a < po:
            samples_todos.append(xs_todas)
            print 'CONSEGUIMOS UM GRUPO!!!!\n\n\n'
            print xs_todas
            print len(samples_todos)

    # concatena sublistas [[a], [b,c]] => [a, b, c]
    return sum(samples_todos, [])

def montecarlo(qtd_aleatorios):
    # carregando dados
    dados = n.loadtxt('notas_filosofos.txt')
    # notas de platão
    minimo = 0.0
    maximo = 10.0
    # cuidado com std, para evitar picos
    std = 0.4

    notas_todas = []
    qtd_medidas = 8 # 8 notas
    qtd_agentes = 7 # 7 filósofos/compositores/...

    for i in range(qtd_medidas):
        # notas para cada coluna (por métrica)
        notas = dados[:,i]
        # 1. plotar a distribuição gaussiana com mu centrado em cada posição
        int_x = p.arange(minimo, maximo, 0.1)
        # não preciso mais disso... agora só basta interpolar...
        #p.plot(int_x, [gauss(v, x, std) for v in int_x])

        # 2. interpolar todas as curvas de distribuição, objetivando a curva toda
        # somamos os vetores produzidos pela aplicação de guass em valores de int_x e depois dividimos por N=7
        # por fim, temos um vetor da média, e então interpolamos linearmente usando interp1d
        # essa é a função interpolada para todas as 8 medidas
        f = interp1d(int_x, reduce(lambda a,b: a + b, [n.array([gauss(v, each, std) for v in int_x]) for each in notas])/qtd_agentes)

        #p.plot(int_x, [f(v) for v in int_x], 'bx')

        # 3. essa curva define a distribuição que usaremos como p(x) em monte carlo
        novos = []
        for i in range(1000):
            # 1. sorteia um valor uniforme entre min e max
            v = p.uniform(minimo+1, maximo-1)

            # 2. calcula p(v)
            fv = f(v)

            # 3. sorteia um valor uniforme entre 0 e 1
            a = p.uniform()

            # 4. se a < v => guarda v na lista
            if a < fv:
                novos.append(v)
            notas_todas.append(novos)

    #p.hist(dados[0], normed=1)
    aleatorios_todos = []
    for i in range(qtd_aleatorios):
        aleatorios_todos.append([r.choice(notas_todas[i]) for i in range(qtd_medidas)])
        p.hist([r.choice(notas_todas[i]) for i in range(qtd_medidas)], normed=1)
    return aleatorios_todos

def principal():
    # I. carregamos a matriz de dados    

    # Filósofos:
    #dados = n.loadtxt('notas_filosofos.txt')

    # Compositores:
    # dados = n.loadtxt('notas_teste.txt')
    # covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
    # print '\n\nautovetores', autovetores.T
    # print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
    # print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)
    # print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
    # print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
    # print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
    # print '\nDados finais:\n', n.around(dados_finais, decimals=2)
    # c1 = dados_finais[:,0]
    # c2 = dados_finais[:,1]

    # p.clf()
    # p.plot(c1, c2)
    # #p.xlim((-4,4))
    # #p.ylim((-4,4))
    # p.savefig('pca.eps')

    # for i in xrange(dados.shape[1]):
    #     dados[:,i]=(dados[:,i]-dados[:,i].mean())/dados[:,i].std()

    # T, P, E = pp.PCA_nipals(dados)
    # princ = T[:,:2]
    # p.plot(princ[:,0], princ[:,1], color="black")
    # p.savefig('pca_outro.eps')

    # # Aleatórios:
    # #num_agentes = [10,15,20,25,30,35,40,45,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    # ##############################
    num_agentes = [100]
    num_vars = 8
    a = []
    autovalores_todos = {}
    for i in num_agentes:
        autovalores_todos[i] = []
        for j in xrange(1):
            # aqui eu jogo os valores sampleados
            #print '\n\ndados toscos\n\n', dados
            #print '\n\ndados monte carlo\n\n', montecarlo(10)
            #d = n.loadtxt('notas_filosofos.txt')
            #dados = n.array([[r.uniform(1,9) for x in range(num_vars)] for y in range(i)])
            # ########3
            # dados = n.array(montecarlo2(i))
            # print 'quantidade de aleatorios: ', len(dados)
            # covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
            # print '\n*** Teste %s para %s agentes ***\n' % (j, i)
            # print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
            # print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)
            # print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
            # print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
            # print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
            # c1 = dados_finais[:,0]
            # c2 = dados_finais[:,1]

            # PCA ALEATORIOS
            dados = n.array(montecarlo2(i))
            print '\n\nCONSEGUIMOS OS DADOS *******\n\n'
            dados_fi = n.loadtxt('notas_compositores.txt')
            dados = list(dados) + list(dados_fi)
            dados_antes = n.array(dados[:])
            dados = n.array(dados)
            print 'dados antes', dados
            covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados)
            print '\n\n*** ALEATORIOS ***\n\n'
            print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
            print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados), decimals=2)
            print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
            print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
            print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
            print 'dados todos', dados_antes
            print '7 últimas colunas', dados_antes[-7:]
            dados7 = n.dot(autovetores.T, dados[-7:].T)
            dados7_t = dados7.T
            c1_7 = dados7_t[:,0]
            c2_7 = dados7_t[:,1]
            c1 = dados_finais[:,0]
            c2 = dados_finais[:,1]
            p.plot(c1_7, c2_7, 'bo')
            p.plot(c1, c2, 'rx')
            p.savefig('pca_final.eps')
            # for i in xrange(dados.shape[1]):
            #     dados[:,i]=(dados[:,i]-dados[:,i].mean())/dados[:,i].std()

            # T, P, E = pp.PCA_nipals(dados)
            # princ = T[:,:2]
            # c1 = P[0]
            # c2 = P[1]
            # cc1 = c1 / sum(abs(c1)) * 100
            # cc2 = c2 / sum(abs(c2)) * 100
            # print 'CONTRIBUICOES'
            # print 'C1', [abs(x) for x in cc1]
            # print 'C2', [abs(x) for x in cc2]
            # print 'princ', princ[:,0][-7:], princ[:,1][-7:]
            #p.plot(princ[:,0][-7:], princ[:,1][-7:], 'rx', label='Random values')
            
            # PCA FILOSOFOS
            #dados_fi = n.loadtxt('notas_compositores.txt')
            # covar, autovetores, autovalores, autovalores_prop, dados_finais = pca(dados_fi)
            # print '\n\n*** ORIGINAIS ***\n\n'
            # print '\nMatriz de covariância:\n', n.around(covar, decimals=2)
            # print '\nMatriz de correlação de Pearson:\n', n.around(pearson(dados_fi), decimals=2)
            # print '\nAutovalores:\n', n.around(autovalores_prop, decimals=2)
            # print '\nVariância dos Autovalores:\n', n.around(n.var(autovalores_prop), decimals=2)
            # print '\nSoma dos dois primeiros:\n', round(autovalores_prop[0] + autovalores_prop[1], ndigits=2)
            # for i in xrange(dados_fi.shape[1]):
            #     dados_fi[:,i]=(dados_fi[:,i]-dados_fi[:,i].mean())/dados_fi[:,i].std()

            # T, P, E = pp.PCA_nipals(dados_fi)
            # princ = T[:,:2]
            # c1 = P[0]
            # c2 = P[1]
            # cc1 = c1 / sum(abs(c1)) * 100
            # cc2 = c2 / sum(abs(c2)) * 100
            # print 'CONTRIBUICOES'
            # print 'C1', [abs(x) for x in cc1]
            # print 'C2', [abs(x) for x in cc2]
            # p.plot(princ[:,0], princ[:,1], 'bo', label='Composers scores')
            # #p.xlim((-4,4))
            # #p.ylim((-4,4))
            # p.legend(loc='lower right')
            # p.savefig('pca_final.eps')

    #####################################################
            # análise filósofos

            # covar_fi, autovetores_fi, autovalores_fi, autovalores_prop_fi, dados_finais_fi = pca(dados_fi)
            # c1_fi = dados_finais_fi[:,0]
            # c2_fi = dados_finais_fi[:,1]
            
            # p.clf()
            # p.plot(c1, c2, 'rx')
            # p.plot(c1_fi, c2_fi, 'bo')
            # p.xlim((-4,4))
            # p.ylim((-4,4))
            # p.savefig('pca_aleatorios.png')

            # #a.append(autovalores_prop[0] + autovalores_prop[1])
            # autovalores_todos[i].append(autovalores_prop)
    # analisa autovalores dos testes aleatórios...
    # for i in num_agentes:
    #     print '\n\nPara os autovalores de % agentes:\n', i
    #     print '\nMédia da soma dos dois primeiros:', n.mean([av[0] + av[1] for av in autovalores_todos[i]])
    #     print '\nVariância dos autovalores:', [n.var(av) for av in autovalores_todos[i]]
    # p.plot(num_agentes, a)
    # p.savefig('foo.eps')
    #p.show()
        
if __name__ == '__main__':
    principal()
