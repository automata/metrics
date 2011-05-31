import os

for i in range(10):
    os.system('python analisa_aleatorio.py > aleatorios/a%i.txt' % i)
    os.system('mv g1.eps aleatorios/a%i_g1.eps' % i)
    os.system('mv g2.eps aleatorios/a%i_g2.eps' % i)
