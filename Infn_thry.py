
# coding: utf-8
__author__ = 'Senthilkumar Lakshmanan'
__email__ = 'senthilkumarl@live.com'
__date__ = '11-04-2018'

########################Information Theory########################################


from math import sqrt, pi
import numpy as np
import scipy
import matplotlib.pyplot as plt

def entropy(p):
    p=np.asarray(p)
    Hi = -p*np.log2(p)
    return sum(Hi)

if __name__ == '__main__':
    rho = 0.5
    r = np.arange(0.001, 1.0, 0.01)
    H = []
    
    for i in range(len(r)):
        rho = r[i]
        #p=[rho,1-rho]
        p=[rho]
        H.append(entropy(p))
    print(H[-1])
    plt.plot(r,H)
    plt.show()
