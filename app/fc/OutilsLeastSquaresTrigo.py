# OutilsLeastSquaresTrigo.py

"""-------------------------------------------------------
Least squares trigonometric approximation. Three parts :
1) CREATION of scattered data (tk,ykd)
2) IMPORT of these data from the file "dataTrigo.txt"
3) Least Square trigonometric approximation

----------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

    # #basis functions for the test function
    # def f0(x) :
    #     return 1.
    # def f1(x) :
    #     return x
    # def f2(x) :
    #     return x**2
    # def f3(x) :
    #     return np.cos(x)
    # def f4(x) :
    #     return np.sin(x)
    # def f5(x) :
    #     return np.cos(3.*x)
    # def f6(x) :
    #     return np.sin(3.*x)
    # def f7(x) :
    #     return np.cos(5.*x)
    # def f8(x) :
    #     return np.sin(5.*x)
    
    # #Initial coefficients of basis functions :
    # a0 = 1; a1 = -2 ; a2 = 0.5; a3 = -1; a4 = -5 ; a5 = -4 ; 
    # a6 = 3; a7 = 3  ; a8 = -4
    
    # # initial test function
    # def f(t):
    #     y = a0*f0(t) + a1*f1(t) + a2*f2(t) + a3*f3(t) + a4*f4(t) + a5*f5(t) \
    #         + a6*f6(t) + a7*f7(t) + a8*f8(t)
    #     return y
    
    # """================================================
    # 1) CREATION of scattered data (tk,ykd)
    # ==================================================="""
    # plt.clf()
    # #---------------------------------------------------
    # # The initial test function
    # #---------------------------------------------------
    # a, b = -4, 7
    # t = np.linspace(a,b,400)
    # y = f(t)
    # plt.plot(t,y,'c:',label='initial curve')
    
    # #---------------------------------------------------
    # # Creation of data (tk,yk)
    # #---------------------------------------------------
    # #Uniform sampling of the curve
    # NbPts = 120
    # tk = np.linspace(a,b,NbPts)
    # yk = f(tk)
    # #plt.plot(tk,yk,'bo',ms=2,label='sampling')
    
    # #---------------------------------------------------
    # # Gaussian disruption of the sampling values : 
    # #---------------------------------------------------
    # mu, sigma = 0, 1.   # mean and standard deviation
    # s = np.random.normal(mu, sigma, NbPts)
    # ykd = yk + s
    # plt.plot(tk,ykd,'bo',ms=2,label='disrupted sampling')
    
    # #---------------------------------------------------
    # # Storage of data (tk,ykd) in the file 'dataTrigo.txt'
    # #---------------------------------------------------
    # # 1) We set the working directory
    # import os   # import operating system
    # myPath = "C:/Users/Luke/Documents/Luc/Enseignements/MAP 101"
    # myWorkingPath = myPath + "/2019-2020 MAP101/students/Dili/2023 stage Dili/Scripts Tkinter/data"
    # os.chdir(myWorkingPath)
    # # 2) Creation of a data file "data.txt" 
    # #    and writting of the data in this file
    # U = (tk,ykd)
    # np.savetxt('dataTrigo.txt', U, fmt='%1.20e')  # exponential notation
    
    
    # """================================================
    # 2) IMPORT scattered data (tk,ykd) 
    #    from the file "dataTrigo.txt"
    # ==================================================="""
    
    # (tk,ykd) = np.loadtxt('data/dataTrigo.txt')
    # plt.plot(tk,ykd,'rx',label='scattered data')


"""================================================
3) LEAST SQUARES Trigonometric Approximation
==================================================="""

def fcos(x,k,T) :
    w = 2*np.pi/T
    return np.cos(k*w*x)

def fsin(x,k,T) :
    w = 2*np.pi/T
    return np.sin(k*w*x)

def LeastSquareTrigo(a,b,tk,ykd,N):
    """ Least Square approximation of data (tk,ykd)
        by a Trigonometric polynomial of degree N
        over the interval [a,b]
    """
    NbPts = np.size(tk)
    t = np.linspace(a,b,400)
    # period
    T = b - a    
    # Least square matrix (transposed)        
    AT = np.ones((2*N+1,NbPts))
    for k in range(1,N+1):
        AT[2*k-1,:] = fcos(tk,k,T)  
        AT[2*k,:]   = fsin(tk,k,T)
    # normal equations and solution (the coefficients ak):
    M = np.dot(AT, AT.T)
    S = np.dot(AT, ykd)
    ak = np.linalg.solve(M, S)
    # we evaluate and plot the solution
    yt = ak[0]*np.ones(np.size(t))
    for k in range(1,N+1):
        yt = yt + ak[2*k-1]*fcos(t,k,T)
        yt = yt + ak[2*k]  *fsin(t,k,T)
    return t,yt


