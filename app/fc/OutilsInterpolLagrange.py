# OutilsInterpolLagrange.py

"""--------------------------------------------------------
Outil pour l'interpolation de Lagrange (base de Lagrange)
--------------------------------------------------------"""
import numpy as np


def LagrangePoly(k,xi,t):
    """ Determination of the k-th Lagrange polynomial L_k(x) 
        associated with points xi,
        for a vector t of numerical values
        Input:
            k = integer designing the Lagrange polynomial to be evaluated
            xi = vector of points defining the Lagrange polynomial
            t = vector of real values (sampling of an interval [a,b])
        Output:
            y = L_k(t)
    """
    n = np.size(xi) # degree + 1
    y = np.ones(np.size(t))
    for j in range(n):
        if j != k:
            y = y * (t - xi[j] ) / ( xi[k] - xi[j] )
    return y


def Lagrange2D(a,b,xi,yi):
    """  Interpolation de Lagrange 2D explicite 
         des donnees xi, yi au dessus de l'intervalle [a,b]
    """
    N = np.size(xi)     # Nombre de points = degree+1
    t = np.linspace(a,b,400)
    # Determination of the interpolation polynomial of data xi,yi
    p = np.zeros(np.size(t))
    for k in range(N):
        p += yi[k] * LagrangePoly(k,xi,t)
    return t,p


def Lagrange3D(a,b,c,d,Ne,Me,zPoints):
    """  Interpolation de Lagrange 3D tensorielle explicite 
         de Ne*Me donnees zPoints (matrice de valeurs)
         au dessus de valeurs equireparties dans [a,b]x[c,d]
    """    
    # grille d'evaluation pour la surface interpolante
    u = np.linspace(a,b,600)
    v = np.linspace(c,d,600)
    U,V = np.meshgrid(u,v)
    U = U.T 
    V = V.T 
    # valeurs "au sol" pour l'interpolation
    xi = np.linspace(a,b,Ne)
    yj = np.linspace(c,d,Me)
    # Lagrange interpolant S = S(u,v)
    S = np.zeros((np.size(u),np.size(v)))
    for i in range(Ne):
        for j in range(Me):
            Li = LagrangePoly(i,xi,u)
            Lj = LagrangePoly(j,yj,v)
            S += zPoints[i,j] * np.outer(Li,Lj)
    return U,V,S



