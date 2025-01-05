# OutilsApproxExplicite3D.py

"""===========================================================
Outils Approx Bezier 3D explicite
Least squares Approximation
by a tensor Bezier surface of bi-degree(d1,d2)
=============================================================="""
import numpy as np
from scipy.special import binom

# Bernstein evaluation
def Bernstein(n, k, x):
    coeff = binom(n, k)
    return coeff * x**k * (1 - x)**(n - k)

# Evaluation of a tensor product polynomial expressed in Bernstein basis
# defined by a matrix of coefficients : CPz
# for values xc, yc
def SurfBezierExplicit(CPz,xc,yc):
    """ 
    the surface is the graph of an explicit function (x,y) -> F(x,y) 
    of degree (d1,d2) expressed in Bezier form
    CPz is the array of control points with shape (d1+1,d2+1)
    We evaluate the function value F(xc,yc)
    """
    Nx,Ny = np.shape(CPz)
    d1 = Nx - 1
    d2 = Ny - 1
    z = 0.
    for i in range(d1+1):
        for j in range(d2+1):
            z += CPz[i,j] * Bernstein(d1, i, xc) * Bernstein(d2, j, yc)
    return z



# Approximation polynomiale explicite 3D
# par une surface tensorielle Bézier de bi-degré (d1,d2)
# sur [0,1]x[0,1]
def LeastSquareExplicit3D(xPoints,yPoints,zPoints,d1,d2):
    """ Approximation polynomiale explicite 
        des points (xPoints,yPoints,zPoints)
        par une surface Bézier de bi-degré (d1,d2) 
        au dessus de [0,1]x[0,1]
    """
    n1, n2 = np.shape(xPoints)
    NbPts = n1*n2
    
    # we flatten data for Least Squares approx
    Xk = xPoints.flatten()
    Yk = yPoints.flatten()
    Zk = zPoints.flatten()
    # linear system for LS approximation
    # by a tensor product Bezier surface of degree d1,d2
    A = np.zeros(( NbPts,(d1+1)*(d2+1))) # matrix of the linear system
    # for each current value (xi,yi) :
    for i in range(NbPts):
            xc = Xk[i]  # xi-current value
            yc = Yk[i]  # yi-current value
            v = 0       # column counter
            # for each Bernstein polynomial :
            for k in range(d1+1):
                for l in range(d2+1):
                    A[i,v] = Bernstein(d1, k, xc) * Bernstein(d2, l, yc)
                    v = v+1
    # normal equations and solution:
    M = np.dot(A.T, A)
    S = np.dot(A.T, Zk)
    Cf = np.linalg.solve(M, S) 
    # We re-organize control points
    Cff = list(Cf)
    Zpc = np.reshape(Cff,(d1+1,d2+1))
    # approximating Bezier surface
    x = np.linspace(0,1,200)
    y = np.linspace(0,1,200)
    X,Y = np.meshgrid(x,y)
    Z = SurfBezierExplicit(Zpc,X,Y)
    return X,Y,Z



# Approximation polynomiale explicite 3D
# par une surface tensorielle Bézier de bi-degré (d1,d2)
# sur un intervalle [a,b]x[c,d]
# --> on se ramène au cas [0,1]^2 
def LeastSquareExplicit3D_abcd(xPoints,yPoints,zPoints,d1,d2,a,b,c,d):
    # on se ramène sur [0,1]x[0,1]
    uPoints = (xPoints - a)/(b-a)
    vPoints = (yPoints - c)/(d-c)
    # approximation sur [0,1]x[0,1]
    U,V,Z = LeastSquareExplicit3D(uPoints,vPoints,zPoints,d1,d2)
    X = (1-U)*a + U*b
    Y = (1-V)*c + V*d
    return X,Y,Z
    





