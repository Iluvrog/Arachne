# OutilsBezierBSpline.py

"""===========================================================
Outils d'evaluation pour les courbes de Bézier et B-spline 2D
=============================================================="""
import numpy as np
from scipy.special import binom

#------------------------------------------------------------
# Bézier
#------------------------------------------------------------
def Bernstein(n, k, x):
    coeff = binom(n, k)
    return coeff * x**k * (1 - x)**(n - k)

# Evaluation d'une courbe de Bézier parametree
def EvaluateBezier(CPx,CPy):
    """ Construction d'une courbe de Bézier paramétrée (donc sur [0,1])
        à partir de son polygone de contrôle
        Input:
            CPx, CPy : vecteurs des coordonnées des points de contrôle
        Output : 
            Cx,Cy = vecteurs des points de la courbe
    """
    degree = np.size(CPx) - 1
    Nb_t = 200 # nombre de points pour la discrétisation
    t = np.linspace(0,1,Nb_t)
    Cx = np.zeros(Nb_t)
    Cy = Cx
    for k in range(degree+1):
        Bnkt = Bernstein(degree,k,t)
        Cx = Cx + CPx[k] * Bnkt
        Cy = Cy + CPy[k] * Bnkt
    return Cx,Cy

#------------------------------------------------------------
# B-spline
#------------------------------------------------------------
def s00(t0,t1,t):
    y = ((t1-t)/(t1-t0))**3
    return y

def s10(t0,t1,t2,t):
    y = ((t-t0)*((t1-t)**2))/((t1-t0)**3) \
        + ((t2-t)*(t1-t)*(t-t0))/((t2-t0)*((t1-t0)**2)) \
        + (((t2-t)**2)*(t-t0))/(((t2-t0)**2)*(t1-t0))
    return y

def s11(t0,t1,t2,t):
    y = ((t2-t)**3)/(((t2-t0)**2) * (t2-t1))
    return y

def s20(t0,t1,t2,t3,t):
    y = (((t-t0)**2)*(t1-t))/((t2-t0)*((t1-t0)**2)) \
        + (((t-t0)**2)*(t2-t))/(((t2-t0)**2)*(t1-t0)) \
        + ((t3-t)*((t-t0)**2))/((t3-t0)*(t1-t0)*(t2-t0))
    return y

def s21(t0,t1,t2,t3,t):
    y = ((t-t0)*((t2-t)**2))/(((t2-t0)**2)*(t2-t1)) \
        + ((t3-t)*(t-t0)*(t2-t))/((t3-t0)*(t2-t0)*(t2-t1)) \
        + (((t3-t)**2)*(t-t1))/((t3-t0)*(t3-t1)*(t2-t1))
    return y

def s22(t0,t1,t2,t3,t):
    y = ((t3-t)**3)/((t3-t0)*(t3-t1)*(t3-t2))
    return y

def si0(a,b,c,d,e,t):
    y = ((t-a)**3)/((d-a)*(c-a)*(b-a))
    return y

def si1(a,b,c,d,e,t):
    y = (((t-a)**2)*(c-t))/((d-a)*(c-a)*(c-b)) \
        + ((t-a)*(t-b)*(d-t))/((d-a)*(d-b)*(c-b)) \
        + (((t-b)**2)*(e-t))/((e-b)*(d-b)*(c-b))
    return y

def si2(a,b,c,d,e,t):
    y = ((t-a)*((d-t)**2))/((d-a)*(d-b)*(d-c)) \
        + ((t-b)*(e-t)*(d-t))/((e-b)*(d-b)*(d-c)) \
        + ((t-c)*((e-t)**2))/((e-b)*(e-c)*(d-c))
    return y

def si3(a,b,c,d,e,t):
    y = ((e-t)**3)/((e-b)*(e-c)*(e-d))
    return y

#---------------------
def Bspline(tk,i,t):
    """
        Calcul de N^3_i(t) pour un scalaire t
        Output :
            valeur en t de la fonction B spline cubique numéro i 
            associé à la suite de noeuds tk
        Input : 
            tk = vecteur des noeuds = np.array([]) de taille n = len(tk)
            RMQ : il faut n >= 6
            t valeur réelle comprise entre tk[0] et tk[n-1]
            i = entier = numéro de la fonction B-spline concernée 
            avec 0 <= i <= n+1
    """
    n = len(tk)

    if i == 0:
        t0 = tk[0]
        t1 = tk[1]
        if t0 <= t < t1:
            return s00(t0,t1,t)
        else:
            return 0

    if i == 1:
        t0 = tk[0]
        t1 = tk[1]
        t2 = tk[2]
        if t0 <= t < t1:
            return s10(t0,t1,t2,t)
        elif t1 <= t < t2:
            return s11(t0,t1,t2,t)
        else:
            return 0

    if i == 2:
        t0 = tk[0]
        t1 = tk[1]
        t2 = tk[2]
        t3 = tk[3]
        if t0 <= t < t1:
            return s20(t0,t1,t2,t3,t)
        elif t1 <= t < t2:
            return s21(t0,t1,t2,t3,t)
        elif t2 <= t < t3:
            return s22(t0,t1,t2,t3,t)
        else:
            return 0

    if i >= 3 and i < n-1:
        t0 = tk[i-3]
        t1 = tk[i-2]
        t2 = tk[i-1]
        t3 = tk[i]
        t4 = tk[i+1]
        if t0 <= t < t1:
            return si0(t0,t1,t2,t3,t4,t)
        elif t1 <= t < t2:
            return si1(t0,t1,t2,t3,t4,t)
        elif t2 <= t < t3:
            return si2(t0,t1,t2,t3,t4,t)
        elif t3 <= t < t4:
            return si3(t0,t1,t2,t3,t4,t)
        else:
            return 0
    
    if i == n-1:
        t0 = tk[i-3]
        t1 = tk[i-2]
        t2 = tk[i-1]
        t3 = tk[i]
        if t0 <= t < t1:
            return s22(t3,t2,t1,t0,t)
        elif t1 <= t < t2:
            return s21(t3,t2,t1,t0,t)
        elif t2 <= t < t3:
            return s20(t3,t2,t1,t0,t)
        else:
            return 0

    if i == n:
        t0 = tk[i-3]
        t1 = tk[i-2]
        t2 = tk[i-1]
        if t0 <= t < t1:
            return s11(t2,t1,t0,t)
        elif t1 <= t < t2:
            return s10(t2,t1,t0,t)
        else:
            return 0 

    if i == n+1:
        t0 = tk[i-3]
        t1 = tk[i-2]
        if t0 <= t <= t1:
            return s00(t1,t0,t)
        else:
            return 0 

#---------------------------------
def EvaluateBspline(CPx,CPy):
    # Evaluation d'une courbe B-spline paramétrée
    N = len(CPx)    # nombre de points du polygone
    nb_noeuds = N-2
    # noeuds uniformes :
    tk = np.linspace(0,1,nb_noeuds) 
    Npts = 400      # pour l'évaluation
    t = np.linspace(0,1,Npts)
    
    # calcul de la courbe B-spline (NON OPTIMISE)
    x = np.zeros(Npts)
    y = np.zeros(Npts)
    for i in range (Npts):
        tt = t[i]
        for k in range (N):
            x[i] = x[i] + CPx[k]*Bspline(tk,k,tt)
            y[i] = y[i] + CPy[k]*Bspline(tk,k,tt)
    return x,y

