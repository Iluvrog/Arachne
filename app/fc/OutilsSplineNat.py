# OutilsSplineNat.py

"""===========================================================
Outils splines naturelles
=============================================================="""
import numpy as np

# Cubic Hermite basis over [0,1] :
def H0(t) :
    y = 1 - 3 * t**2 + 2 * t**3
    return y
def H1(t) :
    y = t - 2 * t**2 + t**3
    return y    
def H2(t) :
    y = - t**2 + t**3
    return y
def H3(t) :
    y = 3 * t**2 - 2 * t**3
    return y   

# Cubic Hermite interpolation over 2 points
def HermiteC1b(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Remark : no plotting in this version HermiteC1b()
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Output :
            x = sampling of 100 values in interval [x0,x1]
            y = image of x by the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    h = x1 - x0
    t = (x - x0) / h
    y = y0 * H0(t) + y0p * h * H1(t) + y1p * h * H2(t) + y1 * H3(t)
    return x, y

# Cubic interpolating Hermite spline C1-C1 of n points
def splineC1C1(xi,yi,yip):
    """ Hermite C1 spline interpolating 'C1 data' (xi,yi,yip)
        --> the data are C1 (Hermite data of order one)
        --> the constructed spline is C1
        Input :
            xi,yi,yip = 3 arrays of size N (points, values, derivatives)
        Output :
            xs = sampling of 100*(N-1) values in [x_first,x_last]
            ys = image of xs by the the Hermite C1 spline
    """
    N = np.size(xi)
    xs = []
    ys = []
    for i in range(N-1):
        (x, y) = HermiteC1b(xi[i],yi[i],yip[i],xi[i+1],yi[i+1],yip[i+1])
        xs = np.append(xs,x)
        ys = np.append(ys,y)
    return xs, ys


def splineC2NatNU(xi,yi):
    """ Natural C2 cubic spline interpolating data (xi,yi)
        NU=nonUniform -> arbitrary values xi in increasing order
        Input:
            xi,yi = vectors of data to be interpolated
        Output :
            xs = sampling of [xi_first, xi_last]
            ys = image of xs by the spline
    """
    n = np.size(xi)
    h = xi[1:n] - xi[0:n-1]
    # construction of the spline matrix :
    hL = h[0:n-2]
    hR = h[1:n-1]
    hD = hL + hR
    hD = np.append(hD,1.)
    hD = np.append(1.,hD)
    hL = np.append(1.,hL)
    hR = np.append(hR,1.)
    A = 2*np.diag(hD,0) + np.diag(hL,1) + np.diag(hR,-1)
    # second member :
    b = np.zeros(n)
    b[0] = (yi[1] - yi[0]) / h[0]
    b[-1] = (yi[-1] - yi[-2]) / h[-1]
    for i in range(1,n-1) :
        u = (yi[i+1] - yi[i]) * h[i-1] / h[i]
        v = (yi[i] - yi[i-1]) * h[i] / h[i-1]
        b[i] = u + v
    b = 3 * b
    # determination of derivatives yip at points xi :
    yip = np.linalg.solve(A,b)
    # evaluation of the spline :
    (xs, ys) = splineC1C1(xi,yi,yip)
    return (xs, ys)


def chordal(xi,yi):
    """ return the chordal parameterization associated with (xi,yi) """
    N = np.size(xi)
    tc = np.zeros(N)
    for i in range(1,N) :
        di =  np.sqrt( (xi[i] - xi[i-1])**2 + (yi[i] - yi[i-1])**2 )
        tc[i] = tc[i-1] + di
    tc = tc / tc[N-1]
    return tc

def Chebyshev(a,b,N):
    """ return the Chebyshev parameterization associated with N on [a,b] """
    degree = N - 1
    k = np.arange(0,degree+1)
    kk = (2*k+1)*np.pi / (2*degree+2)
    tch = (a+b)/2 + ((b-a)/2) * np.cos(kk)
    return tch


