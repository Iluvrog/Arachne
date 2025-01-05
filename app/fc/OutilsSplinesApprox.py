# OutilsSplinesApprox.py

"""=================================================================
C2 cubic spline for approximation (explicite and parametric)
-> splineApproxU(xi,uk,zk)
and
C2 cubic Smoothing Spline 
-> smoothingsplineUGeneral(xi,uk,zk,rho)

for Approximation of scattered data (uk,zk)
U=Uniform : evenly spaced knots xi
data to be approximated are independant from the spline knots xi
===================================================================="""
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
def HermiteC1(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Output :
            x = sampling of 100 values in interval [x0,x1]
            y = image of x by the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100, endpoint=False)
    h = x1 - x0
    t = (x - x0) / h
    y = y0 * H0(t) + y0p * h * H1(t) + y1p * h * H2(t) + y1 * H3(t)
    return x, y

# Cubic interpolating Hermite spline C1-C1 of n points
def splineC1(xi,yi,yip):
    """ Hermite C1 spline interpolating data (xi,yi,yip)
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
        (x, y) = HermiteC1(xi[i],yi[i],yip[i],xi[i+1],yi[i+1],yip[i+1])
        xs = np.append(xs,x)
        ys = np.append(ys,y)
    return xs, ys


##########################################################
# this function is a particular case of the following one 
# that is of : smoothingsplineUGeneral()
def splineApproxU(xi,uk,zk):
    """ Approximation of data (uk,zk)
        by a C2 cubic natural spline with Uniform knots xi
        Input:
            xi = spline knots
            uk,zk = vectors of data to be approximated
            (data uk are assumed to be in the interval [xi_first,xi_last])
        Output :
            xsm = sampling of [xi_first, xi_last]
            ysm = image of xsm by the smoothing spline
    """
    n = np.size(xi)
    h = (xi[n-1] - xi[0])/(n-1)
    # construction of the spline matrix A:
    A = 4*np.eye(n,n) + np.diag(np.ones(n-1),1) \
                      + np.diag(np.ones(n-1),-1)
    A[0,0] = 2
    A[n-1,n-1] = 2
    # construction of matrix R:
    R = np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
    R[0,0] = -1
    R[n-1,n-1] = 1
    R = (3. / h) * R     
    # matrix A1R
    A1 = np.linalg.inv(A)    
    A1R = np.dot(A1,R)
    # construction of matrices H03 and H12:
    Nd = np.size(uk)
    H03 = np.zeros((Nd,n))
    H12 = np.zeros((Nd,n))
    k = 0
    for i in range(n-1) :                      #for each knots interval
        while (k < Nd) and (uk[k] <= xi[i+1]): #for each data in the interval
            u = (uk[k] - xi[i]) / h            #local parameter in [0,1]
            H03[k,i]   = H0(u) 
            H03[k,i+1] = H3(u)
            H12[k,i]   = h * H1(u) 
            H12[k,i+1] = h * H2(u)
            k = k+1
    # matrix H:
    H = H03 + np.dot(H12,A1R)
    # matrix W
    W = np.dot(H.T,H)
    # vector bb 
    bb = np.dot(zk,H) # because vector zk is here horizontal !
    # determination of values yi of the smoothing spline at points xi :
    yi = np.linalg.solve(W,bb)
    # determination of derivatives yip at points xi :
    Ry = np.dot(R, yi)
    yip = np.linalg.solve(A,Ry)
    # evaluation of the spline :
    (xs, ys) = splineC1(xi,yi,yip)
    return (xs, ys)


##########################################################
def smoothingsplineUGeneral(xi,uk,zk,rho):
    """ smoothing C2 cubic spline of data (uk,zk)
        U=uniform -> spline knots xi are evenly spaced, in increasing order
        data (uk,zk) to be approximated are independant from the knots xi
        Input:
            xi = spline knots
            uk,zk = vectors of data to be approximated
            (data uk are assumed to be in the interval [xi_first,xi_last])
            rho = positive scalar = smoothing parameter
        Output :
            xsm = sampling of [xi_first, xi_last]
            ysm = image of xsm by the smoothing spline
    """
    n = np.size(xi)
    h = (xi[n-1] - xi[0])/(n-1)
    # construction of the spline matrix A:
    A = 4*np.eye(n,n) + np.diag(np.ones(n-1),1) \
                      + np.diag(np.ones(n-1),-1)
    A[0,0] = 2
    A[n-1,n-1] = 2
    # construction of matrix R:
    R = np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
    R[0,0] = -1
    R[n-1,n-1] = 1
    R = (3. / h) * R     
    # construction of matrix S:
    S = 2*np.eye(n-2,n-2) + 0.5 * np.diag(np.ones(n-3),1) \
                          + 0.5 * np.diag(np.ones(n-3),-1)
    S = (h / 3.) * S
    # construction of matrix M:
    M = np.eye(n-2,n, 0) - 2 * np.eye(n-2,n, 1) + np.eye(n-2,n, 2)
    M = (3. / (h**2)) * M
    # construction of matrix N:
    N = np.eye(n-2,n, 0) - np.eye(n-2,n, 2)
    N = (1. / h) * N
    # matrix K 
    A1 = np.linalg.inv(A)    
    A1R = np.dot(A1,R)
    K = M + np.dot(N,A1R)
    # construction of matrices H03 and H12:
    Nd = np.size(uk)
    H03 = np.zeros((Nd,n))
    H12 = np.zeros((Nd,n))
    k = 0
    for i in range(n-1) :                      #for each knots interval
        while (k < Nd) and (uk[k] <= xi[i+1]): #for each data in the interval
            u = (uk[k] - xi[i]) / h            #local parameter in [0,1]
            H03[k,i]   = H0(u) 
            H03[k,i+1] = H3(u)
            H12[k,i]   = h * H1(u) 
            H12[k,i+1] = h * H2(u)
            k = k+1
    # matrix H:
    H = H03 + np.dot(H12,A1R)
    # matrix U2
    SK = np.dot(S,K)
    U2 = np.dot(H.T,H) + rho * np.dot(K.T,SK)
    # vector b2 
    b2 = np.dot(zk,H) #vector zk is here horizontal !
    # determination of values yi of the smoothing spline at points xi :
    yi = np.linalg.solve(U2,b2)
    # determination of derivatives yip at points xi :
    b = np.dot(R, yi)
    yip = np.linalg.solve(A,b)
    # evaluation of the spline :
    (xsm, ysm) = splineC1(xi,yi,yip)
    return (xsm, ysm)


