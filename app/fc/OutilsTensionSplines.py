# OutilsTensionSplines.py

"""====================================================
Interpolating C2 tension Splines (exponential splines)
    -> arbitrary points xi in increasing order
    -> natural end conditions : s"(a) = s"(b) = 0
    -> main function : (xs,ys) = TensionSplineEval(sigmai,xi,yi)
======================================================="""
import numpy as np
    
# Exponential generator over [0,1] :
def phi(w,u) :
    # w is the local tension parameter
    y = np.sinh(w*u) - u * np.sinh(w)
    y  = y / (np.sinh(w) - w)
    return y

# Tension Spline interpolation evaluation over 2 points
def TensionSplineLocalEval(sigma,x0,y0,y0p,x1,y1,y1p):
    """ Tension Spline interpolation evaluation over 2 points x0 < x1
        Input :
            sigma (real number) = tension parameter on interval [x0,x1]
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Output :
            x = sampling of 100 values in interval [x0,x1]
            y = image of x by the interpolating tension spline
    """
    x = np.linspace(x0,x1,100)
    h = x1 - x0
    u = (x - x0) / h
    w = sigma * h   # local tension parameter
    alpha = (w * np.cosh(w) - np.sinh(w)) / (np.sinh(w) - w)
    # coefficients de la fonction si :
    ai = y0
    bi = y1
    ci = - (1+alpha) * (y1-y0) + h * (alpha*y0p + y1p)
    ci = ci / (1 - alpha**2)
    di =   (1+alpha) * (y1-y0) - h * (y0p + alpha*y1p)
    di = di / (1 - alpha**2)
    y = ai * (1-u) + bi * u + ci * phi(w, 1-u) + di * phi(w,u)
    return x, y

# Tension Spline interpolation evaluation over N points
def TensionSplineGlobalEval(sigmai,xi,yi,yip):
    """ Tension Spline interpolation evaluation over N points
        Input :
            sigmai = array of size N-1 (tension parameters on each interval)
            xi,yi,yip = 3 arrays of size N (xpoints, yvalues, derivatives)
        Output :
            xs = sampling of 100*(N-1) values in [x_first,x_last]
            ys = image of xs by the the C2 tension spline
    """
    N = np.size(xi)
    xs = []
    ys = []
    for i in range(N-1):
        (x, y) = TensionSplineLocalEval(sigmai[i], \
                    xi[i],yi[i],yip[i],xi[i+1],yi[i+1],yip[i+1])
        xs = np.append(xs,x)
        ys = np.append(ys,y)
    return xs, ys

def TensionSplineEval(sigmai,xi,yi):
    """ Natural C2 Tension Splines interpolating dataset (xi,yi)
        with values xi in increasing order and tension parameters sigmai
        Input:
            sigmai = tension parameters on each interval (array of size N-1)
            xi,yi = vectors of data to be interpolated (2 arrays of size N)
        Output :
            xs = sampling of [xi_first, xi_last]
            ys = image of xs by the tension spline
    """
    N = np.size(xi)
    hi = xi[1:N] - xi[0:N-1]
    # Local tension parameters
    wi = sigmai * hi 
    alphai = (wi * np.cosh(wi) - np.sinh(wi)) / (np.sinh(wi) - wi)
    betai = wi**2 * np.sinh(wi) / (np.sinh(wi) - wi)
    # matrix coefficients :
    mi = np.ones(N)
    mi[1:N-1] = (1-alphai[1:N-1]**2) * hi[1:N-1] * betai[:N-2]
    ni = np.zeros(N)
    ni[1:N-1] = (1-alphai[:N-2]**2) * hi[:N-2] * betai[1:N-1]
    # main diagonal :
    alp1 = np.append(alphai[0],alphai)
    alp2 = np.append(alphai,0.)
    hD = alp1 * mi + alp2 * ni
    # diagonals down (or left hL) and up (or right hR)
    hL = mi[1:N-1]
    hL = np.append(hL,1.)
    hR = ni[1:N-1]
    hR = np.append(1.,hR)
    # tension spline matrix :
    A = np.diag(hD,0) + np.diag(hL,-1) + np.diag(hR,1)
    # second member :
    DT = (1 + alphai) * (yi[1:N] - yi[0:N-1]) / hi
    DT1 = np.append(DT[0],DT)
    DT2 = np.append(DT,0.)
    b = mi * DT1 + ni * DT2
    # determination of derivatives yip at points xi :
    yip = np.linalg.solve(A,b)
    # evaluation of the tension spline :
    (xs, ys) = TensionSplineGlobalEval(sigmai,xi,yi,yip)
    return (xs, ys)


