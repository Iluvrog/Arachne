# OutilsApprox2D.py

"""----------------------------------------------
Least squares approximation:
    by a polynomial of degree 'degree'
----------------------------------------------"""
import numpy as np

def LeastSquares2D(a,b,xi,yi,degree):
    """ Determination of the best polynomial p(t) of degree "degree"
        defined on the interval [a,b]
        approximating data (xi,yi) according to the least squares method
        Returns :
            t = discretization of [a,b] 
            p = p(t) = the approximating polynomial
    """
    nblin = np.size(xi)
    nbcol = degree + 1
    # matrix A (monomial basis)
    A = np.ones((nblin,nbcol))
    for k in range(1,nbcol):
        A[:,k] = A[:,k-1] * xi
    # normal equations and solution:
    M = np.dot(A.T, A)
    S = np.dot(A.T, yi)
    cf = np.linalg.solve(M, S)
    # Horner evaluation and plotting
    t = np.linspace(a,b,400)
    p = cf[degree]
    for k in range(degree-1, -1, -1):
        p = cf[k] + p * t
    return t,p


