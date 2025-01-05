# OutilsSurfaceSampling.py
# CREATION DE FICHIER DE DONNEES 3D 

"""============================================================
Echantillonnage d'une surface pour fabriquer un fichier de data
(en fait 3 fichiers : 1 pour chaque composante X,Y,Z)
- definition de la surface
- echantillonnage
- perturbation des donnees
- stockage dans 3 fichiers
- chargement des donnees + affichage
==============================================================="""
import numpy as np
import matplotlib.pyplot as plt

"""=========================================================
functions
============================================================"""
def F(x,y):
    x0, y0 = 1., 1.
    z1 = np.exp(-(x-x0)**2 - (y-y0)**2) 
    x0, y0 = -1., 1.5
    z2 = np.exp(-(x-x0)**2 - (y-y0)**2)
    x0, y0 = -2., -1.
    z3 = np.exp(-(x-x0)**2 - (y-y0)**2)
    x0, y0 = 0., -1.
    z4 = np.exp(-(x-x0)**2 - (y-y0)**2)
    x0, y0 = 1.5, -1.5
    z5 = np.exp(-(x-x0)**2 - (y-y0)**2)
    z = 1.3 * z1 + 1.8 * z2 + 1.2 * z3 - 2. * z4 + 2. * z5
    return z


"""=========================================================
MAIN PROGRAM
============================================================"""
#------------------------------------------------
# NEW FIGURE
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.set_title('2D approximation')
ax.view_init(35, -40)

#------------------------------------------------
# PLOT 3D SURFACE (graph of F)
xmin = -2; xmax = 2
ymin = -2; ymax = 2
x = np.linspace(xmin,xmax,150)
y = np.linspace(ymin,ymax,150)
X,Y = np.meshgrid(x,y)
Z = F(X,Y)
ax.plot_wireframe(X, Y, Z, color='b', lw=0.3)
# ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.3)

#------------------------------------------------
# SAMPLING
Nx = 28 # number of points according x-direction
Ny = 35 # number of points according y-direction
xi = np.linspace(xmin,xmax,Nx)
yj = np.linspace(ymin,ymax,Ny)
X0,Y0 = np.meshgrid(xi,yj)
Z0 = F(X0,Y0)
ax.scatter(X0, Y0, Z0, c='blue', marker='.', s=30)

#------------------------------------------------
# NOISY SAMPLING : we add a uniform noise on data points
su,sv = np.shape(X0)
X01 = X0 + (np.random.rand(su,sv) - 0.5)/10.
Y01 = Y0 + (np.random.rand(su,sv) - 0.5)/10.
Z01 = F(X01,Y01)
Z01 = Z01 + (np.random.rand(su,sv) - 0.5)/4.
ax.scatter(X01, Y01, Z01, c='red', marker='.', s=20)


"""=====================================================
STORAGE of data (X01,Y01,Z01) in 3 files
========================================================"""
# 1) We set the working directory -> on the desktop
import os   # import operating system
myPath = "C:/Users/Luke/Desktop"
os.chdir(myPath)

# 2) Creation of 3 data files 
#    and writting of the data in these files
np.savetxt('dataBumpsX.txt', X01, fmt='%1.20e')
np.savetxt('dataBumpsY.txt', Y01, fmt='%1.20e')
np.savetxt('dataBumpsZ.txt', Z01, fmt='%1.20e')

"""=====================================================
LOADING data (X01,Y01,Z01) from the files 
========================================================"""
# 3) To import data
#    IMPORT scattered data from 3 files
#    XX, YY, ZZ are 3 matrices with the same shape
import numpy as np
import matplotlib.pyplot as plt
XX = np.loadtxt('dataBumpsX.txt')
YY = np.loadtxt('dataBumpsY.txt')
ZZ = np.loadtxt('dataBumpsZ.txt')

# 4) Display
# We open a 3D figure :
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.set_title('3D data to be approximated')
ax.view_init(35, -40)
# display
ax.scatter(XX, YY, ZZ, c='red', marker='.', s=20)




