import pickle
import math
import numpy as np
import sys 
import os
#sys.path.append(os.path.abspath("C:/Users/brunk/Desktop/Uni/Codes/NGSOLVE/External_Module"))
from ngsolve import *

#-------Numerische Parameter--------
N=2
#SetNumThreads(8) 
outputname = "Nonisothermal_MWS_NS"



outputpath = ""

if(outputpath==""):
    folders=[]
else:
    if(outputpath[-1]!="/"):
        outputpath+="/"
    folders=outputpath.split("/")[:-1]

T = 0.1 
reflevel = 7 # refine -1
dtmax = 0.0001#10**(-3)/nmeshmax
dt = dtmax
stepsmax = math.ceil(T/dtmax)
phiL2 = np.zeros((stepsmax+1,reflevel))
muL2  = np.zeros((stepsmax+1,reflevel))
uL2 = np.zeros((stepsmax+1,reflevel))
uH1 = np.zeros((stepsmax+1,reflevel))
pL2 = np.zeros((stepsmax+1,reflevel))
thetaL2 = np.zeros((stepsmax+1,reflevel))
thetaH1 = np.zeros((stepsmax+1,reflevel))
with TaskManager():
    refine = range(1,reflevel)
    for ref in refine:
        print("Stage:",ref)
        n_mesh = 2**(ref) # Gittergroeße
        n_mesh2 = 2**(1+ref) # Gittergroeße
        #dt = 10**(-3)*2**(-ref)  # Zeitschritt
        #dt2 = 10**(-3)*2**(-ref-1) # Zeitschritt
        #--------File output---------
        itcount = 0 # Newton Überschreitungen
        steps = math.ceil(T/dt) # Anzahl Zeitschritte

        from ngsolve.meshes import MakeStructured2DMesh
        Th = MakeStructured2DMesh(quads=False,nx=n_mesh,ny = n_mesh, periodic_x=True,periodic_y=True)

        Vh  = Periodic(H1(Th,order=1)**N)#1
        Xh = Periodic(VectorH1(Th, order=2))#2
        Qh  = Periodic(H1(Th,order=1))#1
        #R   = NumberSpace(Th)
        FES = FESpace([Vh,Vh,Xh,Qh,Qh])

        phi,mu,u,p,theta=GridFunction(FES).components

        #Th2 = Make_2D(nx=n_mesh2)
        Th2 = MakeStructured2DMesh(quads=False,nx=n_mesh2,ny = n_mesh2, periodic_x=True,periodic_y=True)

        Vh2  = Periodic(H1(Th2,order=1)**N)#1
        Xh2 = Periodic(VectorH1(Th2, order=2))#2
        Qh2  = Periodic(H1(Th2,order=1))#1
        #R   = NumberSpace(Th)
        FES2 = FESpace([Vh2,Vh2,Xh2,Qh2,Qh2])

        phi1,mu1,u1,p1,theta1=GridFunction(FES2).components

        print(n_mesh,"    ",n_mesh2)

        for i in range(0,steps+1):

            if i%100==0:
                print(i)

            with open(outputpath + outputname + str(n_mesh) + "_" + str(i) + ".step", "rb") as infile:
                phi.vec.data,mu.vec.data,u.vec.data,p.vec.data,theta.vec.data = pickle.load(infile)

         
            #phi2,mu2,u2,p2=GridFunction(FES2).components
            #phip,mup,ep,thetap,up,pp=GridFunction(FES).components

            with open(outputpath + outputname + str(n_mesh2) + "_" + str(i) + ".step", "rb") as infile:
                phi1.vec.data,mu1.vec.data,u1.vec.data,p1.vec.data,theta1.vec.data = pickle.load(infile)
            #infile = open(outputpath+outputname+"_"+str(n_mesh2)+"_"+str(2*i)+".step", "rb")
            #phi2,mu2,u2,p2 = pickle.load(infile)

            phiL2[i,ref] = Integrate((phi1-phi)**2,Th2,VOL)
            uL2[i,ref] = Integrate((u1-u)**2,Th2,VOL)
            thetaL2[i,ref] = Integrate((theta1-theta)**2,Th2,VOL)
            muL2[i,ref]  = Integrate((mu1-mu)**2 + InnerProduct(Grad(mu1)-Grad(mu),Grad(mu1)-Grad(mu)),Th2,VOL)
            uH1[i,ref] = Integrate((u1-u)**2+ InnerProduct(Grad(u1)-Grad(u),Grad(u1)-Grad(u)),Th2,VOL)
            pL2[i,ref] = Integrate((p1-p)**2,Th2,VOL)
            thetaH1[i,ref] =  Integrate((theta1-theta)**2,Th2,VOL) + Integrate((Grad(theta1)-Grad(theta))*(Grad(theta1)-Grad(theta)),Th2,VOL) 

    maxphi = phiL2.max(axis=0)
    ratephi = np.log2(np.divide(maxphi[:-1], maxphi[1:])) 

    maxu = uL2.max(axis=0)
    rateu = np.log2(np.divide(maxu[:-1], maxu[1:])) 

    maxtheta = thetaL2.max(axis=0)
    ratetheta = np.log2(np.divide(maxtheta [:-1], maxtheta[1:])) 

    L2mu = dt*np.sum(muL2,axis=0)
    ratemu = np.log2(np.divide(L2mu[:-1], L2mu[1:]) )

    L2u = dt*np.sum(uH1,axis=0)
    rateuH = np.log2(np.divide(L2u[:-1], L2u[1:]) )

    L2p = dt*np.sum(pL2,axis=0)
    ratep = np.log2(np.divide(L2p[:-1], L2p[1:])) 

    H1theta = dt*np.sum(thetaH1,axis=0)
    ratethetaH = np.log2(np.divide(H1theta[:-1], H1theta[1:])) 

    print("Phi:")
    print(maxphi)
    print(ratephi)

    print("U:")
    print(maxu)
    print(rateu)

    
    print("theta:")
    print(maxtheta)
    print(ratetheta)

    print("U:")
    print(L2u)
    print(rateuH)

    print("mu:")
    print(L2mu)
    print(ratemu)

    print("p:")
    print(L2p)
    print(ratep)

    print("thetaH1:")
    print(H1theta)
    print(ratethetaH)