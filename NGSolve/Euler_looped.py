import math
import numpy as np
import sys 
import os
from ngsolve.solvers import *
from ngsolve import *
#from ngsolve.meshes import MakeStructured2DMesh as Make_2D
import netgen.occ as ngocc
import pickle
#SetNumThreads(22)

#import print_options
np.set_printoptions(precision=16)

#-------Numerical Parameter--------
n_mesh = 1
#n = 50 # 1/n is meshsize
dt = 0.0001 # timestep
T = 0.1 # finaltime
N = 2 # number of component 
VvecD = [0.8,0.2]     # V Vector
MWSScale = 10**(-1)   # Scaling factor for MWS-matrix
Unit = [1 for i in range(N)]     


itsteps = 20 # max newtonsteps
iterror = 1e-10 # tolerance newton
#SetNumThreads(5) #limit num of threads


#------Saving params--------
outputpath = ""
paraout = False # Paraview Output   

def tanh(x):
	# Deal with some math overflow problems
	if type(x) is Parameter or type(x) is CoefficientFunction:
		# Added to ensure that the tanh calculation does not return nan
		x_adj = IfPos(x - 350, 350, x)
	else:  # float or int
		if x > 350:
			x_adj = 350
		else:
			x_adj = x
	return (exp(2 * x_adj) - 1.0) / (exp(2 * x_adj) + 1.0)


#------Parameterfunctions & IC------
def MWS(rhop,thetap,N):
     Ma = []
     rhoG = rhop*Unit
     for i in range(N):
        for j in range(N):
            if i==j:
                Ma.append(MWSScale*(rhop[i]-rhop[i]*rhop[j]/rhoG))
            else:
                Ma.append(-MWSScale*rhop[i]*rhop[j]/rhoG)
     return(Ma)

def HeatCon(rhop,thetap,N):
     Ma = 10**(-2)
     return(Ma)


def Pot(rhop,thetap,N):
     rhoG = rhop*Unit
     Ma = -rhoG*thetap*(log(thetap)-1)
     for i in range(N):
       Ma += thetap*rhop[i]*log(rhop[i]/rhoG)
     return(Ma)

def Entropy(rhop,thetap,N):
     rhoG = rhop*Unit
     Ma = rhoG*log(thetap)
     for i in range(N):
       Ma -= rhop[i]*log(rhop[i]/rhoG)
     return(Ma)

def Energy(rhop,thetap,N):
     Ma = rhop*Unit*thetap
     return(Ma)

def etaForm(rhop,thetap,N):
     Ma = 10**(-3)
     #rhoG = rhop*Unit
     for i in range(N):
       Ma += 0*rhop[i]**2
     return(Ma)

def Tensorcontraction(mu,theta,thetaold,GradPsi,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:] - mu[i]/(thetaold*theta)*Grad(theta))*GradPsi[j,:])
    return(Ma)

def Tensorcontraction2(mu,theta,thetaold,w,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:] - mu[i]/(thetaold*theta)*Grad(theta))*\
                     (w/theta*Grad(mu)[j,:] + mu[j]/thetaold*Grad(w) - w*mu[j]/(thetaold*theta)*Grad(theta)))
    return(Ma)

def TensorcontractionNewton(mu,theta,thetadelta,thetaold,GradPsi,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(-1/theta**2*thetadelta*Grad(mu)[i,:] + mu[i]/(thetaold*theta**2)*thetadelta*Grad(theta) - mu[i]/(thetaold*theta)*Grad(thetadelta))*GradPsi[j,:])
    return(Ma)

def Tensorcontraction2Newton(mu,theta,thetadelta,thetaold,w,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(-1/theta**2*thetadelta*Grad(mu)[i,:] + mu[i]/(thetaold*theta**2)*thetadelta*Grad(theta) - mu[i]/(thetaold*theta)*Grad(thetadelta))\
                     *(w/theta*Grad(mu)[j,:] + mu[j]/thetaold*Grad(w) - w*mu[j]/(thetaold*theta)*Grad(theta))
                     \
                     + M[i,j]*(1/theta*Grad(mu)[i,:] - mu[i]/(thetaold*theta)*Grad(theta))\
                        *(-w/theta**2*thetadelta*Grad(mu)[j,:] + w*mu[j]/(thetaold*theta**2)*thetadelta*Grad(theta)- w*mu[j]/(thetaold*theta)*Grad(thetadelta))
                     )
    return(Ma)


M    = lambda rhop,thetap: tuple(MWS(rhop,thetap,N))
F    = lambda rhop,thetap: Pot(rhop,thetap,N)
S    = lambda rhop,thetap: Entropy(rhop,thetap,N)
E    = lambda rhop,thetap: Energy(rhop,thetap,N)
K    = lambda rhop,thetap: HeatCon(rhop,thetap,N)
eta  = lambda rhop,thetap: etaForm(rhop,thetap,N)

Vvec = CF(tuple(VvecD))
Unit = CF(tuple(Unit))

rho0 = CF((1+0.2*sin(2*pi*x)*sin(2*pi*y),1/VvecD[1] - VvecD[0]/VvecD[1]*(1+0.2*sin(2*pi*x)*sin(2*pi*y)))) 
u0 = CF((-sin(pi*x)**2*sin(2*pi*y),sin(pi*y)**2*sin(2*pi*x)))
theta0 = CF(1.0 + 0.1*sin(2*pi*x)*cos(4*pi*y))



#-------Metavariables---------
itcount = 0 # Newton Ãœberschreitungen
steps = math.ceil(T/dt) # Anzahl Zeitschritte
newtonerrs = [] # Letzte Newtonfehler vor Abbruch 

#--------File output---------

if(outputpath==""):
    folders=[]
else:
    if(outputpath[-1]!="/"):
        outputpath+="/"
    folders=outputpath.split("/")[:-1]
if(paraout):
    folders.append("Bilder")
path=""
for folder in folders:
    path+=folder+"/"
    if(not os.path.exists(path)):
        os.mkdir(path)
        print("\nPfad:",path[:-1],"erstellt!")

#-------Start---------




with TaskManager():

    for nn in range(1,9):
        n_mesh = 2**nn
        outputname = "Nonisothermal_MWS_NS"+str(n_mesh)
        print("\nProgrammstart:")
        print("n = ", n_mesh )
        print("h = ", 1.0/n_mesh )
        print("dt = ", dt )
        print("T = ", T )
        print("steps = ", steps )

        print("\nInitialisierung:")
        #------domain--------
        from ngsolve.meshes import MakeStructured2DMesh
        Th = MakeStructured2DMesh(quads=False,nx=n_mesh,ny=n_mesh,periodic_x=True,periodic_y=True)
        # Specify the domain Omega (i.e. the microfluidic container's geometry)


        Vh  = Periodic(H1(Th,order=1)**N)#1
        Vh2 = Periodic(H1(Th,order=1)**N)#1
        Xh  = Periodic(VectorH1(Th, order=2))#2
        Qh  = Periodic(H1(Th,order=1))#1
        R   = NumberSpace(Th)
        FES = FESpace([Vh,Vh2,Xh,Qh,R,Qh])
        dummy = GridFunction(Qh)
        #-----FE functions-------
        rho,mu,u,p,lam,theta= GridFunction(FES).components
        rhoold,muold,uold,pold,lamold,thetaold= GridFunction(FES).components
        solution = GridFunction(FES)
        rhodiff,mudiff,udiff,pdiff,lamdiff,thetadiff=solution.components
        
        rhodelta,mudelta,udelta,pdelta,lamdelta,thetadelta = FES.TrialFunction()
        psi,xi,v,q,sig,w = FES.TestFunction()

        #-----Define Energy -----
        def TotEnergy(rho,theta,u):
            return CF(  E(rho,theta) + 0.5*(rho*Unit)*u**2)
        
        def Entropie(rho,theta):
            return CF(  S(rho,theta))

        #-------Evaluate functions at FE------
        f_rho = F(rho,theta).Diff(rho)
        MM = CF(M(rhoold,thetaold),dims=(N,N))
        KK = CF(K(rhoold,thetaold))
        eta_m = eta(rhoold,thetaold)
        f_rhorho = f_rho.Diff(rho)
        f_rhotheta = f_rho.Diff(theta)

        snew = -F(rho,theta).Diff(theta) #S(rho,theta)
        sold = -F(rhoold,thetaold).Diff(thetaold)# S(rhoold,thetaold)

        snew_rho = snew.Diff(rho)
        snew_theta = snew.Diff(theta)

        #-------Compiling for speed------
        f_rho      = f_rho.Compile()
        f_rhorho   = f_rhorho.Compile()
        f_rhotheta = f_rhotheta.Compile()
        MM         = MM.Compile()
        eta_m        = eta_m.Compile()
        snew       = snew.Compile()
        sold       = sold.Compile()
        snew_rho   = snew_rho.Compile()
        snew_theta = snew_theta.Compile()



        #------Newton weak form-------
        a=BilinearForm(FES)
        a+=1.0/dt*(rhodelta)*psi*dx - (Grad(psi)*udelta)*rhoold*dx + Tensorcontraction(mudelta,theta,thetaold,Grad(psi),MM)*dx\
            + TensorcontractionNewton(mu,theta,thetadelta,thetaold,Grad(psi),MM)*dx\
            + mudelta*xi*dx - f_rhorho*rhodelta*xi*dx - f_rhotheta*thetadelta*xi*dx - (Vvec*xi)*pdelta*dx\
            + 0.5/dt*((rhodelta-rhoold)*Unit)*(u*v)*dx + 0.5/dt*((rho-rhoold)*Unit)*(udelta*v)*dx  + 1.0/dt*(rhoold*Unit)*udelta*v*dx\
            + 0.5*(rhoold*Unit)*(Grad(udelta)*uold)*v*dx - 0.5*(rhoold*Unit)*(Grad(v)*(uold))*udelta*dx\
            + eta_m*InnerProduct(Sym(Grad(udelta)),Sym(Grad(v)))*dx - pdelta*div(v)*dx\
            + (Grad(mudelta)*v)*rhoold*dx - (Grad(pdelta)*v)*(Vvec*rhoold)*dx + sold*grad(thetadelta)*v*dx\
            + div(udelta)*(q)*dx + Tensorcontraction(mudelta,theta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + TensorcontractionNewton(mu,theta,thetadelta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + 1.0/dt*(snew_rho*rhodelta + snew_theta*thetadelta)*w*dx - sold*udelta*grad(w)*dx - 2*eta_m*InnerProduct(Sym(Grad(udelta)),Sym(Grad(u)))*w/theta*dx\
            + eta_m*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/(theta)**2*thetadelta*dx\
            - KK*(1.0/(thetaold*theta)*grad(thetadelta))*(1.0/thetaold*grad(w) -w/(theta*thetaold)*grad(theta))*dx \
            + KK*(1.0/(thetaold*theta**2)*thetadelta*grad(theta))*(1.0/thetaold*grad(w)-w/(theta*thetaold)*grad(theta))*dx \
            + KK*(1.0/(thetaold*theta)*grad(theta))*(w/(theta*thetaold)*grad(thetadelta))*dx \
            - KK*(1.0/(thetaold*theta)*grad(theta))*(w/(theta**2*thetaold)*thetadelta*grad(theta))*dx \
            -  Tensorcontraction2(mudelta,theta,thetaold,w,MM)*dx  -  Tensorcontraction2Newton(mu,theta,thetadelta,thetaold,w,MM)*dx\
            + lamdelta*q*dx + pdelta*sig*dx  
            
        L=LinearForm(FES)
        L+= -1.0*(1.0/dt*(rho-rhoold)*psi*dx - (Grad(psi)*u)*rhoold*dx + Tensorcontraction(mu,theta,thetaold,Grad(psi),MM)*dx\
            + mu*xi*dx - f_rho*xi*dx - (Vvec*xi)*p*dx\
            + 0.5/dt*((rho-rhoold)*Unit)*(u*v)*dx + 1.0/dt*(rhoold*Unit)*(u-uold)*v*dx\
            + 0.5*(rhoold*Unit)*(Grad(u)*uold)*v*dx - 0.5*(rhoold*Unit)*(Grad(v)*(uold))*u*dx\
            + eta_m*InnerProduct(Sym(Grad(u)),Sym(Grad(v)))*dx - p*div(v)*dx\
            + (Grad(mu)*v)*rhoold*dx - (Grad(p)*v)*(Vvec*rhoold)*dx + sold*grad(theta)*v*dx\
            + div(u)*q*dx + Tensorcontraction(mu,theta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + 1.0/dt*(snew-sold)*w*dx - sold*u*grad(w)*dx - eta_m*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/theta*dx\
            + KK*(1/(thetaold*theta)*grad(theta))*(1/thetaold*grad(w)-w/(theta*thetaold)*grad(theta))*dx\
            -  Tensorcontraction2(mu,theta,thetaold,w,MM)*dx\
            + lam*q*dx + p*sig*dx )


        #------Set IC/ compute Metdata-------
        rho.Interpolate(rho0)
        rhoold.Interpolate(rho0)
        theta.Interpolate(theta0)
        u.Interpolate(u0)
        El  = [Integrate(TotEnergy(rho,theta,u),Th,VOL)]
        Sl  = [Integrate(Entropie(rho,theta),Th,VOL)]
        Mr  = [Integrate(rho,Th,VOL)] 
        M2 = [Integrate(rho*Unit,Th,VOL)]
        C  = [Integrate(rho*Vvec-1,Th,VOL)]
        pickle.dump([rho.vec.data,mu.vec.data,u.vec.data,p.vec.data,theta.vec.data],open(outputpath+outputname+"_0.step","wb"))


        #--------Paraview output---------------
        if(paraout):
            dummy.Set(rho*Vvec-1)
            vtk = VTKOutput(ma=Th,coefs=[rho,mu,u,p,theta,dummy],names=["rho","mu","u","p","theta","Constraint"],filename=outputpath+"Bilder/"+outputname)
            vtk.Do(time=0)
        

        #--------TimeStepping----------
        t=0.

        for i in range(1,steps+1):
                
            t+=dt
            print("Zeitschritt"+str(i))
            # Update old values
            rhoold.vec.data   = rho.vec.data
            uold.vec.data     = u.vec.data
            thetaold.vec.data = theta.vec.data

            l = 0
            # Newton for CH
            newtonerror = 1. # newtonerror
            while newtonerror > iterror and l<itsteps:
                l+=1
                a.Assemble()
                L.Assemble()				
                solution.vec.data = a.mat.Inverse(FES.FreeDofs())*(L.vec)
                rhodiff,mudiff,udiff,pdiff,lamdiff,thetadiff = solution.components
                # Update solutions
                rho.vec.data   += rhodiff.vec.data
                mu.vec.data    += mudiff.vec.data
                u.vec.data     += udiff.vec.data
                p.vec.data     += pdiff.vec.data
                lam.vec.data   += lamdiff.vec.data
                theta.vec.data += thetadiff.vec.data

                # Newtonerror
                newtonerror=sqrt(Integrate(rhodiff**2 + mudiff**2 + udiff**2 + pdiff**2 + thetadiff**2, Th, VOL))
                print(newtonerror)

                if(math.isnan(newtonerror)):
                    raise StopIteration("newtonerror is NaN !!!")
                

            
            #--------Set new E,M----------
            El.append(Integrate(TotEnergy(rho,theta,u),Th,VOL)) # Energie
            Sl.append(Integrate(Entropie(rho,theta),Th,VOL))
            Mr.append(Integrate(rho,Th,VOL)) # Energie
            M2.append(Integrate(rho*Unit,Th,VOL))
            C.append(Integrate(rho*Vvec-1,Th,VOL))

            # Plot
            if(paraout):
                dummy.Interpolate(rho*Vvec-1)
                vtk.Do(time=t)
            pickle.dump([rho.vec.data,mu.vec.data,u.vec.data,p.vec.data,theta.vec.data],open(outputpath+outputname+"_"+str(i)+".step","wb"))

    #-------Finished output---------
    print("\n\n\nSimulation beendet:")
    print("\nn:",n_mesh,", dt:",dt,", T:",T,", Steps:",steps)
    print("\n Newtonueberschreitungen:",itcount," ,",newtonerrs)

    #-------Write metdata------------
        
    timefile = open(outputpath+outputname+"_timeoutput.txt", "w")  
    timefile.write("N: "+str(n_mesh)+"\n")
    timefile.write("dt: "+str(dt)+"\n")
    timefile.write("T: "+str(T)+"\n")
    timefile.write("Steps: "+str(steps)+"\n")
    timefile.write("Newtonueberschreitungen: " + str(itcount)+"\n")
    timefile.close()


    #-------Print out metadata---------
    fp = open(outputpath+outputname+"_TotEnergy.txt", "w")
    for item in El:
        fp.write("%s\n" % item)
    print('Done')

    fp = open(outputpath+outputname+"_Entropy.txt", "w")
    for item in Sl:
        fp.write("%s\n" % item)
    print('Done')

    fp = open(outputpath+outputname+"_Mass.txt", "w")
    for item in Mr:
        fp.write("%s\n" % item.NumPy())
    print('Done')

    fp = open(outputpath+outputname+"Total_Mass.txt", "w")
    for item in M2:
        fp.write("%s\n" % item)
    print('Done')

    fp = open(outputpath+outputname+"_Constraint.txt", "w")
    for item in C:
        fp.write("%s\n" % item)
    print('Done')