import math
import numpy as np
import sys 
import os
from ngsolve.solvers import *
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh as Make_2D
#SetNumThreads(22)
import pickle
#import print_options
np.set_printoptions(precision=16)

#-------Numerical Parameter--------
n_mesh = 1 # 1/n is meshsize
dt = 0.0001 # timestep
T = 0.1 # finaltime
N = 2 # number of component 
VvecD = [0.8,0.2]     # V Vector
MWSScale = 10**(-1)   # Scaling factor for MWS-matrix
Unit = [1 for i in range(N)]           

itsteps = 20 # max newtonsteps
iterror = 1e-10 # tolerance newton
#SetNumThreads(22) #limit num of threads
time_quad = 10 # quadrature order in time for f' ()
[xx,ww] = np.polynomial.legendre.leggauss(time_quad)

#------Saving params--------
outputpath = ""
paraout = True # Paraview Output   

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
                Ma.append(MWSScale*(rhop[i]-rhop[i]**2/rhoG))
            else:
                Ma.append(-MWSScale*rhop[i]*rhop[j]/rhoG)
     return(Ma)

def HeatCon(rhop,thetap,N):
     Ma = 10**(-2)
     return(Ma)


def Entropy(rhop,thetap,N):
     rhoG = rhop*Unit
     Ma = rhoG*log(thetap)
     for i in range(N):
       Ma -= rhop[i]*log(rhop[i]/rhoG)
     return(Ma)

def Energy(rhop,sp,N):
     Mr =0
     rhoG = rhop*Unit
     for i in range(N):
          Mr += rhop[i]*log(rhop[i]/rhoG)
     Ma = rhoG*exp(sp/rhoG + Mr/rhoG)
     return(Ma)

def Energy_rhop(rhop,sp,N):
     Mr = 0
     rhoG = rhop*Unit
     for i in range(N):
          Mr += rhop[i]*log(rhop[i]/rhoG)
     Ma = []
     for i in range(N):
        Ma.append(exp(sp/rhoG + Mr/rhoG)*(log(rhop[i]/rhoG) - (sp/rhoG + Mr/rhoG) + 1))
     return(Ma)

def Energy_sp(rhop,sp,N):
     Mr =0
     rhoG = rhop*Unit
     for i in range(N):
          Mr += rhop[i]*log(rhop[i]/rhoG)
     Ma = exp(sp/rhoG + Mr/rhoG)
     return(Ma)

def etaForm(rhop,thetap,N):
     Ma = 10**(-2)
     #rhoG = rhop*Unit
     for i in range(N):
       Ma += 0*rhop[i]**2
     return(Ma)

def Tensorcontraction(mu,theta,GradPsi,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:] - mu[i]/(theta*theta)*Grad(theta))*GradPsi[j,:])
    return(Ma)

def Tensorcontraction2(mu,theta,w,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:] - mu[i]/(theta*theta)*Grad(theta))*\
                     (w/theta*Grad(mu)[j,:] + mu[j]/theta*Grad(w) - w*mu[j]/(theta*theta)*Grad(theta)))
    return(Ma)

def TensorcontractionNewton(mu,theta,thetadelta,GradPsi,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF(M[i,j]*(-1/theta**2*thetadelta*Grad(mu)[i,:] + 2*mu[i]/(theta**3)*thetadelta*Grad(theta) -2*mu[i]/theta**2*Grad(thetadelta))*GradPsi[j,:])
    return(Ma)

def Tensorcontraction2Newton(mu,theta,thetadelta,w,M):
    Ma = CF(0)
    for i in range(N):
        for j in range(N):
            Ma += CF( M[i,j]*(-1/theta**2*thetadelta*Grad(mu)[i,:] + 2*mu[i]/(theta**3)*thetadelta*Grad(theta) - 2*mu[i]/theta**2*Grad(thetadelta))*\
                     (w/theta*Grad(mu)[j,:] + mu[j]/theta*Grad(w) - w*mu[j]/(theta*theta)*Grad(theta)))
            Ma += CF( M[i,j]*(  1/theta*Grad(mu)[i,:] - mu[i]/(theta*theta)*Grad(theta)  )*\
                     ( -w/theta**2*thetadelta*Grad(mu)[j,:] - mu[j]/theta**2*thetadelta*Grad(w)\
                      + 2*w*mu[j]/(theta**3)*thetadelta*Grad(theta) - w*mu[j]/(theta*theta)*Grad(thetadelta) ) )
    return(Ma)


M    = lambda rhop,thetap: tuple(MWS(rhop,thetap,N))
Entropy_val= lambda rhop,thetap: Entropy(rhop,thetap,N)
varrhoE    = lambda rhop,sp: Energy(rhop,sp,N)
varrhoE_rhop    = lambda rhop,sp: tuple(Energy_rhop(rhop,sp,N))
varrhoE_sp    = lambda rhop,sp: Energy_sp(rhop,sp,N)
K    = lambda rhop,thetap: HeatCon(rhop,thetap,N)
etaR  = lambda rhop,thetap: etaForm(rhop,thetap,N)

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

print("\nProgrammstart:")
print("n = ", n_mesh )
print("h = ", 1.0/n_mesh )
print("dt = ", dt )
print("T = ", T )
print("steps = ", steps )

print("\nInitialisierung:")


with TaskManager():

    for nn in range(1,9):
        n_mesh = 2**nn
        outputname = "Nonisothermal_MWS_NS"+str(n_mesh)
        #------domain--------
        from ngsolve.meshes import MakeStructured2DMesh
        Th = MakeStructured2DMesh(quads=False,nx=n_mesh,ny=n_mesh,periodic_x=True,periodic_y=True)

        Vh  = Periodic(H1(Th,order=1)**N)#1
        Xh = Periodic(VectorH1(Th, order=2))#2
        Qh  = Periodic(H1(Th,order=1))#1
        R   = NumberSpace(Th)
        FES = FESpace([Vh,Vh,Xh,Xh,Qh,R,Qh,Qh])
        dummy = GridFunction(Qh)
        #-----FE functions-------
        rho,mu,u,m,p,lam,snew,theta= GridFunction(FES).components
        rhoold,muold,uold,mold,pold,lamold,sold,thetaold= GridFunction(FES).components
        solution = GridFunction(FES)
        rhodiff,mudiff,udiff,mdiff,pdiff,lamdiff,sdiff,thetadiff=solution.components
        
        rhodelta,mudelta,udelta,mdelta,pdelta,lamdelta,sdelta,thetadelta = FES.TrialFunction()
        psi,xi,v,vv,q,sig,w,gamma = FES.TestFunction()

        #-----Define Energy -----
        def TotEnergy(rho,snew,m):
            return CF(  varrhoE(rho,snew) + 0.5*m*m)

        #-------Evaluate functions at FE------
        #f_rho =CF(0*varrhoE_rhop(rho,snew))
        f_rho = sum(0.5*ww[i]*CF(varrhoE_rhop(0.5*(xx[i]+1.0)*rho + 0.5*(1.0-xx[i])*rhoold,\
                                            0.5*(xx[i]+1.0)*snew + 0.5*(1.0-xx[i])*sold)) for i in range(time_quad) )

        MM = CF(M(rhoold,sold),dims=(N,N))
        KK = CF(K(rhoold,sold))
        eta = etaR(rhoold,sold)
        f_rho_rho = f_rho.Diff(rho)
        f_rho_s = f_rho.Diff(snew)

        varrhoE_s = sum(CF(0.5*ww[i]*varrhoE_sp(0.5*(xx[i]+1.0)*rho + 0.5*(1.0-xx[i])*rhoold,\
                                            0.5*(xx[i]+1.0)*snew + 0.5*(1.0-xx[i])*sold)) for i in range(time_quad) )

        sqrtrho = sum(CF(0.5*ww[i]*sqrt(0.5*(xx[i]+1.0)*rho*Unit + 0.5*(1.0-xx[i])*rhoold*Unit)) for i in range(time_quad) )
        sqrtrho_rho=sqrtrho.Diff(rho)

        varrhoE_s_s = varrhoE_s.Diff(snew)

        #-------Compiling for speed------
        f_rho      = f_rho.Compile()
        f_rho_rho   = f_rho_rho.Compile()
        f_rho_s = f_rho_s.Compile()
        MM         = MM.Compile()
        eta        = eta.Compile()
        varrhoE_s  = varrhoE_s.Compile()
        varrhoE_s_s     = varrhoE_s_s.Compile()


        my_rule = IntegrationRule(TRIG, order=10)

        # Use this rule in a bilinear form
        dx = dx(intrules={TRIG: my_rule})

        #------Newton weak form-------
        a=BilinearForm(FES)
        a+=1.0/dt*rhodelta*psi*dx - 0.5*(Grad(psi)*(udelta))*(rho+rhoold)*dx - 0.5*(Grad(psi)*(u))*(rhodelta)*dx\
            + Tensorcontraction(mudelta,theta,Grad(psi),MM)*dx  + TensorcontractionNewton(mu,theta,thetadelta,Grad(psi),MM)*dx\
            + mudelta*xi*dx - f_rho_rho*rhodelta*xi*dx - f_rho_s*sdelta*xi*dx - (Vvec*xi)*pdelta*dx\
            + 1.0/dt*sqrtrho*mdelta*v*dx + 1.0/dt*sqrtrho_rho*rhodelta*(m-mold)*v*dx\
            + (0.5)*(rhodelta*Unit)*((Grad(u))*(u))*v*dx\
            - (0.5)*(rhodelta*Unit)*(Grad(v)*(u))*(u)*dx\
            + (0.5)*(rhoold*Unit+rho*Unit)*((Grad(udelta))*(u))*v*dx\
            - (0.5)*(rhoold*Unit+rho*Unit)*(Grad(v)*(udelta))*(u)*dx\
            + (0.5)*(rhoold*Unit+rho*Unit)*((Grad(u))*(udelta))*v*dx\
            - (0.5)*(rhoold*Unit+rho*Unit)*(Grad(v)*(u))*(udelta)*dx\
            + eta*InnerProduct(Sym(Grad(udelta)),Sym(Grad(v)))*dx - pdelta*div(v)*dx\
            + 0.5*(Grad(mudelta)*v)*(rhoold+rho)*dx - 0.5*(Grad(pdelta)*v)*(Vvec*rhoold+Vvec*rho)*dx + 0.5*(sdelta)*grad(theta)*v*dx\
            + 0.5*(Grad(mu)*v)*(rhodelta)*dx - 0.5*(Grad(p)*v)*(Vvec*rhodelta)*dx + 0.5*(snew+sold)*grad(thetadelta)*v*dx\
            + div(udelta)*q*dx + Tensorcontraction(mudelta,theta,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + TensorcontractionNewton(mu,theta,thetadelta,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + 0.5*mdelta*vv*dx - sqrtrho*udelta*vv*dx - sqrtrho_rho*rhodelta*u*vv*dx\
            + lamdelta*q*dx - sig*pdelta*dx\
            + 1.0/dt*sdelta*w*dx - 0.5*sdelta*u*grad(w)*dx  - 0.5*(snew+sold)*udelta*grad(w)*dx\
            - 2*eta*InnerProduct(Sym(Grad(udelta)),Sym(Grad(u)))*w/theta*dx\
            + eta*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/theta**2*thetadelta*dx\
            + KK*(1/(theta**2)*grad(thetadelta))*(1/theta*grad(w)-w/(theta**2)*grad(theta))*dx \
            - 2*KK*(1/(theta**3)*thetadelta*grad(theta))*(1/theta*grad(w)-w/(theta**2)*grad(theta))*dx \
            - KK*(1/(theta**2)*grad(theta))*(1/theta**2*thetadelta*grad(w)-2*w/(theta**3)*thetadelta*grad(theta))*dx\
            - KK*(1/(theta**2)*grad(theta))*(w/(theta**2)*grad(thetadelta))*dx \
            -  Tensorcontraction2(mudelta,theta,w,MM)*dx -  Tensorcontraction2Newton(mu,theta,thetadelta,w,MM)*dx \
            + thetadelta*gamma*dx - f_rho_s*rhodelta*gamma*dx - varrhoE_s_s*sdelta*gamma*dx
        

        L=LinearForm(FES)
        L+= -1.0*(1.0/dt*(rho-rhoold)*psi*dx - 0.5*(Grad(psi)*u)*(rho+rhoold)*dx + Tensorcontraction(mu,theta,Grad(psi),MM)*dx\
            + mu*xi*dx - f_rho*xi*dx - (Vvec*xi)*p*dx\
            + 1.0/dt*sqrtrho*(m-mold)*v*dx\
            + (0.5)*(rhoold*Unit+rho*Unit)*(Grad(u)*u)*v*dx\
            - (0.5)*(rhoold*Unit+rho*Unit)*(Grad(v)*u)*u*dx\
            + eta*InnerProduct(Sym(Grad(u)),Sym(Grad(v)))*dx - p*div(v)*dx\
            + 0.5*(Grad(mu)*v)*(rhoold+rho)*dx - 0.5*(Grad(p)*v)*Vvec*(rhoold+rho)*dx \
            + 0.5*(snew+sold)*grad(theta)*v*dx\
            + div(u)*q*dx + Tensorcontraction(mu,theta,OuterProduct(Vvec,Grad(q)),MM)*dx\
            + 0.5*(m+mold)*vv*dx - sqrtrho*u*vv*dx\
            + lam*q*dx - sig*p*dx\
            + 1.0/dt*(snew-sold)*w*dx - 0.5*(snew+sold)*u*grad(w)*dx\
            - eta*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/theta*dx\
            + KK*(1/(theta*theta)*grad(theta))/theta*(grad(w)-w/theta*grad(theta))*dx\
            -  Tensorcontraction2(mu,theta,w,MM)*dx \
            + theta*gamma*dx - varrhoE_s*gamma*dx)


        #------Set IC/ compute Metdata-------
        rho.Interpolate(rho0)
        snew.Interpolate(Entropy_val(rho,theta0))
        theta.Interpolate(varrhoE_sp(rho,snew))
        mu.Interpolate(varrhoE_rhop(rho,snew))
        u.Interpolate(u0)
        m.Interpolate(sqrt(rho*Unit)*u0)

        El  = [Integrate(TotEnergy(rho,snew,m),Th,VOL,order=10)]
        Sl  = [Integrate(snew,Th,VOL)]
        Mr  = [Integrate(rho,Th,VOL)] 
        M2 = [Integrate(rho*Unit,Th,VOL)]
        C  = [Integrate(rho*Vvec-1,Th,VOL)]
        pickle.dump([rho.vec.data,mu.vec.data,m.vec.data,u.vec.data,p.vec.data,snew.vec.data,theta.vec.data],open(outputpath+outputname+"_0.step","wb"))


        #--------Paraview output---------------
        if(paraout):
            dummy.Set(rho*Vvec-1)
            vtk = VTKOutput(ma=Th,coefs=[rho,mu,u,m,p,snew,theta,dummy],names=["rho","mu","u","m","p","s","theta","Constraint"],filename=outputpath+"Bilder/"+outputname)
            vtk.Do(time=0)
        

        #--------TimeStepping----------
        t=0.

        for i in range(1,steps+1):
                
            t+=dt
            print("Zeitschritt"+str(i))
            # Update old values
            rhoold.vec.data   = rho.vec.data
            mold.vec.data     = m.vec.data
            sold.vec.data     = snew.vec.data


            l = 0
            # Newton for CH
            newtonerror = 1. # newtonerror
            while newtonerror > iterror and l<itsteps:
                l+=1
                a.Assemble()
                L.Assemble()				
                solution.vec.data = a.mat.Inverse(FES.FreeDofs())*(L.vec)
                # Update solutions
                rho.vec.data   += rhodiff.vec.data
                mu.vec.data    += mudiff.vec.data
                u.vec.data     += udiff.vec.data
                m.vec.data     += mdiff.vec.data
                p.vec.data     += pdiff.vec.data
                lam.vec.data   += lamdiff.vec.data
                snew.vec.data  += sdiff.vec.data
                theta.vec.data += thetadiff.vec.data
                # Newtonerror
                newtonerror=sqrt(Integrate(rhodiff**2 + mudiff**2 + udiff**2 + mdiff**2  + pdiff**2 + thetadiff**2 + sdiff**2, Th, VOL))
                print(newtonerror)
                if(math.isnan(newtonerror)):
                    raise StopIteration("newtonerror is NaN !!!")
                

            
            #--------Set new E,M----------
            El.append(Integrate(TotEnergy(rho,snew,m),Th,VOL,order=10)) # Energie
            #print("Expandend full:",Integrate((snew-sold)*theta +(rho-rhoold)*(mu-Vvec*p) + 0.5*(m+mold)*(m-mold),Th,VOL,order=10))
            #print("Expandend full:",Integrate((snew-sold)*(theta-varrhoE_s) +(rho-rhoold)*(f_rho-mu+Vvec*p) + (0.5*(m+mold)-sqrtrho*u)*(m-mold),Th,VOL,order=10))
            #print("Non-expandend:",Integrate(TotEnergy(rho,snew,m)-TotEnergy(rhoold,sold,mold),Th,VOL,order=10))
            #test = Integrate((0.5*(m+mold)*(m-mold)-sqrtrho*(m-mold)*u)**2,Th,VOL)
            #print(test)
            Sl.append(Integrate(snew,Th,VOL))
            Mr.append(Integrate(rho,Th,VOL)) # Energie
            M2.append(Integrate(rho*Unit,Th,VOL))
            C.append(Integrate(rho*Vvec-1,Th,VOL))
            pickle.dump([rho.vec.data,mu.vec.data,m.vec.data,u.vec.data,p.vec.data,snew.vec.data,theta.vec.data],open(outputpath+outputname+"_"+str(i)+".step","wb"))
            # Plot
            if(paraout):
                dummy.Interpolate(rho*Vvec-1)
                vtk.Do(time=t)
            

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