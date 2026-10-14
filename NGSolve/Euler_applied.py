import math
import numpy as np
import sys 
import os
from ngsolve.solvers import *
from ngsolve import *
import netgen.occ as ngocc

#import print_options
np.set_printoptions(precision=16)

#-------Numerical Parameter--------
maxh = 0.075 # mesh size

dt = 1e-2# timestep
Ti = 5 # finaltime
N = 2 # number of component 
itsteps = 40 # max newtonsteps
iterror = 1e-7 # tolerance newton

free = 0
ap = 0 # =1 really robin temperature, =0 (rhos*u-J_s)*n= Robin

#SetNumThreads(5) #limit num of threads

#------Saving params--------
outputpath = ""
outputname = "Paper_test_scaledenergy2"
paraout = True # Paraview Output  

eta = (6e-4)            # Shear viscosity, Pa s
zeta = (1e-7)  

lame_ND = (zeta / eta) - 1.0


# Constants for the pure species (at the ambient temperature and pressure)
M_1 = (0.078)               # Molar mass of benzene, kg / mol
M_2 = (0.084)               # Molar mass of cyclohexane, kg / mol
rho_pure_1 = (876)          # Density of pure benzene, kg / m^3
rho_pure_2 = (773)          # Density of pure cyclohexane, kg / m^3
c_pure_1 = rho_pure_1 / M_1         # Concentration of pure benzene, mol / m^3
c_pure_2 = rho_pure_2 / M_2         # Concentration of pure cyclohexane, mol / m^3

# Constants for the equimolar mixture
c_equi_tot = 1.0 / ((0.5 / c_pure_2) + (0.5 / c_pure_1))  # Total equimolar concentration, mol / m^3
c_equi_1 = 0.5 * c_equi_tot                     # Equimolar concentration of benzene, mol / m^3
c_equi_2 = 0.5 * c_equi_tot                     # Equimolar concentration of cyclohexane, mol / m^3
rho_equi = (M_1 * c_equi_1) + (M_2 * c_equi_2)  # Equimolar density, kg / m^3

# Reference concentration, density and molar mass
rho_ref = rho_equi                              # Reference density, kg / m^3
c_ref = c_equi_tot                              # Reference concentration, mol / m^3
M_ref = rho_ref / c_ref                         # Reference molar mass, kg / mol

v_ref_1 = (0.4e-6)                      # Reference inflow velocity of benzene, m / s
                                                # This can be increased if one desires a greater Peclet number flow
v_ref_2 = (c_pure_1 / c_pure_2) * v_ref_1       # Reference inflow velocity of cyclohexane, m / s
v_ref = (0.5 * (v_ref_1 + v_ref_2))         # Reference velocity, m / s

#velo = [v_ref_1,(c_pure_1 / c_pure_2) * v_ref_1  ]
L_ref = (2e-3)          # Reference length, m
p_ref = eta * v_ref / L_ref     # Reference pressure, Pa

# Non-dimensionalised molar masses
M_1_ND = M_1 / M_ref
M_2_ND = M_2 / M_ref

# Non-dimensionalised partial specific volumes
V_1_ND = rho_ref / rho_pure_1
V_2_ND = rho_ref / rho_pure_2

# Free energy of the mixture
T_ref = (298.15)        # Reference temperature, K
R = (8.314)             # Ideal gas constant, J / K * mol

mu_ref = R * T_ref              # Reference chemical potential, J / mol

Me_1 = p_ref / (mu_ref * c_pure_1)      # Non-dimensionalised partial molar volume of benzene, dimensionless
Me_2 = p_ref / (mu_ref * c_pure_2)      # Non-dimensionalised partial molar volume of cyclohexane, dimensionless

# Margules model parameters
A_12 = (0.4498)         # Dimensionless
A_21 = (0.4952)         # Dimensionless

# Partial molar heat capacities of the species
Cp_1 = (135.69)         # Of Benzene, J / K * mol
Cp_2 = (156.00)         # Of Cyclohexane, J / K * mol

k_ref = (0.125)  
k_1 = (0.14)                            # Thermal conductivity of benzene, W / m * K
k_2 = (0.11)

D_sm = (2.1e-9)                         # Stefan--Maxwell diffusivity, m^2 / s


Pe = v_ref * L_ref / D_sm 
Pe_k = v_ref * L_ref * (c_ref * R / k_ref) 
Me = p_ref / (R * T_ref * c_ref) 
Re = (rho_ref * v_ref * L_ref) / eta 

print("Peclet number",Pe)
print("Thermal Peclet number",Pe_k)
print("Pressure diffusion",Me)
print("Reynolds number",Re)

wall_thickness = L_ref * (0.01)                 # Thickness of the wall, m
wall_tc = k_ref * (0.1)                         # Wall thermal conductivity, W / m * K
wcf = (wall_thickness / L_ref) * (k_ref / wall_tc)



#T_wall = CF(1 + (0.1 * (((y + 1) / 2) ** 10)) + (0.05 * ((x / 4) ** 10)))
#T_wall = 1 + 0.15 * ((((y + 1)/2)**10) * (1 - (x/4)**2)**10)
T_wall = 1 + 0.1 * ((y + 1) / 2)


#------Parameterfunctions & IC------
def MWS(rhop,N):
     Ma = []
     #rhoG = rhop*Unit
     x1 = (rhop[0] / M_1_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     x2 = (rhop[1] / M_2_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     xx  = x1*M_1_ND + x2*M_2_ND
     val = rhop[0]*rhop[1]*M_1_ND*M_2_ND/(rhop*Unit*xx)
     Ma = val*CF((1,-1,-1,1),dims=(2,2))
     return(Ma)

def HeatCon(rhop,thetap,N):
     x1 = (rhop[0] / M_1_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     x2 = (rhop[1] / M_2_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     Ma = ((k_1 * x1) + (k_2 * x2))*thetap**2/k_ref 
     return(Ma)


def Pot(rhop,thetap,N):
     x1 = (rhop[0] / M_1_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     x2 = (rhop[1] / M_2_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     Ma = 1.0/R*(-rhop[0]*Cp_1/(M_1_ND)*thetap*(log(thetap)-1) -rhop[1]*Cp_2/(M_2_ND)*thetap*(log(thetap)-1))
     Ma += thetap*rhop[0]/(M_1_ND)*log(x1) + thetap*rhop[0]/(M_1_ND)*(x2 ** 2 * (A_12 + (2.0 * (A_21 - A_12) * x1)))
     Ma += thetap*rhop[1]/(M_2_ND)*log(x2) + thetap*rhop[1]/(M_2_ND)*(x1 ** 2 * (A_21 + (2.0 * (A_12 - A_21) * x2)))
     return(Ma)

def Entropy(rhop,thetap,N):
     x1 = (rhop[0] / M_1_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     x2 = (rhop[1] / M_2_ND) / ((rhop[0] / M_1_ND) + (rhop[1] / M_2_ND))
     Ma = 1.0/R*(rhop[0]*Cp_1/(M_1_ND)*log(thetap) + rhop[1]*Cp_2/(M_2_ND)*log(thetap))
     Ma -= rhop[0]/(M_1_ND)*log(x1) + rhop[0]/(M_1_ND)*(x2 ** 2 * (A_12 + (2.0 * (A_21 - A_12) * x1)))
     Ma -= rhop[1]/(M_2_ND)*log(x2) + rhop[1]/(M_2_ND)*(x1 ** 2 * (A_21 + (2.0 * (A_12 - A_21) * x2)))
     return(Ma)

def Energy(rhop,thetap,N):
     Ma = (rhop[0]*Cp_1/(M_1_ND)+rhop[1]*Cp_2/(M_2_ND))*thetap/R
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


M    = lambda rhop,thetap: MWS(rhop,N)
F    = lambda rhop,thetap: Pot(rhop,thetap,N)
S    = lambda rhop,thetap: Entropy(rhop,thetap,N)
E    = lambda rhop,thetap: Energy(rhop,thetap,N)
K    = lambda rhop,thetap: HeatCon(rhop,thetap,N)

Unit = [1 for i in range(N)]   
Vvec = CF((V_1_ND,V_2_ND))
Unit = CF(tuple(Unit))
#MDIVec = CF(tuple(MDIVec))

x_1_init = (0.5)
#x_2_init = 1 - x_1_init
c_1_init = x_1_init * (c_equi_tot / c_ref)
rho0 = CF((M_1_ND * c_1_init,(1 - (M_1_ND * c_1_init * V_1_ND)) / V_2_ND))

u0 = CF((0,0))
theta0 = CF(1.0)



#-------Metavariables---------
itcount = 0 # Newton Ãœberschreitungen
steps = math.ceil(Ti/dt) # Anzahl Zeitschritte
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

print("dt = ", dt )
print("T = ", Ti )
print("steps = ", steps )

print("\nInitialisierung:")


with TaskManager():
    #------domain--------
    wp = ngocc.WorkPlane()
    wp.MoveTo(0, 1)
    wp.Spline([ngocc.Pnt(1, 0), ngocc.Pnt(0, -1)])
    wp.LineTo(1, -1)
    wp.Spline([ngocc.Pnt(3, -0.5), ngocc.Pnt(4, -0.5)], tangents={ 1 : ngocc.gp_Vec2d(1, 0) })
    wp.LineTo(4, 0.5)
    wp.Spline([ngocc.Pnt(3, 0.5), ngocc.Pnt(1, 1)], tangents={ 0 : ngocc.gp_Vec2d(-1, 0) })
    wp.LineTo(0, 1)
    Omega = wp.Face()



    # Label the inlets and outlet;
    Omega.edges.Max(ngocc.Y).name = "inlet1"
    Omega.edges.Min(ngocc.Y).name = "inlet2"
    Omega.edges.Max(ngocc.X).name = "outlet"

    for e in Omega.edges:
        if e.name not in ("inlet1", "inlet2", "outlet"):
            e.name = "wall"
    


    # Construct the mesh
    ngmesh = ngocc.OCCGeometry(Omega, dim=2).GenerateMesh(maxh=maxh)
    Th = Mesh(ngmesh)

    

    Vh  = (H1(Th,order=1)**N)#1
    Vh2 = (H1(Th,order=1)**N)#1
    Xh  = (VectorH1(Th, order=2, dirichlet="wall|inlet1|inlet2"))#2dirichlet="^(?!outlet$).*"dirichlet="wall|inlet1|inlet2"
    Qh  = (H1(Th,order=1))#1
    RR   = NumberSpace(Th)
    FES = FESpace([Vh,Vh2,Xh,Qh,RR,Qh])
    dummy = GridFunction(Qh)

    #-----FE functions-------
    rho,mu,u,p,lam,theta= GridFunction(FES).components
    rhoold,muold,uold,pold,lamold,thetaold= GridFunction(FES).components
    solution = GridFunction(FES)
    rhodiff,mudiff,udiff,pdiff,lamdiff,thetadiff=solution.components
    
    rhodelta,mudelta,udelta,pdelta,lamdelta,thetadelta = FES.TrialFunction()
    psi,xi,v,q,sig,w = FES.TestFunction()

    x12 = GridFunction(Vh)


    test1 = GridFunction(Qh)
    test2 = GridFunction(Qh)

   # test1.Interpolate(1, definedon=Th.Boundaries("^(?!outlet$).*"))
    #test2.Interpolate(1, definedon=Th.Boundaries("inlet1|inlet2|wall"))

    #vtkA = VTKOutput(
    #    ma=Th,
    #    coefs=[test1,test2],
    #    names=["solution1","solution2"],
    #    filename=outputpath+"Bilder/"+ outputname+"_test"
    #)      
    #vtkA.Do()

    #-----Define Energy -----
    def TotEnergy(rho,theta,u):
        return CF(  E(rho,theta) + 0.5*(rho*Unit)*u**2)
    
    def Entropie(rho,theta):
        return CF(  S(rho,theta))

    #-------Evaluate functions at FE------  
    f_rho = F(rho,theta).Diff(rho) 
    MM = M(rhoold,thetaold)
    KK = CF(K(rhoold,thetaold))
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
    snew       = snew.Compile()
    sold       = sold.Compile()
    snew_rho   = snew_rho.Compile()
    snew_theta = snew_theta.Compile()

    n = specialcf.normal(2)

    def Tensorcontraction_BC(mu,theta,thetaold,Psi,M):
        Ma = CF(0)
        for i in range(N):
            for j in range(N):
                Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:])*Psi[j]*n)
        return(Ma)


    def Tensorcontraction2_BC(mu,theta,thetaold,w,M):
        Ma = CF(0)
        for i in range(N):
            for j in range(N):
                Ma += CF(M[i,j]*(1/theta*Grad(mu)[i,:])*(w*mu[j]*n/theta))
        return(Ma)

    J1in1 = CF(-M_1_ND * (v_ref_1 / v_ref) * (c_pure_1 / c_ref)*2*x*(x-1))*CF((2,-1))
    J2in2 = CF(-M_2_ND * (v_ref_2 / v_ref) * (c_pure_2 / c_ref)*2*x*(x-1))*CF((2,1))

    #J1out = CF(-M_1_ND * (v_ref_1 / v_ref) * (c_pure_1 / c_ref)*2*(y+0.5)*(y-0.5))*CF((1,0))
    #J2out = CF(-M_2_ND * (v_ref_2 / v_ref) * (c_pure_2 / c_ref)*2*(y+0.5)*(y-0.5))*CF((1,0))

    uBCI1 = CF(J1in1/(rhoold*Unit))
    uBCI2 = CF(J2in2/(rhoold*Unit))
    #uBCO  = CF(((J1out+J2out)/(rhoold*Unit)))

    dS1 = ds(definedon=Th.Boundaries("inlet1"))
    dS2 = ds(definedon=Th.Boundaries("inlet2"))
    #dO  = ds(definedon=Th.Boundaries("outlet"))


    #------Newton weak form-------
    a=BilinearForm(FES)
    a+=  1.0*(1.0/dt*(rhodelta)*psi*dx - (Grad(psi)*udelta)*rhoold*dx + 1/Pe*Tensorcontraction(mudelta,theta,thetaold,Grad(psi),MM)*dx\
        + 1/Pe*TensorcontractionNewton(mu,theta,thetadelta,thetaold,Grad(psi),MM)*dx +rhoold*psi*udelta*n*ds("outlet")\
        #- ( free/Pe*Tensorcontraction_BC(mudelta,theta,thetaold,psi,MM))*ds("outlet", skeleton=True)\
        + mudelta*xi*dx - f_rhorho*rhodelta*xi*dx - f_rhotheta*thetadelta*xi*dx - (Me_1/M_1_ND*xi[0]+Me_2/M_2_ND*xi[1])*pdelta*dx\
        + Re*0.5/dt*((rhodelta-rhoold)*Unit)*(u*v)*dx + Re*0.5/dt*((rho-rhoold)*Unit)*(udelta*v)*dx  + Re*1.0/dt*(rhoold*Unit)*udelta*v*dx\
        + Re*0.5*(rhoold*Unit)*(Grad(udelta)*uold)*v*dx - Re*0.5*(rhoold*Unit)*(Grad(v)*(uold))*udelta*dx\
        + 0.5*Re*rhoold*Unit*IfPos(u*n,u*n,0)*udelta*v*ds("outlet")\
		+ 2.0*InnerProduct(Sym(Grad(udelta)),Sym(Grad(v)))*dx + lame_ND*div(udelta)*div(v)*dx - pdelta*div(v)*dx\
        - udelta*grad(q)*dx + 1/Pe*Tensorcontraction(mudelta,theta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx + udelta*q*n*ds("outlet")\
        #- (free/Pe*Tensorcontraction(mudelta,theta,thetaold,OuterProduct(Vvec,q*n),MM))*ds("outlet", skeleton=True)\
        + 1/Pe*TensorcontractionNewton(mu,theta,thetadelta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx\
        + 1.0/dt*(snew_rho*rhodelta + snew_theta*thetadelta)*w*dx - sold*udelta*grad(w)*dx\
        - Me*4*InnerProduct(Sym(Grad(udelta)),Sym(Grad(u)))*w/theta*dx - 2.0*lame_ND*Me*div(u)*div(udelta)*w/theta*dx\
        + Me*2*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/(theta**2)*thetadelta*dx + lame_ND*Me*div(u)**2*w/(theta**2)*thetadelta*dx\
        + 1/Pe_k*KK*(1.0/(thetaold*theta)*grad(thetadelta))*(1.0/thetaold*grad(w) -w/(theta*thetaold)*grad(theta))*dx \
        - 1/Pe_k*KK*(1.0/(thetaold*theta**2)*thetadelta*grad(theta))*(1.0/thetaold*grad(w)-w/(theta*thetaold)*grad(theta))*dx \
        - 1/Pe_k*KK*(1.0/(thetaold*theta)*grad(theta))*(w/(theta*thetaold)*grad(thetadelta))*dx \
        + 1/Pe_k*KK*(1.0/(thetaold*theta)*grad(theta))*(w/(theta**2*thetaold)*thetadelta*grad(theta))*dx \
        - 1/Pe*Tensorcontraction2(mudelta,theta,thetaold,w,MM)*dx  -  1/Pe*Tensorcontraction2Newton(mu,theta,thetadelta,thetaold,w,MM)*dx\
        + 1/(Pe_k*wcf)*(T_wall/(theta**2)*thetadelta)*w*ds("inlet1|inlet2") \
        #+ ap*(J1in1*n*mudelta[0]/theta-(rhoold*Unit*uBCI1*n)*(mudelta[0]+mudelta[1])/theta)*w*dS1 \
        #+ ap*(J2in2*n*mudelta[0]/theta-(rhoold*Unit*uBCI2*n)*(mudelta[0]+mudelta[1])/theta)*w*dS2\
        #- ap*(J1in1*n*mu[0]/(theta**2)*thetadelta-(rhoold*Unit*uBCI1*n)*(mu[0]+mu[1])/(theta**2)*thetadelta)*w*dS1 \
        #- ap*(J2in2*n*mu[0]/(theta**2)*thetadelta-(rhoold*Unit*uBCI2*n)*(mu[0]+mu[1])/(theta**2)*thetadelta)*w*dS2\
        #+ (free/Pe*Tensorcontraction2_BC(mudelta,theta,thetaold,w,MM))*ds("outlet", skeleton=True)\
        + sold*udelta*w*n*ds("outlet"))
           #+ 1.0/Me*( (Grad(mudelta)*v)*rhoold - Me*(Grad(pdelta)*v)*(V_1_ND*rhoold[0]+V_2_ND*rhoold[1]) + sold*grad(thetadelta)*v)*dx  \    

    L=LinearForm(FES)
    L+= -1.0*(1.0/dt*(rho-rhoold)*psi*dx - (Grad(psi)*u)*rhoold*dx + 1/Pe*Tensorcontraction(mu,theta,thetaold,Grad(psi),MM)*dx\
        + J1in1*n*psi[0]*dS1 + J2in2*n*psi[1]*dS2 + rhoold*psi*u*n*ds("outlet")\
        #- (free/Pe*Tensorcontraction_BC(mu,theta,thetaold,psi,MM))*ds("outlet", skeleton=True)\
        + mu*xi*dx - f_rho*xi*dx - (Me_1/M_1_ND*xi[0]+Me_2/M_2_ND*xi[1])*p*dx\
        + Re*0.5/dt*((rho-rhoold)*Unit)*(u*v)*dx + Re*1.0/dt*(rhoold*Unit)*(u-uold)*v*dx\
        + Re*0.5*(rhoold*Unit)*(Grad(u)*uold)*v*dx - Re*0.5*(rhoold*Unit)*(Grad(v)*(uold))*u*dx\
        + 0.5*Re*rhoold*Unit*IfPos(u*n,u*n,0)*u*v*ds("outlet")\
		+ 2.0*InnerProduct(Sym(Grad(u)),Sym(Grad(v)))*dx + lame_ND*div(u)*div(v)*dx - p*div(v)*dx\
		#+ 1.0/Me*( (Grad(mu)*v)*rhoold - Me*(Grad(p)*v)*(V_1_ND*rhoold[0]+V_2_ND*rhoold[1]) + sold*grad(theta)*v)*dx\
        - u*grad(q)*dx + 1/Pe*Tensorcontraction(mu,theta,thetaold,OuterProduct(Vvec,Grad(q)),MM)*dx + (V_1_ND*rhoold[0]+V_2_ND*rhoold[1])*u*q*n*ds("outlet")\
        - (V_1_ND * rhoold[0] + V_2_ND * rhoold[1] - 1)*u*n*q*ds("outlet")\
        #- 1/dt*(V_1_ND * rhoold[0] + V_2_ND * rhoold[1] - 1)*q*dx\
        #- (free/Pe*Tensorcontraction(mu,theta,thetaold,OuterProduct(Vvec,q*n),MM))*ds("outlet", skeleton=True)\
        + (V_1_ND*J1in1*n*q)*dS1 + (V_2_ND*J2in2*n*q)*dS2\
        + 1.0/dt*(snew-sold)*w*dx - sold*u*grad(w)*dx - Me*2.0*InnerProduct(Sym(Grad(u)),Sym(Grad(u)))*w/theta*dx\
        - Me*lame_ND*div(u)**2*w/theta*dx\
        + 1/Pe_k*KK*(1/(thetaold*theta)*grad(theta))*(1/thetaold*grad(w)-w/(theta*thetaold)*grad(theta))*dx\
        + 1/(Pe_k*wcf)*(1-T_wall/theta)*w*ds("inlet1|inlet2")
        - 1/Pe*Tensorcontraction2(mu,theta,thetaold,w,MM)*dx\
        #+ ap*(sold*uBCI1*n + J1in1*n*mu[0]/theta-(rhoold*Unit*uBCI1*n)*(mu[0]+mu[1])/theta)*w*dS1\
        #+ ap*(sold*uBCI2*n + J2in2*n*mu[0]/theta-(rhoold*Unit*uBCI2*n)*(mu[0]+mu[1])/theta)*w*dS2\
        + sold*u*w*n*ds("outlet")\
        #+ ( free/Pe*Tensorcontraction2_BC(mu,theta,thetaold,w,MM))*ds("outlet", skeleton=True)
        )

    #------Set IC/ compute Metdata-------
    rho.Interpolate(rho0)
    rhoold.Interpolate(rho0)
    theta.Interpolate(theta0)
    u.Interpolate(u0)
    mu.Interpolate(f_rho)

    u.Interpolate(uBCI1, definedon=Th.Boundaries("inlet1"))
    u.Interpolate(uBCI2, definedon=Th.Boundaries("inlet2"))

    El  = [Integrate(TotEnergy(rho,theta,u),Th,VOL)]
    Sl  = [Integrate(Entropie(rho,theta),Th,VOL)]
    Mr  = [Integrate(rho,Th,VOL)] 
    M2 = [Integrate(rho*Unit,Th,VOL)]
    C  = [Integrate(rho*Vvec-1,Th,VOL)]


    #--------Paraview output---------------
    if(paraout):
        dummy.Interpolate(V_1_ND*rho[0] + V_2_ND*rho[1] - 1)
        x1 = (rho[0] / M_1_ND) / ((rho[0] / M_1_ND) + (rho[1] / M_2_ND))
        x2 = (rho[1] / M_2_ND) / ((rho[0] / M_1_ND) + (rho[1] / M_2_ND))

        Z1 = rho[0]*u - 1/Pe*MM[0,0]*(grad(mu)[0,:]/theta-mu[0]/(theta*thetaold)*grad(theta)) - 1/Pe*MM[0,1]*(grad(mu)[1,:]/theta-mu[1]/(theta*thetaold)*grad(theta))
        Z2 = rho[1]*u - 1/Pe*MM[0,1]*(grad(mu)[0,:]/theta-mu[0]/(theta*thetaold)*grad(theta)) - 1/Pe*MM[1,1]*(grad(mu)[1,:]/theta-mu[1]/(theta*thetaold)*grad(theta))
        x12.Interpolate((x1,x2))
        vtk = VTKOutput(ma=Th,coefs=[rho,mu,u,p,theta,dummy,x12,Z1,Z2],names=["rho","mu","u","p","theta","Constraint","x","J1","J2"],filename=outputpath+"Bilder/"+outputname)
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

        uBCI1 = CF(J1in1/(rhoold*Unit))
        uBCI2 = CF(J2in2/(rhoold*Unit))



        u.Interpolate(uBCI1, definedon=Th.Boundaries("inlet1"))
        u.Interpolate(uBCI2, definedon=Th.Boundaries("inlet2"))


        l = 0
        # Newton for CH
        newtonerror = 1. # newtonerror
        while newtonerror > iterror and l<itsteps:
            l+=1
            a.Assemble()
            L.Assemble()				
            solution.vec.data = a.mat.Inverse(FES.FreeDofs())*(L.vec)
            #rhodiff,mudiff,udiff,pdiff,lamdiff,thetadiff = solution.components
            # Update solutions
            rho.vec.data   += rhodiff.vec.data
            mu.vec.data    += mudiff.vec.data
            u.vec.data     += udiff.vec.data
            p.vec.data     += pdiff.vec.data
            lam.vec.data   += lamdiff.vec.data
            theta.vec.data += thetadiff.vec.data

            #n1=sqrt(Integrate(rhodiff**2, Th, VOL))
            #n2=sqrt(Integrate(mudiff**2, Th, VOL))
            #n3=sqrt(Integrate(udiff**2, Th, VOL))
            #n4=sqrt(Integrate(pdiff**2, Th, VOL))
            #n5=sqrt(Integrate(thetadiff**2, Th, VOL))

            #print("Rho",n1)
            #print("mu",n2)
            #print("u",n3)
            #print("p",n4)
            #print("T",n5)

            # Newtonerror
            newtonerror=sqrt(Integrate(rhodiff**2 + mudiff**2 + udiff**2 + pdiff**2 + thetadiff**2, Th, VOL))
            print("Newton error:",newtonerror)

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
            dummy.Interpolate(V_1_ND*rho[0] + V_2_ND*rho[1] - 1)
            x1 = (rho[0] / M_1_ND) / ((rho[0] / M_1_ND) + (rho[1] / M_2_ND))
            x2 = (rho[1] / M_2_ND) / ((rho[0] / M_1_ND) + (rho[1] / M_2_ND))
            Z1 = rho[0]*u - 1/Pe*MM[0,0]*(grad(mu)[0,:]/theta-mu[0]/(theta*thetaold)*grad(theta)) - 1/Pe*MM[0,1]*(grad(mu)[1,:]/theta-mu[1]/(theta*thetaold)*grad(theta))
            Z2 = rho[1]*u - 1/Pe*MM[0,1]*(grad(mu)[0,:]/theta-mu[0]/(theta*thetaold)*grad(theta)) - 1/Pe*MM[1,1]*(grad(mu)[1,:]/theta-mu[1]/(theta*thetaold)*grad(theta))
            x12.Interpolate((x1,x2))
            vtk.Do(time=t)
        

#-------Finished output---------
print("\n\n\nSimulation beendet:")
print("\nn:",n,", dt:",dt,", T:",Ti,", Steps:",steps)
print("\n Newtonueberschreitungen:",itcount," ,",newtonerrs)

#-------Write metdata------------
    
timefile = open(outputpath+outputname+"_timeoutput.txt", "w")  
timefile.write("N: "+str(n)+"\n")
timefile.write("dt: "+str(dt)+"\n")
timefile.write("T: "+str(Ti)+"\n")
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