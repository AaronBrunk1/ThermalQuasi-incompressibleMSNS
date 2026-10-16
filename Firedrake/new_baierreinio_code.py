import sys
import numpy as np
from firedrake import *
import netgen.occ as ngocc
from irksome import *
# NOTE: This script requires using an Irksome branch with EquationBC support:
#       See https://github.com/firedrakeproject/Irksome/tree/baierreinio/galerkin_in_time_equationbc

# The polynomial order we will use for our mesh and finite element spaces
k = 2                           # The polynomial degree (for the velocity spaces)
maxh = 0.05                     # Maximum cell diameter for mesh
deg_max = 15                    # Maximum quadrature degree

newton_atol = 1e-11				# Newton absolute tolerance
newton_rtol = 0            	    # Newton relative tolerance
newton_max_it = 15              # Newton maximum number of iterations

ksp_atol = 1e-13				# PETSc KSP absolute tolerance
ksp_rtol = 1e-13				# PETSc KSP relative tolerance

T_final_ND = 5				    # The final time (non-dimensionalized), dimensionless
N_timesteps = 1000              # The number of time-steps
N_timesteps_output = 1			# After how many time-steps to output to PVD

outfolder = sys.argv[1]
outfile = VTKFile(f"{outfolder}/sln.pvd")	# The solution output file
midpoint_outfile = VTKFile(f"{outfolder}/sln_midpoint.pvd")

# Specify the domain Omega (i.e. the microfluidic container's geometry)
wp = ngocc.WorkPlane()
wp.MoveTo(0, 1)
wp.Spline([ngocc.Pnt(1, 0), ngocc.Pnt(0, -1)])
wp.LineTo(1, -1)
wp.Spline([ngocc.Pnt(3, -0.5), ngocc.Pnt(4, -0.5)], tangents={ 1 : ngocc.gp_Vec2d(1, 0) })
wp.LineTo(4, 0.5)
wp.Spline([ngocc.Pnt(3, 0.5), ngocc.Pnt(1, 1)], tangents={ 0 : ngocc.gp_Vec2d(-1, 0) })
wp.LineTo(0, 1)

Omega = wp.Face()

# Label the inlets and outlet; the other walls are unlabelled
Omega.edges.Max(ngocc.Y).name = "inlet_1"
Omega.edges.Min(ngocc.Y).name = "inlet_2"
Omega.edges.Max(ngocc.X).name = "outlet"

# Construct the mesh
ngmesh = ngocc.OCCGeometry(Omega, dim=2).GenerateMesh(maxh=maxh)
mesh = Mesh(ngmesh)

N = FacetNormal(mesh)

# Get the IDs of the inlets, outlet and walls
inlet_1_id = tuple(i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "inlet_1")
inlet_2_id = tuple(i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "inlet_2")
outlet_id = tuple(i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "outlet")
walls_ids = tuple(i+1 for i, name in enumerate(ngmesh.GetRegionNames(dim=1)) if name == "default")

assert(len(inlet_1_id) == 1)
assert(len(inlet_2_id) == 1)
assert(len(outlet_id) == 1)
assert(len(walls_ids) == 3)

# Define the surface and volume measures using a fixed quadrature degree
ds = ds(mesh, degree=deg_max)
dx = dx(mesh, degree=deg_max)

# Define the spatial coordinates on the mesh
x_sc, y_sc = SpatialCoordinate(mesh)

# UFL objects for time dependence
MC = MeshConstant(mesh)
t_ND = MC.Constant(0.0)                        		# Current non-dimensionalized time, dimensionless
dt_ND = MC.Constant(T_final_ND / N_timesteps)     	# Non-dimensionalized timestep, dimensionless

# The finite element spaces
X_h = VectorFunctionSpace(mesh, "CG", k)            # Space for vector fields
V_h = FunctionSpace(mesh, "CG", k - 1)              # Space for scalar fields
Q_h = V_h                                           # Space for zero-mean scalar fields
L_h = FunctionSpace(mesh, "R", 0)                   # Lagrange multiplier space

# The mixed space (rho_1, rho_2, m, rho_s, mu_mass_1, mu_mass_2, u, p, vtheta)
Z_h = V_h * V_h * X_h * V_h * V_h * V_h * X_h * Q_h * V_h * L_h
PETSc.Sys.Print(f"Mesh has {mesh.num_cells()} cells, with {Z_h.dim()} finite element DOFs")

num_stages = 1              # Number of RK stages
num_fields = 10             # Number of fields in Z_h
num_lagrange_mults = 1      # Number of Lagrange multipliers in Z_h

# The discrete solution
# NOTE: By convention, mu's without an _mass, shall refer to mole-based chemical potential quantities
solution = Function(Z_h)
rho_1, rho_2, m, rho_s, mu_mass_1, mu_mass_2, u, p, vtheta, l1 = split(solution)

# The test functions
psi_1_t, psi_2_t, w_t, omega_t, zeta_1_t, zeta_2_t, v_t, q_t, gamma_t, l1_t = TestFunctions(Z_h)

# Set the Lagrange multiplier to zero; it is never used
lagrange_mult_terms = inner(l1, l1_t) * dx

# The (dimensional) Stokes viscosities
eta = Constant(6e-4)            # Shear viscosity, Pa s
zeta = Constant(1e-7)           # Bulk viscosity, Pa s

# Non-dimensionalised Lamé parameter, dimensionless
lame_ND = (zeta / eta) - 1.0

# Navier-Stokes reference quantities used for non-dimensionalisation
v_ref = Constant(0.0)           # Reference velocity (to be specified later), m / s
L_ref = Constant(2e-3)          # Reference length, m
p_ref = eta * v_ref / L_ref     # Reference pressure, Pa

# Constants for the pure species (at the ambient temperature and pressure)
M_1 = Constant(0.078)               # Molar mass of benzene, kg / mol
M_2 = Constant(0.084)               # Molar mass of cyclohexane, kg / mol
rho_pure_1 = Constant(876)          # Density of pure benzene, kg / m^3
rho_pure_2 = Constant(773)          # Density of pure cyclohexane, kg / m^3
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

# Non-dimensionalised molar masses
M_1_ND = M_1 / M_ref
M_2_ND = M_2 / M_ref

# Non-dimensionalised partial specific volumes
V_1_ND = rho_ref / rho_pure_1
V_2_ND = rho_ref / rho_pure_2

# Free energy of the mixture
T_ref = Constant(298.15)        # Reference temperature, K
R = Constant(8.314)             # Ideal gas constant, J / K * mol

mu_ref = R * T_ref              # Reference chemical potential, J / mol

Me_1 = p_ref / (mu_ref * c_pure_1)      # Non-dimensionalised partial molar volume of benzene, dimensionless
Me_2 = p_ref / (mu_ref * c_pure_2)      # Non-dimensionalised partial molar volume of cyclohexane, dimensionless

# Margules model parameters
A_12 = Constant(0.4498)         # Dimensionless
A_21 = Constant(0.4952)         # Dimensionless

# Partial molar heat capacities of the species
Cp_1 = Constant(135.69)         # Of Benzene, J / K * mol
Cp_2 = Constant(156.00)         # Of Cyclohexane, J / K * mol

# Margules model for chemical potentials, assuming constant partial molar volumes, and constant partial molar heat capacities
def mu_relation(x_1, x_2, p, T):
    mu_1_r = (Me_1 * p) + (T * (ln(x_1) + (x_2 ** 2) * (A_12 + (2.0 * (A_21 - A_12) * x_1)))) + ((Cp_1 / R) * (T - (T * ln(T))))
    mu_2_r = (Me_2 * p) + (T * (ln(x_2) + (x_1 ** 2) * (A_21 + (2.0 * (A_12 - A_21) * x_2)))) + ((Cp_2 / R) * (T - (T * ln(T))))

    return (mu_1_r, mu_2_r)

# Formula for the mole fractions from the partial densities
x_1 = (rho_1 / M_1_ND) / ((rho_1 / M_1_ND) + (rho_2 / M_2_ND))
x_2 = (rho_2 / M_2_ND) / ((rho_1 / M_1_ND) + (rho_2 / M_2_ND))

# Enforce the L^2-projection of the chemical potential relationship
g_1, g_2 = mu_relation(x_1, x_2, p, vtheta)
gibbs_terms = (inner(mu_mass_1 - (g_1 / M_1_ND), zeta_1_t) + inner(mu_mass_2 - (g_2 / M_2_ND), zeta_2_t)) * dx

# Form the OSM mobility matrix
D_sm = Constant(2.1e-9)                         # Stefan--Maxwell diffusivity, m^2 / s
Pe = v_ref * L_ref / D_sm                       # Péclet number, dimensionless

M_OSM_factor = (rho_1 * rho_2 * M_1_ND * M_2_ND) \
               / ((rho_1 + rho_2) * ((M_1_ND * x_1) + (M_2_ND * x_2)))

M_OSM = M_OSM_factor * as_matrix([[1, -1],
                                  [-1, 1]])     # Dimensionless OSM mobility matrix

# Dimensionless diffusional mass fluxes
J_1_d = -((M_OSM[0, 0] * grad(mu_mass_1 / vtheta)) + (M_OSM[0, 1] * grad(mu_mass_2 / vtheta)))
J_2_d = -((M_OSM[1, 0] * grad(mu_mass_1 / vtheta)) + (M_OSM[1, 1] * grad(mu_mass_2 / vtheta)))

# Mass continuity equations
mc_terms = inner(Dt(rho_1), psi_1_t) * dx + inner((rho_1 * u) + (J_1_d / Pe), -grad(psi_1_t)) * dx \
         + inner(Dt(rho_2), psi_2_t) * dx + inner((rho_2 * u) + (J_2_d / Pe), -grad(psi_2_t)) * dx

# Weakly imposed boundary conditions on the mass fluxes
v_ref_1 = Constant(0.4e-6)                      # Reference inflow velocity of benzene, m / s
v_ref_2 = (c_pure_1 / c_pure_2) * v_ref_1       # Reference inflow velocity of cyclohexane, m / s
v_ref.assign(0.5 * (v_ref_1 + v_ref_2))         # Reference velocity, m / s

parabola_inflow_1 = 2.0 * x_sc * (x_sc - 1.0) * as_vector([2.0, -1.0])
parabola_inflow_2 = 2.0 * x_sc * (x_sc - 1.0) * as_vector([2.0, 1.0])
G_1_inflow_bc_func = -M_1_ND * (v_ref_1 / v_ref) * (c_pure_1 / c_ref) * parabola_inflow_1# * tanh(10 * t_ND)
G_2_inflow_bc_func = -M_2_ND * (v_ref_2 / v_ref) * (c_pure_2 / c_ref) * parabola_inflow_2# * tanh(10 * t_ND)
# NOTE: Omitting tanh makes the linear-in-time part of the solution checkerboard, midpoint evaluation should then be used!

nml = FacetNormal(mesh)

mc_terms += inner(G_1_inflow_bc_func, nml * psi_1_t) * ds(inlet_1_id) \
            + inner(G_2_inflow_bc_func, nml * psi_2_t) * ds(inlet_2_id) \
            + inner(rho_1 * u, nml * psi_1_t) * ds(outlet_id) \
            + inner(rho_2 * u, nml * psi_2_t) * ds(outlet_id) \

# Equation for div(u)
div_u_terms = -inner(u, grad(q_t)) * dx \
              - (1 / Pe) * inner((V_1_ND * J_1_d) + (V_2_ND * J_2_d), grad(q_t)) * dx

div_u_terms += inner(V_1_ND * G_1_inflow_bc_func, nml * q_t) * ds(inlet_1_id) \
               + inner(V_2_ND * G_2_inflow_bc_func, nml * q_t) * ds(inlet_2_id) \
               + inner(u, nml * q_t) * ds(outlet_id)

# Equation for the temperature
rho_s_ref = R * c_ref       # Reference volumetric entropy

ln_T_expr = rho_s \
            + ((rho_1 / M_1_ND) * (ln(x_1) + (x_2 ** 2) * (A_12 + (2.0 * (A_21 - A_12) * x_1)))) \
            + ((rho_2 / M_2_ND) * (ln(x_2) + (x_1 ** 2) * (A_21 + (2.0 * (A_12 - A_21) * x_2))))
ln_T_expr = ln_T_expr / (((rho_1 / M_1_ND) * (Cp_1 / R)) + ((rho_2 / M_2_ND) * (Cp_2 / R)))

vtheta_terms = inner(vtheta - exp(ln_T_expr), gamma_t) * dx     # vtheta, as a function of rho_s, rho_1 and rho_2

# Heat transport quantities
k_1 = Constant(0.14)                            # Thermal conductivity of benzene, W / m * K
k_2 = Constant(0.11)                            # Thermal conductivity of cyclohexane, W / m * K
k_tc = ((k_1 * x_1) + (k_2 * x_2))              # Basic molar mixing rule for thermal conductivity, W / m * K
k_ref = Constant(0.125)                         # Reference thermal conductivity, W / m * K

Pe_k = v_ref * L_ref * (c_ref * R / k_ref)      # Thermal Peclet number, dimensionless

# Entropy equation terms
Me = p_ref / (R * T_ref * c_ref)                # Pressure diffusion number, dimensionless

entropy_prod_visc = (2.0 * inner(sym(grad(u)), sym(grad(u))) + lame_ND * inner(div(u), div(u))) * (omega_t / vtheta)    # Viscous entropy production
entropy_prod_therm = inner((k_tc / k_ref) * (vtheta ** 2) * grad(1 / vtheta), grad(omega_t / vtheta))       # Thermal entropy production
entropy_prod_diff = inner(-J_1_d, grad(mu_mass_1 * omega_t / vtheta)) \
                  + inner(-J_2_d, grad(mu_mass_2 * omega_t / vtheta))                                       # Diffusive entropy production

entropy_terms = inner(Dt(rho_s), omega_t) * dx - inner(rho_s * u, grad(omega_t)) * dx \
              - Me * entropy_prod_visc * dx \
              - (1 / Pe_k) * entropy_prod_therm * dx \
              - (1 / Pe) * entropy_prod_diff * dx

# Robin BCs which express the finite thermal conductivity of the inlets
# We assume the inlets are 100x thinner than the reference length, and 10x more thermally insulating than the mixture

wall_thickness = L_ref * Constant(0.01)                 # Thickness of the inlets, m
wall_tc = k_ref * Constant(0.1)                         # Inlet thermal conductivity, W / m * K
wcf = (wall_thickness / L_ref) * (k_ref / wall_tc)

T_wall = 1 + (0.1 * (((y_sc + 1) / 2)))

entropy_terms += (1 / (Pe_k * wcf)) * inner(1 - (T_wall / vtheta), omega_t) * ds(inlet_1_id + inlet_2_id)
entropy_terms += inner(rho_s * u, nml * omega_t) * ds(outlet_id + walls_ids)    # Do-nothing BCs on walls and outlets

# Auxiliary m-field terms
m_field_terms = inner(m - (sqrt(rho_1 + rho_2) * u), w_t) * dx

# Constants relevant to the transient problem
t_ref = L_ref / v_ref                           # Reference time, s
Re = (rho_ref * v_ref * L_ref) / eta            # Reynolds number, dimensionless

# Navier-Stokes terms
ns_terms = Re * inner(sqrt(rho_1 + rho_2) * Dt(m), v_t) * dx
ns_terms += 0.5 * Re * inner(grad(u), (rho_1 + rho_2) * outer(u, v_t)) * dx
ns_terms -= 0.5 * Re * inner(grad(v_t), (rho_1 + rho_2) * outer(u, u)) * dx
ns_terms += 2.0 * inner(sym(grad(u)), sym(grad(v_t))) * dx
ns_terms += lame_ND * inner(div(u), div(v_t)) * dx
ns_terms -= inner(p, div(v_t)) * dx
ns_terms += 0.5 * Re * (rho_1 + rho_2) * max_value(dot(u, nml), 0.0) * inner(u, v_t) * ds(outlet_id)

# Altogether, our total residual is the sum of forms built above
tot_res = lagrange_mult_terms \
        + gibbs_terms \
        + mc_terms \
        + div_u_terms \
        + vtheta_terms \
        + entropy_terms \
        + m_field_terms \
        + ns_terms

# Make note of dimensionless numbers
PETSc.Sys.Print(f"Re is {float(Re)}, Pe is {float(Pe)}, Pe_k is {float(Pe_k)}, Me is {float(Me)}, t_ref is {float(t_ref)}")

# Boundary conditions on the barycentric velocity are enforced via EquationBC
rho_v_inflow_1_bc_func = G_1_inflow_bc_func
rho_v_inflow_2_bc_func = G_2_inflow_bc_func

bc_data = {inlet_1_id: rho_v_inflow_1_bc_func, inlet_2_id: rho_v_inflow_2_bc_func}
F_bc = sum(inner(u - ((1 / (rho_1 + rho_2)) * flux), v_t) * ds(*subdomain) for subdomain, flux in bc_data.items())
ridges = [(i, j) for i in [*inlet_1_id, *inlet_2_id] for j in walls_ids]
sub_bcs = [DirichletBC(Z_h.sub(6), 0, ridges)]  # bc on the boundary of the boundary
v_bc_mavg = EquationBC(F_bc == 0, solution, (*inlet_1_id, *inlet_2_id), V=Z_h.sub(6), bcs=sub_bcs)
v_bc_zero = DirichletBC(Z_h.sub(6), 0, walls_ids)

bcs = [v_bc_mavg, v_bc_zero]

# We provide an initial guess based on an equimolar distribution at temperature T_ref
rho_1, rho_2, m, rho_s, mu_mass_1, mu_mass_2, u, p, vtheta, l1 = solution.subfunctions

x_1_init = Constant(0.5)
x_2_init = 1 - x_1_init
c_1_init = x_1_init * (c_equi_tot / c_ref)
rho_1_init = M_1_ND * c_1_init
rho_2_init = (1 - (rho_1_init * V_1_ND)) / V_2_ND

rho_1.interpolate(rho_1_init)
rho_2.interpolate(rho_2_init)

vtheta_init = 1
vtheta.interpolate(vtheta_init)

S_1_init = (-(ln(x_1_init) + (x_2_init ** 2) * (A_12 + (2.0 * (A_21 - A_12) * x_1_init))) - ((Cp_1 / R) * (-ln(vtheta_init))))  # Partial molar entropy
S_2_init = (-(ln(x_2_init) + (x_1_init ** 2) * (A_21 + (2.0 * (A_12 - A_21) * x_2_init))) - ((Cp_2 / R) * (-ln(vtheta_init))))  # Partial molar entropy

rho_s.interpolate(((rho_1 / M_1_ND) * S_1_init) + ((rho_2 / M_2_ND) * S_2_init))    # Initial volumetric entropy

# Set up the time-stepper
fieldsplit_0_fields = []
fieldsplit_1_fields = []

for stage in range(0, num_stages):
    fieldsplit_0_fields += range(0 + (stage * num_fields), num_fields - num_lagrange_mults + (stage * num_fields))
    fieldsplit_1_fields += range(num_fields - num_lagrange_mults + (stage * num_fields), num_fields + (stage * num_fields))

fieldsplit_0_fields_str = str(fieldsplit_0_fields)[1:-1].replace(' ', '')
fieldsplit_1_fields_str = str(fieldsplit_1_fields)[1:-1].replace(' ', '')

newton_solver_parameters = {"snes_type" : "newtonls",
							 	"snes_linesearch_type" : "l2",
								"snes_monitor" : "",
								"snes_atol" : newton_atol,
								"snes_stol" : 0.0,
								"snes_rtol" : newton_rtol,
								"snes_max_it" : newton_max_it,
								"mat_type" : "matfree",
								"ksp_type" : "fgmres",
								"ksp_gmres_cgs_refinement_type" : "refine_always",
								"ksp_atol" : ksp_atol,
								"ksp_rtol" : ksp_rtol,
								"ksp_monitor" : "",
								"ksp_converged_reason" : "",
								"pc_type": "fieldsplit",
								"pc_fieldsplit_type": "schur",
								"pc_fieldsplit_schur_fact_type": "full",
								"pc_fieldsplit_0_fields": fieldsplit_0_fields_str,
								"pc_fieldsplit_1_fields" : fieldsplit_1_fields_str,
								"fieldsplit_0": {
									"ksp_type": "preonly",
									"pc_type": "python",
									"pc_python_type": "firedrake.AssembledPC",
									"assembled": {
										"ksp_type": "gmres",
										"ksp_max_it" : 3,
										"ksp_atol" : ksp_atol,
										"ksp_rtol" : ksp_rtol,
										"pc_type" : "lu",
										"pc_factor_mat_solver_type" : "mumps",
										"mat_mumps_icntl_14" : 120
									}
								},
								"fieldsplit_1" : {
									"ksp_type" : "gmres",
									"ksp_max_it" : num_lagrange_mults,
									"ksp_atol" : ksp_atol,
									"ksp_rtol" : ksp_rtol,
								}}

scheme = "cpg"      # What scheme to use
timedeg = 1         # What degree in time

scheme_dict = {
    "cpg"   : ContinuousPetrovGalerkinScheme(timedeg, quadrature_degree=15, basis_type="integral"),
    "gauss" : GaussLegendre(timedeg),
    "radau" : RadauIIA(timedeg)
}
if scheme == "cpg":
    stepper = TimeStepper(
        tot_res, scheme_dict[scheme], t_ND, dt_ND, solution,
        solver_parameters=newton_solver_parameters,
        bcs=bcs,
        aux_indices=[4, 5, 6, 7, 8, 9],
        sample_points=[0.5])
else:
    stepper = TimeStepper(
        tot_res, scheme_dict[scheme], t_ND, dt_ND, solution,
        bcs=bcs,
        solver_parameters=newton_solver_parameters,
        sample_points=[0.5])

midpoint_solution = Function(Z_h)

# Output numpy data
numpy_out = np.empty((N_timesteps + 1, 4))

# Timestepping loop
for ts in range(0, N_timesteps + 1):

    rho_1, rho_2, m, rho_s, mu_mass_1, mu_mass_2, u, p, vtheta, l1 = solution.subfunctions
    rho_1_mp, rho_2_mp, m_mp, rho_s_mp, mu_mass_1_mp, mu_mass_2_mp, u_mp, p_mp, vtheta_mp, l1_mp = midpoint_solution.subfunctions

    # Output total energy, entropy, and masses
    e_kin_ND = (1 / 2) * dot(m, m)      # Kinetic energy density (dimensionless)
    e_kin = e_kin_ND * rho_ref * (v_ref ** 2)

    x_1 = (rho_1 / M_1_ND) / ((rho_1 / M_1_ND) + (rho_2 / M_2_ND))
    x_2 = (rho_2 / M_2_ND) / ((rho_1 / M_1_ND) + (rho_2 / M_2_ND))

    ln_T_expr = rho_s \
            + ((rho_1 / M_1_ND) * (ln(x_1) + (x_2 ** 2) * (A_12 + (2.0 * (A_21 - A_12) * x_1)))) \
            + ((rho_2 / M_2_ND) * (ln(x_2) + (x_1 ** 2) * (A_21 + (2.0 * (A_12 - A_21) * x_2))))
    ln_T_expr = ln_T_expr / (((rho_1 / M_1_ND) * (Cp_1 / R)) + ((rho_2 / M_2_ND) * (Cp_2 / R)))

    e_int_ND = exp(ln_T_expr) * (((rho_1 / M_1_ND) * (Cp_1 / R)) + ((rho_2 / M_2_ND) * (Cp_2 / R)))    # Internal energy density (dimensionless)
    e_int = e_int_ND * R * c_ref * T_ref

    e_tot = float(assemble((e_kin + e_int) * (L_ref ** 2) * dx))            # Total energy
    s_tot = float(assemble(rho_s * rho_s_ref * (L_ref ** 2) * dx))          # Total entropy

    m1_tot = assemble(rho_1 * rho_ref * (L_ref ** 2) * dx)                  # Total mass of species 1
    m2_tot = assemble(rho_2 * rho_ref * (L_ref ** 2) * dx)                  # Total mass of species 2

    PETSc.Sys.Print(f"Total energy={e_tot} and entropy={s_tot}")
    PETSc.Sys.Print(f"Total masses {m1_tot} and {m2_tot}")

    numpy_out[ts, 0] = e_tot
    numpy_out[ts, 1] = s_tot
    numpy_out[ts, 2] = m1_tot
    numpy_out[ts, 3] = m2_tot

    if ts == N_timesteps:
        break

    # Do a timestep
    PETSc.Sys.Print(f"Doing time-step ts={ts} of {N_timesteps}.")
    
    stepper.advance()
    t_ND.assign((ts + 1) * float(dt_ND))

    midpoint_solution.assign(stepper.sample_values[0])
    if scheme == "cpg":
        for field in [4, 5, 6, 7, 8, 9]:
            midpoint_solution.subfunctions[field].assign(solution.subfunctions[field])

    PETSc.Sys.Print("Done time-step.")

    # Compute the density constraint, to output to Paraview
    DG_out = FunctionSpace(mesh, "DG", k)
    rho_constraint = Function(DG_out, name="rho_constraint").interpolate(((rho_1 * V_1_ND) + (rho_2 * V_2_ND)) - 1)

    # Compute the mass fluxes, to output to Paraview
    M_OSM_factor = (rho_1 * rho_2 * M_1_ND * M_2_ND) \
               / ((rho_1 + rho_2) * ((M_1_ND * x_1) + (M_2_ND * x_2)))

    M_OSM = M_OSM_factor * as_matrix([[1, -1],
                                    [-1, 1]])

    J_1_d = -((M_OSM[0, 0] * grad(mu_mass_1 / vtheta)) + (M_OSM[0, 1] * grad(mu_mass_2 / vtheta)))
    J_2_d = -((M_OSM[1, 0] * grad(mu_mass_1 / vtheta)) + (M_OSM[1, 1] * grad(mu_mass_2 / vtheta)))

    DG_out_vec = VectorFunctionSpace(mesh, "DG", k)
    J_1_out = Function(DG_out_vec, name="J_1").interpolate(((rho_1 * u) + (J_1_d / Pe)) * (rho_ref * v_ref))
    J_2_out = Function(DG_out_vec, name="J_2").interpolate(((rho_2 * u) + (J_2_d / Pe)) * (rho_ref * v_ref))

    # Keeping track of the error between m and sqrt(rho_1 + rho_2) * u, at the endpoint and midpoint
    m_u_diff = m - (sqrt(rho_1 + rho_2) * u)
    m_u_err = sqrt(assemble(dot(m_u_diff, m_u_diff) * dx))

    m_u_diff_mp = m_mp - (sqrt(rho_1_mp + rho_2_mp) * u_mp)
    m_u_err_mp = sqrt(assemble(dot(m_u_diff_mp, m_u_diff_mp) * dx))

    # Print errors
    PETSc.Sys.Print(f"m_u_err is {m_u_err:.2e} at endpoint and {m_u_err_mp} at midpoint")
    PETSc.Sys.Print(f"Constraint error is {sqrt(assemble((rho_constraint ** 2) * dx)):.2e}")

    # Output to Paraview
    names = ["rho_1", "rho_2", "m", "rho_s", "mu_mass_1", "mu_mass_2", "u", "p", "vtheta", "l1"]
    for field, name in enumerate(names):
        solution.subfunctions[field].rename(name)
        midpoint_solution.subfunctions[field].rename(name)

    outfile.write(*solution.subfunctions[0:-num_lagrange_mults], rho_constraint, J_1_out, J_2_out, time=ts)
    midpoint_outfile.write(*midpoint_solution.subfunctions[0:-num_lagrange_mults], time=ts)

np.save(f"{outfolder}/out.npy", numpy_out, allow_pickle=False)