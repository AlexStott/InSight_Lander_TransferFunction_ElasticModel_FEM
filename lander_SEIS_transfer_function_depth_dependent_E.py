"""
FEM simulation modelling the ground below the NASA InSight lander for Stott et al. (2021) "The Site Tilt and Lander Transfer Function from the Short-Period Seismometer of InSight on Mars" in BSSA.

This code produces displacement ratio map for a perturbation from the lander feet, indicating the transfer to the seismometer's feet.

Uncomment the correct boundary conditions for displacement in vertical, north-south and east-west directions.

Uncomment the correct section to plot the respective vertical, north-south and east-west displacement ratio.

This code was adapted from that of Myhill et al. (2018) "Near-field seismic propagation and coupling through Mars’ regolith: Implications for the InSight mission" 

Modified from the FEniCS tutorial demo program: Linear elastic problem.
https://fenicsproject.org/pub/tutorial/html/._ftut1008.html

The current model is used to approximate the displacement field of an elastic block when it is deformed by small surface loads imposed by circular feet.

"""

from __future__ import print_function
from fenics import *
import dolfin
from dolfin import MPI
from dolfin.cpp.mesh import MeshFunctionBool
from ufl import nabla_div

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
plt.rc('text', usetex=True)



# analytical function from Sneddon (1946)
def homogeneous_surface_displacement(r, r_f, z_0, nu):
    beta = np.abs(np.array([r_f/r_i if np.abs(r_i) > 1.e-10 else 1.e10 for r_i in r]))
    return np.array([[z_0/np.pi*(1. - 2.*nu)/(1. - nu)*b
                      if b < 1
                      else z_0/np.pi*(1. - 2.*nu)/(1. - nu)*(b - np.sqrt(b*b - 1.))
                      for b in beta],
                     [2.*z_0/np.pi*np.arcsin(b)
                      if b < 1
                      else z_0
                      for b in beta]])



# Number of unrefined cells in each dimension
n_W = 32
n_H = 32

# Physical parameters
lander_foot_radius = 0.145 # 0.145 m
lander_foot_displacement = 1.e-3 # m, some small number, this is normalised anyway


# The width and height of the domain (the domain has a square cross-section)
fac = 2.
W = n_W*lander_foot_radius*fac
H = n_H*lander_foot_radius*fac


# Lander feet and SEIS foot locations
lander_foot_centers = np.array([[-1.0638262, -0.59925, H/2.],
                                [1.0483534, -0.6228205, H/2.],
                                [0.0154728, 1.222077, H/2.]])

SEIS_foot_centers = np.array([[-0.525731608071264, -2.2513114189326, H/2.],
                              [-0.55211709490411, -2.46620396437007, H/2.],
                              [-0.725026754920435, -2.33590718976287, H/2.]])


print('Domain dimensions: {0} m, {0} m, {1} m'.format(W, H))


try:
    print('Searching for a preexisting mesh...')
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, 'mesh.xdmf') as xdmf:
        xdmf.read(mesh)
    print('Preexisting mesh found in mesh.xdmf. Using this one.')
    print('WARNING: if you have changed the geometry of the problem '
          'or the mesh resolution,')
    print('you will need to delete the old mesh.xdmf and restart this run.')
except:
    print('Mesh not found. Making a new one.')
    # Create mesh and define function space
    mesh = BoxMesh(Point(-W/2., -W/2., -H/2.),
                   Point(W/2., W/2., H/2.), n_W, n_W, n_H)

    # Refine the mesh close to the centers of the feet
    # Use a generator expression
    # We increase the number of refinement steps closer to the feet
    print('Refining mesh...')
    def refinement_radii(r_max, r_limit):
        i = r_max
        while i >= r_limit:
            yield i
            i /= 2.

    refinement_radii = [r for r in refinement_radii(lander_foot_radius*n_W*fac/2.,
                                                    lander_foot_radius*1.1)]

    n_refine = len(refinement_radii)
    for i in range(n_refine):
        print('Refinement {0}/{1}'.format(i+1, n_refine))
        cell_markers = MeshFunctionBool(mesh, 3, False)
        for foot_center in lander_foot_centers:
            for cell in cells(mesh):
                v = cell.midpoint().array() - foot_center
                dist = np.sqrt(np.dot(v, v))
                if dist < refinement_radii[i]:
                    cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)
    print('Mesh refinement complete.')

    print('Saving mesh to mesh.xdmf...')
    with XDMFFile(MPI.comm_world, 'mesh.xdmf') as xdmf:
        xdmf.write(mesh)


# Get the mesh coordinates
mesh_points = mesh.coordinates()

# Get the surface mask for later
tol = 1e-8
surface_mask = [i for i, coord in enumerate(mesh_points) if np.abs(H/2. - coord[2]) < tol]
x_axis_mask = [i for i, coord in enumerate(mesh_points) if (H/2. - coord[2] < tol and
                                                   np.abs(coord[1]) < tol)]



# Create boundary conditions
print('Imposing boundary conditions...')
def bottom_boundary(x, on_boundary):
    return on_boundary and x[2] < tol - H/2.

def top_boundary_foot_0(x, on_boundary): # foot on boundary
    return on_boundary and x[2] > H/2. - tol and sqrt(np.power(x[0] - lander_foot_centers[0][0], 2.) + np.power(x[1] - lander_foot_centers[0][1], 2.)) < lander_foot_radius + tol
def top_boundary_foot_1(x, on_boundary): # foot on boundary
    return on_boundary and x[2] > H/2. - tol and sqrt(np.power(x[0] - lander_foot_centers[1][0], 2.) + np.power(x[1] - lander_foot_centers[1][1], 2.)) < lander_foot_radius + tol
def top_boundary_foot_2(x, on_boundary): # foot on boundary
    return on_boundary and x[2] > H/2. - tol and sqrt(np.power(x[0] - lander_foot_centers[2][0], 2.) + np.power(x[1] - lander_foot_centers[2][1], 2.)) < lander_foot_radius + tol

# Create a vector function space for the solution
# and a scalar function space for the rheological parameters
V = VectorFunctionSpace(mesh, 'P', 1)
D = FunctionSpace(mesh, "DG", 1) # linear interpolation required

uorig = Function(V)
uorig_norm = np.array([uorig(pt) for pt in mesh_points]) 

# Boundary conditions with no slip base and free slip under feet - uncomment for desired displacement direction

# Displace lander feet in West direction (minus East)
# bcs = [DirichletBC(V, Constant((0, 0, 0)), bottom_boundary),
#       DirichletBC(V.sub(0), Constant(-lander_foot_displacement), top_boundary_foot_0),
#       DirichletBC(V.sub(0), Constant(-lander_foot_displacement), top_boundary_foot_1),
#       DirichletBC(V.sub(0), Constant(-lander_foot_displacement), top_boundary_foot_2)]

# Displace lander feet in South direction (minus North)
# bcs = [DirichletBC(V, Constant((0, 0, 0)), bottom_boundary),
#       DirichletBC(V.sub(1), Constant(-lander_foot_displacement), top_boundary_foot_0),
#       DirichletBC(V.sub(1), Constant(-lander_foot_displacement), top_boundary_foot_1),
#       DirichletBC(V.sub(1), Constant(-lander_foot_displacement), top_boundary_foot_2)]

# Displace lander feet in Vertical direction
bcs = [DirichletBC(V, Constant((0, 0, 0)), bottom_boundary),
      DirichletBC(V.sub(2), Constant(-lander_foot_displacement), top_boundary_foot_0),
      DirichletBC(V.sub(2), Constant(-lander_foot_displacement), top_boundary_foot_1),
      DirichletBC(V.sub(2), Constant(-lander_foot_displacement), top_boundary_foot_2)]



# Define regolith rheology
print('Defining regolith rheology...')

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, d):
    return lmda*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)


nu = 0.25 # Poisson ratio is 0.25 for an isotropic elastic solid
f_lambda = nu/((1. + nu)*(1. - 2.*nu)) # scaling factor to convert E to lambda
f_mu = 1./(2.*(1.+nu)) # scaling factor to convert Young's modulus to shear modulus


# Power law Young's modulus profile
E_0 = 1. # unimportant
c = 1.

#Set 
k = 0.9
b = 10
# youngs_modulus_type = '$E = E_0(b + (cz/f_r))^k$; $E_0$ = {0:.2f}, $b$ = {1:.2f}, $c$ = {2:.2f}, $k$ = {3:.2f}'.format(E_0, b, c, k)
youngs_modulus_type = '$b$ = {0:.2f},  $k$ = {1:.2f}'.format(b, k)

youngs_modulus_expression = "E_0*pow(b + c*(H/2. - x[2])/f_r, k)"
E = interpolate(Expression(youngs_modulus_expression,
                           c=c, b=b, k=k, E_0=E_0, f_r=lander_foot_radius, H=H, degree=3), D)

func_mu = Expression("f_mu*"+youngs_modulus_expression,
                     f_mu=f_mu, b=b, c=c, k=k, E_0=E_0, f_r=lander_foot_radius, H=H, degree=3)

func_lmda  = Expression("f_lambda*"+youngs_modulus_expression,
                        f_lambda=f_lambda,
                        c=c, b=b, k=k, E_0=E_0, f_r=lander_foot_radius, H=H, degree=3)



# Interpolate Lame parameters over function space
mu = interpolate(func_mu, D)
lmda = interpolate(func_lmda, D)

# Define the variational problem
print('Defining the variational problem...')
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)


# f = Constant((0, 0, -rho*g)) # include gravity
f = Constant((0, 0, 0.)) # ignore gravity

T = Constant((0, 0, 0))
a = inner(sigma(u, d), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds
#L = dot(f, v)*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs, solver_parameters={'linear_solver': 'gmres',
                                         'preconditioner': 'ilu'})


# NORMALISED displacement of the surface
u_norm = np.array([u(pt) for pt in mesh_points]) #/ lander_foot_displacement


print('Plotting the solution')
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                       figsize=(9,5))

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

x = mesh_points[surface_mask,0]
y = mesh_points[surface_mask,1]

# To plot the vertical displacement relative to initial displacement
log_disp = np.log10(np.abs((u_norm[surface_mask,2]) / lander_foot_displacement ) ) 

# To plot the east-west displacement relative to initial displacement
# log_disp = np.log10(np.abs((u_norm[surface_mask,0]) / lander_foot_displacement ) ) 

# To plot the north-south displacement relative to initial displacement
# log_disp = np.log10(np.abs((u_norm[surface_mask,1]) / lander_foot_displacement ) ) 


#To calculate the relative displacement at each of the SEIS feet in each axis

tol = 3e-2
for SEIS_foot in SEIS_foot_centers:
  FootLoc = [i for i, coord in enumerate(mesh_points) if (np.abs((SEIS_foot[0] - coord[0]))+np.abs(SEIS_foot[1] - coord[1])+np.abs((SEIS_foot[2] - coord[2]))) < tol]
  print('Deformation for foot location:')
  print(SEIS_foot)
  Num = np.log10(np.abs((u_norm[FootLoc]) / lander_foot_displacement ) )
  for elementNum in  Num:
    print('East ratio')
    print(elementNum[0])
    print('North ratio')
    print(elementNum[1])
    print('Vertical ratio')
    print(elementNum[2])


ax[0].set_title(youngs_modulus_type)
ax[0].tricontour(x, y, log_disp, levels=14, linewidths=1, colors='k')
lim_levels = np.arange(-28,0)*0.1
contour_function = ax[0].tricontourf(x, y, log_disp, levels=lim_levels, cmap=cc.cm.rainbow) #colorwheel


cbar = fig.colorbar(contour_function, ax=ax[0])
cbar.set_label('$\\log_{10} (w/w_{0})$', fontsize=20 , rotation=270, labelpad=20) # labelpad shifts the label right

# SEIS feet have pad radii of 3 cm
for SEIS_foot in SEIS_foot_centers:
    ax[0].add_artist(plt.Circle(SEIS_foot[0:2], 0.03, color='black'))
# lander feet have pad radii of 14.5 cm
for lander_foot in lander_foot_centers:
    ax[0].add_artist(plt.Circle(lander_foot[0:2], 0.145, color='black'))


ax[0].set_xlabel('E (m)', fontsize=20)
ax[0].set_ylabel('N (m)', fontsize=20)
for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
    item.set_fontsize(20)

for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
    item.set_fontsize(20)

hs = np.linspace(H/2., -H/2., 101)
Es = np.empty_like(hs)
for i in range(len(hs)):
    Es[i] = E([0., 0., hs[i]])

ax[1].plot(Es, (hs - H/2.))
ax[1].set_xlim(0.,)
ax[1].set_ylim(-H, 0.)
# ax[1].set_xlabel('Youngs modulus (non-dimensional)')
ax[1].set_ylabel('z (m)')

fig.tight_layout()


fig.savefig('transfer_function_map_b10kpt9_vert_test.pdf')

plt.show()
